using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>
/// FireRedASR-LLM (FireRed Team, 2025): the Encoder-Adapter-LLM recognizer of FireRedASR, which feeds the FireRedASR-AED
/// Conformer encoder's output through an adapter into Qwen2-7B-Instruct.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> FireRedASR-LLM listens with the encoder of FireRedASR's encoder-decoder model, shrinks its
/// output to half the frames, and hands it to a large language model (Qwen2-7B-Instruct) as part of a chat message that
/// asks for a transcript. The language model then writes the transcript.</para>
/// <para>
/// §2.2 and Figure 1 of the paper: the encoder is initialized from FireRedASR-AED's; the adapter splices adjacent frames
/// and applies a linear layer with ReLU and a linear layer into the LLM's embedding space; the LLM is Qwen2-7B-Instruct.
/// The release (<c>fireredasr_llm.py</c>, <c>adapter.py</c>, <c>llm_tokenizer.py</c>) fixes the details: two frames per
/// adapter step with the remainder discarded; the prompt <c>&lt;|im_start|&gt;user\n&lt;speech&gt;请转写音频为文字
/// &lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c> with the adapter's output in place of <c>&lt;speech&gt;</c>; and
/// LoRA of rank 64 (alpha 16, dropout 0.05) on every attention and feed-forward projection. Training minimizes the
/// cross-entropy over the transcript only. The paper trains the encoder, the adapter and the LoRA adapters with the LLM
/// fixed (<see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>); a model built here starts from random weights, so
/// the default updates every parameter. The paper states no optimizer; AdamW with the library's defaults is used.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 160000, outputSize: 152064);
/// var model = new FireRedASRLLM&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration", "https://arxiv.org/abs/2501.14350", Year = 2025, Authors = "FireRed Team")]
public partial class FireRedASRLLM<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly FireRedASRLLMOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private FireRedAsrLlmNetwork<T>? _network;
    internal FireRedAsrLlmNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public FireRedASRLLM(NeuralNetworkArchitecture<T> architecture, string modelPath, FireRedASRLLMOptions? options = null) : base(architecture)
    {
        _options = options ?? new FireRedASRLLMOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public FireRedASRLLM(NeuralNetworkArchitecture<T> architecture, FireRedASRLLMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new FireRedASRLLMOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>Transcribes audio: the LLM's greedy answer to the transcription prompt, at most as many tokens as the
    /// adapter emits frames (the release's <c>max_new_tokens</c>) and <see cref="FireRedASRLLMOptions.MaxTextLength"/>.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for FireRedASR-LLM.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var speech = network.Speech(Samples(audio), training: false);
            tokens = network.Greedy(network.Prompt(speech, _options.Instruction), Math.Min(_options.MaxTextLength, Math.Max(1, speech.Shape[0])));
        }
        return new TranscriptionResult<T>
        {
            Text = Text(network, tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("FireRedASR-LLM does not support streaming.");

    // Qwen's tokenizer decodes when the vocabulary is Qwen's; otherwise each token is its vocabulary entry or code point.
    private string Text(FireRedAsrLlmNetwork<T> network, int[] tokens)
    {
        if (network.UsesQwenVocabulary) return network.Tokenizer.Decode(tokens).Trim();
        var vocabulary = _options.Vocabulary;
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (vocabulary is not null && t < vocabulary.Length) sb.Append(vocabulary[t]);
            else if (t <= 0x10FFFF && (t < 0xD800 || t > 0xDFFF)) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The encoder, adapter and LLM (see the class remarks). It used to be a generic LLM-ASR template.
        var o = _options;
        if (o.EncoderDim <= 0 || o.EncoderFeedForwardDim % o.EncoderDim != 0)
            throw new ArgumentException("The encoder's feed-forward width must be a multiple of its width.", nameof(Options));
        var layers = new List<LayerBase<T>>();
        _network = new FireRedAsrLlmNetwork<T>(Engine, new ConformerAsrConfiguration
        {
            SampleRate = o.SampleRate, NumMels = o.NumMels, EncoderDim = o.EncoderDim, NumEncoderLayers = o.NumEncoderLayers,
            NumAttentionHeads = o.NumAttentionHeads, FeedForwardMultiplier = o.EncoderFeedForwardDim / o.EncoderDim, ConvKernel = o.ConvKernel,
            RopeTheta = o.RopeTheta, Conv4xChannels = 32, VocabSize = 1,
        }, o.AdapterFrameSplicingFactor, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.LlmDim, Layers = o.NumLlmLayers, Heads = o.NumLlmAttentionHeads, KeyValueHeads = o.NumLlmKvHeads,
            Intermediate = o.LlmFeedForwardDim, RmsEpsilon = o.LlmRmsEpsilon, RopeTheta = o.LlmRopeTheta, AttentionBias = true,
            LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout, LoraTargets = LlamaLoraTargets.All,
        }, layers, Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42));
        Layers.AddRange(layers);
    }

    // The 16 kHz waveform's samples, truncated to the longest recording transcribed.
    private double[] Samples(Tensor<T> audio)
    {
        int length = Math.Min(audio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var samples = new double[length];
        var span = audio.Data.Span;
        for (int i = 0; i < length; i++) samples[i] = NumOps.ToDouble(span[i]);
        return samples;
    }

    private Tensor<T> Row(Tensor<T> input, int row, int rows)
    {
        if (rows == 1) return input;
        int samples = input.Length / rows;
        return Engine.Reshape(Engine.TensorSlice(Engine.Reshape(input, new[] { rows, samples }), new[] { row, 0 }, new[] { 1, samples }), new[] { samples });
    }

    // The logits [1, vocabulary] of the answer's first token for a waveform [S], or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> FirstLogits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            outputs[r] = _network.FirstLogits(_network.Prompt(_network.Speech(Samples(Row(input, r, rows)), training), _options.Instruction));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return FirstLogits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : FirstLogits(input, training: true);

    // The transcript a training target names: ids [length], or one distribution per token [length, vocabulary], read as
    // its most likely tokens; the end token is never a target.
    private int[] Transcript(FireRedAsrLlmNetwork<T> network, Tensor<T> target)
    {
        int vocabulary = _options.VocabSize, end = network.EndToken;
        var span = target.Data.Span;
        if (target.Rank >= 2 && target.Shape[^1] == vocabulary)
        {
            var tokens = new int[target.Length / vocabulary];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = -1;
                for (int v = 0; v < vocabulary; v++)
                    if (v != end && (best < 0 || NumOps.GreaterThan(span[r * vocabulary + v], span[r * vocabulary + best]))) best = v;
                tokens[r] = best;
            }
            return tokens;
        }
        var ids = new int[target.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value < 0 || value >= vocabulary || (int)value == end)
                throw new ArgumentException($"Token {value} is outside the vocabulary or the end token.", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    private Tensor<T> BatchLoss(FireRedAsrLlmNetwork<T> network, Tensor<T> input, Tensor<T> target, bool training)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var prompt = network.Prompt(network.Speech(Samples(Row(input, r, rows)), training), _options.Instruction);
            var loss = Engine.Reshape(network.Loss(prompt, Transcript(network, Row(target, r, rows))), new[] { 1 });
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The transcript's cross-entropy after the prompt, averaged over the batch, with LoRA dropout active.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        network.Llm.Training = true;
        try { TrainWithCustomLoss(input, _ => BatchLoss(network, input, target, training: true)); }
        finally { network.Llm.Training = false; }
        return true;
    }

    /// <inheritdoc />
    /// <remarks>With <see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>, the encoder, the adapter and the LoRA
    /// adapters (§2.2); otherwise every parameter.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var network = _network;
        if (network is null || _options.TrainingScope == SpeechLlmTrainingScope.AllParameters) return parameters;
        var trainedLayers = network.EncoderLayers.Concat(network.AdapterLayers).Concat(network.Llm.LoraLayers);
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(trainedLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return BatchLoss(network, input, target, training: false)[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "FireRedASRLLM-Native" : "FireRedASRLLM-ONNX", Description = "FireRedASR-LLM: Conformer encoder, adapter and Qwen2-7B-Instruct (FireRed Team 2025)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FireRedASRLLM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
