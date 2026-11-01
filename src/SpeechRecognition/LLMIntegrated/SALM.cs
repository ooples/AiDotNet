using AiDotNet.LearningRateSchedulers;
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
/// SALM (Chen et al., 2023): a speech-augmented language model that feeds a Fast Conformer encoder's output through a
/// Conformer modality adapter into a Megatron GPT with LoRA, for multitask speech recognition and translation with
/// in-context learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SALM listens with a speech encoder, shortens what it hears to one token per 320 ms with
/// two more Conformer layers, and places those tokens in front of a written instruction for a large language model. The
/// language model then writes the answer, here a transcript. Words given in the instruction's text context can steer
/// the answer (the paper's keyword boosting) without retraining.</para>
/// <para>
/// §3.1 of the paper: the audio encoder is the 110M Fast Conformer-large; the modality adapter is two Conformer layers
/// with 4x subsampling, giving a 320 ms frame shift, projected to the LLM's width and prepended to the text context and
/// instruction; the LLM is NVIDIA's 2B Megatron GPT (GPT-2B-001: 24 layers of width 2,048, SwiGLU, rotary embeddings on
/// half of each head, LayerNorm, no biases, untied embeddings, a 256,000-piece SentencePiece vocabulary), with LoRA of
/// rank 128 added; the LLM is frozen and the rest is trained. NeMo's <c>modular_speechllm</c> release fixes the rest:
/// the adapter is NeMo's <c>ConformerEncoder</c> with depthwise-separable striding; LoRA sits on the fused
/// query-key-value projection; the template is <c>Q: {input}\nA: {output}</c> without a beginning-of-sequence token and
/// with <c>&lt;/s&gt;</c> after the answer; the loss covers the answer only. §4 trains with Adam (learning rate 1e-4,
/// weight decay 1e-3), cosine annealing after 2,000 warm-up steps and gradients clipped to 5. A model built here starts
/// from random weights and a frozen random LLM cannot fit a transcript, so the default updates every parameter; the
/// paper's scope is <see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 160000, outputSize: 256000);
/// var model = new SALM&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SALM: Speech-augmented Language Model with In-context Learning for Speech Recognition and Translation", "https://arxiv.org/abs/2310.09424", Year = 2024, Authors = "Chen et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, WeightDecay = 1e-3,
                WarmupSteps = 2000, MinLearningRate = 0, MaxGradientNorm = 5.0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Chen et al. 2023, Sec. 4: the Adam optimizer with a learning rate of 1e-4 and "
                        + "a weight decay of 1e-3, cosine annealing with 2000 warm-up steps, and "
                        + "gradients clipped to 5.0.")]
public partial class SALM<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly SALMOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private SalmNetwork<T>? _network;
    internal SalmNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public SALM(NeuralNetworkArchitecture<T> architecture, string modelPath, SALMOptions? options = null) : base(architecture)
    {
        _options = options ?? new SALMOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public SALM(NeuralNetworkArchitecture<T> architecture, SALMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new SALMOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, warmupStepsOverride: _options.WarmupSteps)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>Transcribes audio: the LLM's greedy answer (the paper's default decoding) to the instruction, at most
    /// <see cref="SALMOptions.MaxTextLength"/> tokens.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for SALM.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        int[] tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            tokens = network.Greedy(network.Prompt(network.Speech(Samples(audio), training: false), _options.Instruction), _options.MaxTextLength);
        return new TranscriptionResult<T>
        {
            Text = Text(network, tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Length > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SALM does not support streaming.");

    // Megatron's tokenizer decodes when the vocabulary is Megatron's; otherwise each token is its vocabulary entry or code point.
    private string Text(SalmNetwork<T> network, int[] tokens)
    {
        if (network.UsesMegatronVocabulary) return network.Tokenizer.Decode(tokens).Trim();
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
        var layers = new List<LayerBase<T>>();
        _network = new SalmNetwork<T>(Engine, new ConformerAsrConfiguration
        {
            SampleRate = o.SampleRate, NumMels = o.NumMels, EncoderDim = o.EncoderDim, NumEncoderLayers = o.NumEncoderLayers,
            NumAttentionHeads = o.NumAttentionHeads, FeedForwardMultiplier = o.FeedForwardMultiplier, ConvKernel = o.ConvKernel,
            Subsampling = ConformerAsrSubsampling.DepthwiseStriding8x, SubsamplingChannels = o.SubsamplingChannels, VocabSize = 1,
        }, new ConformerAsrConfiguration
        {
            SampleRate = o.SampleRate, EncoderDim = o.AdapterDim, NumEncoderLayers = o.NumAdapterLayers, NumAttentionHeads = o.NumAdapterHeads,
            FeedForwardMultiplier = o.FeedForwardMultiplier, ConvKernel = o.ConvKernel, Subsampling = ConformerAsrSubsampling.DepthwiseStriding4x,
            SubsamplingChannels = o.SubsamplingChannels, VocabSize = 1,
        }, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.LlmDim, Layers = o.NumLlmLayers, Heads = o.NumLlmAttentionHeads, KeyValueHeads = o.NumLlmAttentionHeads,
            Intermediate = o.LlmFeedForwardDim, RmsEpsilon = o.LlmNormEpsilon, RopeTheta = o.LlmRopeTheta, PartialRotaryFactor = o.PartialRotaryFactor,
            LayerNorm = true, LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout, FusedLora = true,
            LoraTargets = LlamaLoraTargets.Query | LlamaLoraTargets.Key | LlamaLoraTargets.Value,
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
    private int[] Transcript(SalmNetwork<T> network, Tensor<T> target)
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

    private Tensor<T> BatchLoss(SalmNetwork<T> network, Tensor<T> input, Tensor<T> target, bool training)
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
    /// <remarks>The answer's cross-entropy after the prompt, averaged over the batch, with LoRA dropout active.</remarks>
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
    /// <remarks>With <see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>, the encoder, the adapter, the projection
    /// and the LoRA adapters (§3.1); otherwise every parameter.</remarks>
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
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SALM-Native" : "SALM-ONNX", Description = "SALM: Fast Conformer, Conformer modality adapter and Megatron GPT with LoRA (Chen et al. 2023)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumAdapterLayers + _options.NumLlmLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SALM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
