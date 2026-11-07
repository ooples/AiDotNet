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

namespace AiDotNet.SpeechRecognition.NeMo;

/// <summary>
/// Canary-Qwen-2.5B (NVIDIA, 2025): a speech-augmented language model (SALM, Chen et al. 2023) built from
/// canary-1b-flash's Fast Conformer encoder, a linear projection and Qwen3-1.7B with LoRA.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Canary-Qwen listens with Canary's speech encoder, maps each 80 ms of what it hears into
/// the language model's word space with one linear layer, and asks Qwen3 "Transcribe the following:" with that audio in
/// the chat message. Qwen3 then writes the transcript.</para>
/// <para>
/// The release (<c>config.json</c> and the model card) fixes the network: NeMo's <c>ConformerEncoder</c> at width 1,024
/// with 32 blocks, 8 heads, kernel 9 and 8x depthwise-separable striding over 128 log-mel bands; an identity modality
/// adapter; <c>AudioPerceptionModule.proj</c>, a linear map to 2,048; and Qwen3-1.7B with LoRA of rank 128 (alpha 256,
/// dropout 0.01) on <c>q_proj</c> and <c>v_proj</c>. The prompt is NeMo's <c>qwen</c> format,
/// <c>&lt;|im_start|&gt;user\nTranscribe the following: &lt;audio&gt;&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>,
/// and the answer ends with <c>&lt;|im_end|&gt;</c>. Training freezes the LLM (<c>freeze_params</c>) and updates the
/// encoder, the projection and LoRA, with AdamW (learning rate 5e-4, betas 0.9 and 0.98, weight decay 1e-3) under cosine
/// annealing after 1,000 warm-up steps to 1e-6. A model built here starts from random weights and a frozen random LLM
/// cannot fit a transcript, so the default updates every parameter; the release's scope is
/// <see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 160000, outputSize: 151936);
/// var model = new CanaryQwen&lt;float&gt;(architecture);
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
[ResearchPaper("Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition", "https://arxiv.org/abs/2305.05084", Year = 2023, Authors = "Rekesh et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.98, WeightDecay = 1e-3,
                WarmupSteps = 1000, MinLearningRate = 1e-6, Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "nvidia/canary-qwen-2.5b config.json: torch.optim.AdamW with lr 5e-4, betas (0.9, 0.98) and "
                        + "weight decay 1e-3; NeMo CosineAnnealing with 1000 warm-up steps and min_lr 1e-6.")]
public partial class CanaryQwen<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly CanaryQwenOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private CanaryQwenNetwork<T>? _network;
    internal CanaryQwenNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public CanaryQwen(NeuralNetworkArchitecture<T> architecture, string modelPath, CanaryQwenOptions? options = null) : base(architecture)
    {
        _options = options ?? new CanaryQwenOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { "en" }; InitializeLayersCore();
    }

    public CanaryQwen(NeuralNetworkArchitecture<T> architecture, CanaryQwenOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new CanaryQwenOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, warmupStepsOverride: _options.WarmupSteps)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { "en" }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>Transcribes audio: the LLM's greedy answer to the instruction, at most
    /// <see cref="CanaryQwenOptions.MaxTextLength"/> tokens.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for Canary-Qwen.");
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
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Canary-Qwen does not support streaming.");

    // Qwen's tokenizer decodes when the vocabulary is Qwen's; otherwise each token is its vocabulary entry or code point.
    private string Text(CanaryQwenNetwork<T> network, int[] tokens)
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
        var layers = new List<LayerBase<T>>();
        _network = new CanaryQwenNetwork<T>(Engine, new ConformerAsrConfiguration
        {
            SampleRate = o.SampleRate, NumMels = o.NumMels, EncoderDim = o.EncoderDim,
            NumEncoderLayers = o.NumEncoderLayers, NumAttentionHeads = o.NumAttentionHeads, FeedForwardMultiplier = o.FeedForwardMultiplier,
            ConvKernel = o.ConvKernel, Subsampling = ConformerAsrSubsampling.DepthwiseStriding8x, SubsamplingChannels = o.SubsamplingChannels,
            VocabSize = 1,
        }, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.DecoderDim, Layers = o.NumDecoderLayers, Heads = o.NumDecoderHeads, KeyValueHeads = o.NumDecoderKvHeads,
            HeadDim = o.DecoderHeadDim, Intermediate = o.DecoderFeedForwardDim, RmsEpsilon = o.DecoderRmsEpsilon, RopeTheta = o.DecoderRopeTheta,
            QueryKeyNorm = true, TieWordEmbeddings = true, LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout,
            LoraTargets = LlamaLoraTargets.Query | LlamaLoraTargets.Value,
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
    private int[] Transcript(CanaryQwenNetwork<T> network, Tensor<T> target)
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

    private Tensor<T> BatchLoss(CanaryQwenNetwork<T> network, Tensor<T> input, Tensor<T> target, bool training)
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
    /// <remarks>With <see cref="SpeechLlmTrainingScope.EncoderAdapterAndLora"/>, the encoder, the projection and the LoRA
    /// adapters (the release's <c>freeze_params</c> and <c>prevent_freeze_params</c>); otherwise every parameter.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var network = _network;
        if (network is null || _options.TrainingScope == SpeechLlmTrainingScope.AllParameters) return parameters;
        var trainedLayers = network.EncoderLayers.Concat(network.ProjectionLayers).Concat(network.Llm.LoraLayers);
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
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "CanaryQwen-Native" : "CanaryQwen-ONNX", Description = "Canary-Qwen-2.5B: Fast Conformer, linear projection and Qwen3-1.7B with LoRA (NVIDIA 2025)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CanaryQwen<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
