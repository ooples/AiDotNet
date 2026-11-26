using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Whisper;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.ProprietaryAPI;

/// <summary>
/// Groq's hosted Whisper: openai-whisper large-v3 served on Groq's inference hardware.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Groq Whisper listens to 16 kHz audio in 30-second windows turned into a log-mel spectrogram,
/// encodes each window with a Transformer, and a second Transformer writes the transcript token by token while reading
/// the encoded audio.</para>
/// <para>
/// Groq serves OpenAI's released Whisper checkpoints; the network here is large-v3 (32 encoder and 32 decoder layers, width 1280, 128 mel bins). Every forward reads the 30-second window openai-whisper builds (<c>pad_or_trim</c>); <c>Predict</c> returns the
/// decoder's logits over the start sequence (start of transcript, language, transcribe, no timestamps), whose last row is
/// the first token's distribution, and <c>Train</c> minimizes the teacher-forced cross-entropy of a transcript given as
/// token ids or as one distribution per token.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var waveform = Tensor&lt;float&gt;.CreateRandom(1, 16000);
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 480000, outputSize: 51866);
/// var model = new GroqWhisper&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Robust Speech Recognition via Large-Scale Weak Supervision", "https://arxiv.org/abs/2212.04356", Year = 2023, Authors = "Radford et al.")]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-6,
                WeightDecay = 0.1, Schedule = LearningRateSchedulerType.LinearWarmup,
                WarmupSteps = 2048, PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
                MinLearningRate = 0, MaxGradientNorm = 1.0, ReferenceBatchSize = 256,
                Source = "Radford et al. 2022, Table 17 (Appendix F): AdamW, beta1 0.9, beta2 0.98, eps 1e-6, weight decay 0.1, 2048 warmup updates, linear decay, max grad norm 1.0, batch size 256. No learning rate is declared because Table 19 states one per model size and this type does not select one.")]
public partial class GroqWhisper<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary: <c>Predict</c> returns one distribution over it per start token.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly GroqWhisperOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The openai-whisper network on the native path; null in ONNX mode or with custom layers.
    private WhisperRecognizer<T>? _whisper;
    internal WhisperRecognizer<T>? Whisper => _whisper;
    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc/>
    public bool SupportsWordTimestamps => true;

    public GroqWhisper(NeuralNetworkArchitecture<T> architecture, string modelPath, GroqWhisperOptions? options = null) : base(architecture)
    {
        _options = options ?? new GroqWhisperOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = OpenAiWhisperTranscriber<T>.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = Languages(_options.VocabSize); InitializeLayersCore();
    }

    public GroqWhisper(NeuralNetworkArchitecture<T> architecture, GroqWhisperOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new GroqWhisperOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this) ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = OpenAiWhisperTranscriber<T>.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = Languages(_options.VocabSize); InitializeLayersCore();
    }

    // 99 language tokens, or 100 from large-v3's 51,866-token vocabulary on.
    private static IReadOnlyList<string> Languages(int vocabulary) => WhisperTiktoken.LanguageCodes.Take(vocabulary - 51765 - 1).ToList();

    /// <summary>Transcribes 16 kHz audio with openai-whisper's <c>transcribe</c>: 30-second windows, temperature fallback,
    /// and word timing when timestamps are requested.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        return TranscribeNativeWhisper(audio, _whisper, language ?? _options.Language, ResolveReturnTimestamps(includeTimestamps) || _options.WordTimestamps, _options.Temperatures);
    }

    /// <inheritdoc/>
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc/>
    public string DetectLanguage(Tensor<T> audio) { ThrowIfDisposed(); return DetectNativeWhisperLanguage(audio, _whisper); }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { ThrowIfDisposed(); return DetectNativeWhisperLanguageProbabilities(audio, _whisper); }

    /// <inheritdoc/>
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("GroqWhisper does not support streaming.");

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _whisper = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // openai-whisper's network (Radford et al. 2022, section 2.2). It used to be a generic encoder-decoder template.
        var layers = new List<LayerBase<T>>();
        _whisper = new WhisperRecognizer<T>(Engine, layers, _options.ToDimensions(), _options.Seed ?? 42);
        Layers.AddRange(layers);
    }
    private Tensor<T> PromptLogits(Tensor<T> input) => ForwardNativeWhisperBatch(input, _whisper, _options.Language);

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return PromptLogits(input); }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => PromptLogits(input);

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target) =>
        TrainNativeWhisperBatch(input, target, _whisper, _optimizer, _options.Language);

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) =>
        EvaluateNativeWhisperBatchObjective(input, target, _whisper, _options.Language);


    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "GroqWhisper-Native" : "GroqWhisper-ONNX", Description = "Groq Whisper: openai-whisper large-v3 served on Groq's inference hardware", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GroqWhisper<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
