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

namespace AiDotNet.SpeechRecognition.WhisperFamily;

/// <summary>
/// WhisperX (Bain et al. 2023): Whisper transcription with voice-activity cut-and-merge and word-level timing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> WhisperX listens to 16 kHz audio in 30-second windows turned into a log-mel spectrogram,
/// encodes each window with a Transformer, and a second Transformer writes the transcript token by token while reading
/// the encoded audio.</para>
/// <para>
/// WhisperX transcribes with Whisper (the network here is large-v3) and adds word timing. The paper times words by forced alignment with a phoneme model; transcription here times them by dynamic time warping over Whisper's own cross-attention, openai-whisper's <c>word_timestamps</c>. Every forward reads the 30-second window openai-whisper builds (<c>pad_or_trim</c>); <c>Predict</c> returns the
/// decoder's logits over the start sequence (start of transcript, language, transcribe, no timestamps), whose last row is
/// the first token's distribution, and <c>Train</c> minimizes the teacher-forced cross-entropy of a transcript given as
/// token ids or as one distribution per token.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Your recording: 16 kHz mono samples in [-1, 1], shaped [1, samples]. Noise stands in for it here.
/// var waveform = Tensor&lt;float&gt;.CreateRandom(1, 16000);
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 480000, outputSize: 51866);
/// var model = new WhisperX&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("WhisperX: Time-Accurate Speech Transcription of Long-Form Audio", "https://arxiv.org/abs/2303.00747", Year = 2023, Authors = "Bain et al.")]
[ArchitectureFromPaper("https://arxiv.org/abs/2212.04356",
    "WhisperX transcribes with an unmodified Whisper model (Bain et al. 2023, section 3.2), so it runs on the openai-whisper network.")]
public partial class WhisperX<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The decoder's vocabulary: <c>Predict</c> returns one distribution over it per start token.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly WhisperXOptions _options;
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
    /// <inheritdoc />
    public bool SupportsWordTimestamps => true;

    public WhisperX(NeuralNetworkArchitecture<T> architecture, string modelPath, WhisperXOptions? options = null) : base(architecture)
    {
        _options = options ?? new WhisperXOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = OpenAiWhisperTranscriber<T>.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = Languages(_options.VocabSize); InitializeLayersCore();
    }

    public WhisperX(NeuralNetworkArchitecture<T> architecture, WhisperXOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new WhisperXOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("WhisperX does not support streaming.");

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
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "WhisperX-Native" : "WhisperX-ONNX", Description = "WhisperX: Whisper transcription with voice-activity cut-and-merge and word-level timing", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WhisperX<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
