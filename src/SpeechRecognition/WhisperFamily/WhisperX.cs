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
public partial class WhisperX<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
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
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
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
        var whisper = _whisper ?? throw new NotSupportedException("Transcription runs on the native Whisper network.");
        return whisper.Transcribe(Samples(audio), language ?? _options.Language, ResolveReturnTimestamps(includeTimestamps) || _options.WordTimestamps, _options.Temperatures);
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    public string DetectLanguage(Tensor<T> audio) => DetectLanguageProbabilities(audio).OrderByDescending(p => NumOps.ToDouble(p.Value)).First().Key;

    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var whisper = _whisper ?? throw new NotSupportedException("Language detection runs on the native Whisper network.");
        return whisper.DetectLanguage(Samples(audio)).ToDictionary(p => p.Key, p => NumOps.FromDouble(p.Value));
    }

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

    private double[] Samples(Tensor<T> waveform)
    {
        var samples = new double[waveform.Length];
        var span = waveform.Data.Span;
        for (int i = 0; i < samples.Length; i++) samples[i] = NumOps.ToDouble(span[i]);
        return samples;
    }

    // The decoder's logits over the start sequence for a waveform [S], or [batch, prompt, vocabulary] for [batch, S].
    private Tensor<T> PromptLogits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_whisper is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        string language = _options.Language ?? "en";
        if (input.Rank == 1) return _whisper.PromptLogits(_whisper.Window(Samples(input)), language);
        int rows = input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            outputs[r] = _whisper.PromptLogits(_whisper.Window(Samples(Engine.TensorSlice(input, new[] { r, 0 }, new[] { 1, samples }))), language);
        return Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return PromptLogits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => PromptLogits(input);

    /// <inheritdoc />
    /// <remarks>Whisper's objective (section 2.3): the teacher-forced cross-entropy of the target transcript after the start
    /// sequence, closed by the end-of-text token.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var whisper = _whisper;
        if (whisper is null) return false;
        string language = _options.Language ?? "en";
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var windows = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            windows[r] = whisper.Window(Samples(rows == 1 ? input : Engine.TensorSlice(input, new[] { r, 0 }, new[] { 1, samples })));
        var transcripts = new int[rows][];
        int perRow = target.Length / rows;
        for (int r = 0; r < rows; r++)
        {
            var row = rows == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
            transcripts[r] = whisper.TranscriptTokens(row);
        }
        TrainWithCustomLoss(input, _ =>
        {
            Tensor<T>? total = null;
            for (int r = 0; r < rows; r++)
            {
                var loss = whisper.TranscriptLoss(windows[r], transcripts[r], language);
                total = total is null ? loss : Engine.TensorAdd(total, loss);
            }
            return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
        }, _optimizer);
        return true;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "WhisperX-Native" : "WhisperX-ONNX", Description = "WhisperX: Whisper transcription with voice-activity cut-and-merge and word-level timing", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WhisperX<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
