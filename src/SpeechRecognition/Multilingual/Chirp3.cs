using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.Foundation;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.SpeechRecognition.Multilingual;

/// <summary>
/// Chirp 3: Google Cloud's third-generation Chirp recognizer, a USM-family Conformer with a CTC layer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages" (Zhang et al., Google, 2023)</item>
/// <item>Model: "Chirp 3" (Google Cloud Speech-to-Text V2, 2025)</item></list></para>
/// <para><b>For Beginners:</b> Chirp 3 is the third generation of the speech model Google serves in its cloud. It
/// listens to 16 kHz audio turned into a 128-band spectrogram, encodes it with a large Conformer, and outputs for every
/// 40 ms frame a probability for each token; collapsing repeats and blanks gives the transcript.</para>
/// <para>
/// Google has not published Chirp 3's network; it presents Chirp 3 as the latest generation of its USM-based Chirp
/// models. This class therefore builds the published USM-CTC architecture: the paper's Conformer-2B (Table 2: 32
/// layers, 1536 wide, 16 heads, kernel 5) over 128-dimensional log-mel features (section 2.7) with BEST-RQ's 4x
/// convolutional reduction, 8-second chunk-wise attention (section 2.4) and a CTC layer, with BEST-RQ pre-training
/// (<see cref="Pretrain"/>) under the 16-codebook multi-softmax loss. Rotary position encoding stands in for the paper's
/// relative attention.
/// The MOST speech-text stage and noisy student training of the USM paper are data procedures outside the network.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Chirp 3 recognizer
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputHeight: 16000, inputWidth: 1, inputDepth: 1, outputSize: 32000);
/// var model = new Chirp3&lt;double&gt;(architecture);
///
/// // Or load an exported ONNX model
/// var onnxModel = new Chirp3&lt;double&gt;(architecture, "chirp3.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages", "https://arxiv.org/abs/2303.01037", Year = 2023, Authors = "Zhang et al.")]
[ArchitectureFromPaper("https://arxiv.org/abs/2202.01855",
    "Google has not published Chirp 3's network and presents it as the latest generation of its USM-based "
    + "Chirp models, so it runs on the BEST-RQ network USM pre-trains, at USM's size, codebook count and chunk-wise attention.")]
public partial class Chirp3<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    /// <inheritdoc />
    /// <remarks>The CTC layer's width: the pre-training softmax layers over the codebooks are used only by <see cref="Pretrain"/>.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly Chirp3Options _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _pretrainingOptimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private BestRqNetwork<T>? _network;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    internal BestRqNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => false;

    public Chirp3(NeuralNetworkArchitecture<T> architecture, string modelPath, Chirp3Options? options = null) : base(architecture) { _options = options ?? new Chirp3Options(); Options = _options; _useNativeMode = false; base.SampleRate = _options.SampleRate; base.NumMels = 0; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions); SupportedLanguages = new[] { "en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru", "ar", "hi", "tr", "it", "nl", "pl", "sv" }; InitializeLayers(); }
    public Chirp3(NeuralNetworkArchitecture<T> architecture, Chirp3Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new Chirp3Options(); Options = _options; _useNativeMode = true; _suppliedOptimizer = optimizer; _optimizer = optimizer
        // The paper states no optimizer settings (section 2.7 gives only separate encoder and decoder optimizers).
        ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.NumMels = 0; SupportedLanguages = new[] { "en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru", "ar", "hi", "tr", "it", "nl", "pl", "sv" }; InitializeLayers(); }

    /// <summary>
    /// Transcribes audio: the encoder, projection and CTC layer give per-frame distributions, decoded greedily.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        Tensor<T> logits;

        logits = Logits(features, training: false);

        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegments(text, duration, confidence) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) { var logits = Logits(PreprocessAudio(audio), training: false); var (tokens, _) = CTCGreedyDecodeWithConfidence(logits); return ClassifyLanguageFromTokens(tokens); }
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) { var detected = DetectLanguage(audio); var result = new Dictionary<string, T>(); double primaryProb = 0.85; double otherProb = SupportedLanguages.Count > 1 ? (1.0 - primaryProb) / (SupportedLanguages.Count - 1) : 0.0; foreach (var lang in SupportedLanguages) result[lang] = NumOps.FromDouble(lang == detected ? primaryProb : otherProb); return result; }
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("Chirp3 does not support streaming.");

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // USM's network: Conformer-2B over normalized 128-bin log-mel features with chunk-wise attention, BEST-RQ's
        // quantizers with the multi-softmax loss, and a projection + CTC layer (Zhang et al. 2023). It used to be a generic
        // template whose blocks had no convolution module.
        var layers = new List<LayerBase<T>>();
        _network = new BestRqNetwork<T>(Engine, layers, _options.ToConfiguration(), Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        Layers.AddRange(layers);
    }

    // Chirp 3 reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_network is null) return base.PreprocessAudio(rawAudio);
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private double[] Samples(Tensor<T> waveform, int offset, int count)
    {
        var samples = new double[count];
        var span = waveform.Data.Span;
        for (int i = 0; i < count; i++) samples[i] = NumOps.ToDouble(span[offset + i]);
        return samples;
    }

    // CTC logits [frames, vocab] for a waveform [S], or [batch, frames, vocab] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input, bool training)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        if (input.Rank == 1) return _network.CtcLogits(_network.Encode(_network.Features(Samples(input, 0, input.Length)), training));
        int rows = input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
            outputs[r] = _network.CtcLogits(_network.Encode(_network.Features(Samples(input, r * samples, samples)), training));
        return Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input, training: false); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input, training: true);

    /// <summary>
    /// The BEST-RQ pre-training loss of waveforms <c>[S]</c> or <c>[batch, S]</c> with the multi-softmax of Zhang et al.
    /// 2023 (section 2.2.1), averaged over the batch: the encoder sees the input with spans replaced by noise, and each
    /// softmax layer predicts, at every masked frame, the label its fixed random-projection quantizer gives the unmasked
    /// input; the codebooks are weighted equally.
    /// </summary>
    /// <param name="audio">The waveforms.</param>
    /// <param name="random">The mask and noise draws; the model's own generator when null.</param>
    public T PretrainingLoss(Tensor<T> audio, Random? random = null)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        return PretrainingLossTensor(audio, training: false, random ?? _random)[0];
    }

    private Tensor<T> PretrainingLossTensor(Tensor<T> audio, bool training, Random random)
    {
        var network = _network ?? throw new InvalidOperationException("Pre-training runs on the native Chirp3 network.");
        int rows = audio.Rank == 1 ? 1 : audio.Shape[0], samples = audio.Length / rows;
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = network.PretrainingLoss(Samples(audio, r * samples, samples), training, random);
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(audio)), NumOps.FromDouble(1.0 / rows));
    }

    /// <summary>One BEST-RQ pre-training step (section 2.2) on waveforms <c>[S]</c> or <c>[batch, S]</c>.</summary>
    /// <param name="audio">The unlabeled waveforms.</param>
    /// <returns>The step's loss.</returns>
    public T Pretrain(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (_network is null) throw new NotSupportedException("Pre-training runs on the native Chirp3 network.");
        // A caller's optimizer serves both phases; the paper states no pre-training optimizer.
        _pretrainingOptimizer ??= _suppliedOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        return TrainWithCustomLoss(audio, _ => PretrainingLossTensor(audio, training: true, _random), _pretrainingOptimizer);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "Chirp3-Native" : "Chirp3-ONNX", Description = "Chirp 3: Google Cloud's third-generation recognizer on the published USM-CTC network (Google, 2025)", FeatureCount = 1, Complexity = _options.NumEncoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };



    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(Tensor<T> logits) { var tokens = new List<int>(); double totalConf = 0; int confCount = 0; int prevToken = -1; int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1; int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0]; for (int t = 0; t < numFrames && tokens.Count < _options.MaxTextLength; t++) { int maxIdx = 0; double maxVal = double.NegativeInfinity; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); if (val > maxVal) { maxVal = val; maxIdx = v; } } double sumExp = 0; for (int v = 0; v < vocabSize; v++) { double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]); sumExp += Math.Exp(val - maxVal); } double frameConf = 1.0 / sumExp; if (maxIdx != prevToken && maxIdx > 0) { tokens.Add(maxIdx); totalConf += frameConf; confCount++; } prevToken = maxIdx; } return (tokens, confCount > 0 ? totalConf / confCount : 0.0); }
    private static string TokensToText(List<int> tokens) { var sb = new System.Text.StringBuilder(); foreach (var t in tokens) { if (t > 0 && t <= char.MaxValue) sb.Append((char)t); else if (t > char.MaxValue && t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t)); } return sb.ToString().Trim(); }
    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration, double confidence) { if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>(); return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(confidence) } }; }
    private string ClassifyLanguageFromTokens(List<int> tokens) { if (tokens.Count == 0) return _options.Language; int cjkCount = 0, latinCount = 0; foreach (var t in tokens) { if (t >= 0x4E00 && t <= 0x9FFF) cjkCount++; else if (t >= 0x41 && t <= 0x7A) latinCount++; } if (cjkCount > latinCount && SupportedLanguages.Contains("zh")) return "zh"; return _options.Language; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Chirp3<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
