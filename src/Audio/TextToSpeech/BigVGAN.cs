using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// BigVGAN universal neural vocoder (Lee et al., 2023, NVIDIA).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BigVGAN (Lee et al., 2023, NVIDIA) is a universal neural vocoder using anti-aliased
/// multi-periodicity composition (AMP) with Snake activation functions. Unlike HiFi-GAN which
/// is optimized for speech, BigVGAN handles speech, music, and environmental sounds equally well.
/// BigVGAN v2 achieves PESQ 4.2 and UTMOS 4.2, supporting up to 44.1 kHz output for music.
/// </para>
/// <para>
/// <b>For Beginners:</b> BigVGAN is an upgraded version of HiFi-GAN that works great with
/// all types of audio, not just speech. It's like upgrading from a specialist (speech-only)
/// to a generalist (any sound).
///
/// Key innovations:
/// - Snake activation: Special math function that's great at representing periodic signals
///   (musical notes, vowels, harmonic sounds)
/// - Anti-aliased processing: Prevents "buzzy" or "metallic" artifacts in the output
/// - Universal: Works for speech, music, environmental sounds without retraining
///
/// When to use BigVGAN vs HiFi-GAN:
/// - Use HiFi-GAN for: Pure speech TTS (smaller, faster)
/// - Use BigVGAN for: Music, singing, sound effects, or mixed audio
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 100, outputSize: 1);
/// var vocoder = new BigVGAN&lt;float&gt;(arch, "bigvgan_v2.onnx");
/// var waveform = vocoder.Predict(melSpectrogram); // Direct vocoding
/// var audio = vocoder.Synthesize("Text is encoded elsewhere; use Predict for mel input.");
/// </code>
/// </para>
/// </remarks>
public class BigVGAN<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    #region Fields

    private readonly BigVGANOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ITextToSpeech Properties

    /// <inheritdoc />
    public IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; } = new[]
    {
        new VoiceInfo<T> { Id = "default", Name = "Default", Language = "en", Gender = VoiceGender.Neutral, Style = "neutral" }
    };

    /// <inheritdoc />
    public bool SupportsVoiceCloning => false;

    /// <inheritdoc />
    public bool SupportsEmotionControl => false;

    /// <inheritdoc />
    public bool SupportsStreaming => true;

    #endregion

    #region Constructors

    /// <summary>Creates a BigVGAN model in ONNX inference mode.</summary>
    public BigVGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, BigVGANOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new BigVGANOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        InitializeLayers();
    }

    /// <summary>Creates a BigVGAN model in native training mode.</summary>
    public BigVGAN(NeuralNetworkArchitecture<T> architecture, BigVGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BigVGANOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<BigVGAN<T>> CreateAsync(BigVGANOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new BigVGANOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("bigvgan", $"bigvgan_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new BigVGAN<T>(arch, mp, options);
    }

    #endregion

    #region ITextToSpeech

    /// <inheritdoc />
    public Tensor<T> Synthesize(string text, string? voiceId = null, double speakingRate = 1.0, double pitch = 0.0)
    {
        ThrowIfDisposed();
        var encoded = EncodeText(text);
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(encoded) : Predict(encoded);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> SynthesizeAsync(string text, string? voiceId = null, double speakingRate = 1.0,
        double pitch = 0.0, CancellationToken cancellationToken = default)
        => Task.Run(() => Synthesize(text, voiceId, speakingRate, pitch), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoiceCloning(string text, Tensor<T> referenceAudio, double speakingRate = 1.0, double pitch = 0.0)
        => throw new NotSupportedException("BigVGAN is a vocoder; use it with Predict(melSpectrogram) for mel-to-waveform conversion.");

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithEmotion(string text, string emotion, double emotionIntensity = 0.5,
        string? voiceId = null, double speakingRate = 1.0)
        => throw new NotSupportedException("BigVGAN is a vocoder and does not support emotion control.");

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
        => throw new NotSupportedException("BigVGAN is a vocoder and does not support speaker embedding extraction.");

    /// <inheritdoc />
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
        => new BigVGANStreamingSession(this);

    /// <summary>
    /// Converts a mel-spectrogram directly to a waveform (vocoder mode).
    /// </summary>
    /// <param name="melSpectrogram">Mel-spectrogram tensor [frames, numMels] or [numMels].</param>
    /// <returns>Audio waveform tensor [samples].</returns>
    public Tensor<T> Vocode(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(melSpectrogram) : Predict(melSpectrogram);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultBigVGANLayers(
            numMels: _options.NumMels, upsampleInitialChannel: _options.UpsampleInitialChannel,
            upsampleRates: _options.UpsampleRates, numResBlocks: _options.NumResBlocks,
            dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BigVGAN-Native" : "BigVGAN-ONNX",
            Description = $"BigVGAN {_options.Variant} universal vocoder (Lee et al., 2023, NVIDIA)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.UpsampleRates.Length
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["UseSnakeActivation"] = _options.UseSnakeActivation.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.NumMels); w.Write(_options.HopLength);
        w.Write(_options.UpsampleInitialChannel); w.Write(_options.NumResBlocks);
        w.Write(_options.UseSnakeActivation); w.Write(_options.SnakeAlpha);
        w.Write(_options.UpsampleRates.Length);
        foreach (int r in _options.UpsampleRates) w.Write(r);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.NumMels = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.UpsampleInitialChannel = r.ReadInt32(); _options.NumResBlocks = r.ReadInt32();
        _options.UseSnakeActivation = r.ReadBoolean(); _options.SnakeAlpha = r.ReadDouble();
        int ratesLen = r.ReadInt32(); _options.UpsampleRates = new int[ratesLen]; for (int i = 0; i < ratesLen; i++) _options.UpsampleRates[i] = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new BigVGAN<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodeText(string text)
    {
        var encoding = _tokenizer.Encode(text);
        int dim = _options.NumMels;
        var tokens = new Tensor<T>([dim]);
        int copyCount = Math.Min(encoding.TokenIds.Count, dim);
        for (int i = 0; i < copyCount; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    #endregion

    #region Streaming Session

    private sealed class BigVGANStreamingSession : IStreamingSynthesisSession<T>
    {
        private readonly BigVGAN<T> _model;
        private readonly List<string> _textChunks = new();
        private bool _disposed;

        public BigVGANStreamingSession(BigVGAN<T> model) { _model = model; }

        public void FeedText(string textChunk) { if (_disposed) throw new ObjectDisposedException(nameof(BigVGANStreamingSession)); _textChunks.Add(textChunk); }

        public IEnumerable<Tensor<T>> GetAvailableAudio()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BigVGANStreamingSession));
            foreach (var chunk in _textChunks) yield return _model.Synthesize(chunk);
            _textChunks.Clear();
        }

        public IEnumerable<Tensor<T>> Finalize()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(BigVGANStreamingSession));
            var result = GetAvailableAudio().ToList();
            _disposed = true;
            return result;
        }

        public void Dispose() { _disposed = true; }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BigVGAN<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
