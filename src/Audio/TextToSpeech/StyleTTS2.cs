using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// StyleTTS 2 text-to-speech model (Li et al., 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// StyleTTS 2 achieves human-level naturalness (MOS 4.16 on LJSpeech) by disentangling
/// speech into content and style, then using diffusion-based style generation. It supports
/// zero-shot voice cloning from a short reference clip and fine-grained prosody control.
/// </para>
/// <para>
/// <b>For Beginners:</b> StyleTTS 2 is like having a professional voice actor:
/// 1. You write the script (text)
/// 2. You optionally provide a voice sample to clone (reference audio)
/// 3. The model generates natural-sounding speech with realistic intonation
///
/// It's one of the most natural-sounding open-source TTS models available.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 1);
/// var model = new StyleTTS2&lt;float&gt;(arch, "styletts2_base.onnx");
/// var audio = model.Synthesize("Hello, how are you today?");
/// // Clone a voice
/// var cloned = model.SynthesizeWithVoiceCloning("Hi there!", referenceAudio);
/// </code>
/// </para>
/// </remarks>
public class StyleTTS2<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    #region Fields

    private readonly StyleTTS2Options _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
    public bool SupportsVoiceCloning => true;

    /// <inheritdoc />
    public bool SupportsEmotionControl => true;

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    #endregion

    #region Constructors

    /// <summary>Creates a StyleTTS 2 model in ONNX inference mode.</summary>
    public StyleTTS2(NeuralNetworkArchitecture<T> architecture, string modelPath, StyleTTS2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new StyleTTS2Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates a StyleTTS 2 model in native training mode.</summary>
    public StyleTTS2(NeuralNetworkArchitecture<T> architecture, StyleTTS2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new StyleTTS2Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<StyleTTS2<T>> CreateAsync(StyleTTS2Options? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new StyleTTS2Options();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("styletts2", $"styletts2_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new StyleTTS2<T>(arch, mp, options);
    }

    #endregion

    #region ITextToSpeech

    /// <inheritdoc />
    public Tensor<T> Synthesize(string text, string? voiceId = null, double speakingRate = 1.0, double pitch = 0.0)
    {
        ThrowIfDisposed();
        var encoded = EncodeText(text);
        var styled = ApplyStyle(encoded, speakingRate, pitch);
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(styled) : Predict(styled);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> SynthesizeAsync(string text, string? voiceId = null, double speakingRate = 1.0,
        double pitch = 0.0, CancellationToken cancellationToken = default)
        => Task.Run(() => Synthesize(text, voiceId, speakingRate, pitch), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoiceCloning(string text, Tensor<T> referenceAudio, double speakingRate = 1.0, double pitch = 0.0)
    {
        ThrowIfDisposed();
        var styleEmbedding = ExtractSpeakerEmbedding(referenceAudio);
        var encoded = EncodeText(text);
        // Concatenate style embedding with encoded text features
        var combined = new Tensor<T>(new[] { encoded.Length + styleEmbedding.Length });
        for (int i = 0; i < encoded.Length; i++) combined[i] = encoded[i];
        for (int i = 0; i < styleEmbedding.Length; i++) combined[encoded.Length + i] = styleEmbedding[i];
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(combined) : Predict(combined);
    }

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithEmotion(string text, string emotion, double emotionIntensity = 0.5,
        string? voiceId = null, double speakingRate = 1.0)
    {
        ThrowIfDisposed();
        var encoded = EncodeText(text);
        var styled = ApplyStyle(encoded, speakingRate, pitch: 0.0, emotion: emotion, emotionIntensity: emotionIntensity);
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(styled) : Predict(styled);
    }

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(referenceAudio);
        var features = PreprocessAudio(referenceAudio);
        var output = Predict(features);
        // Take first StyleDim elements as embedding
        int embDim = _options.SpeakerEmbeddingDim;
        var embedding = new Tensor<T>(new[] { embDim });
        for (int i = 0; i < embDim && i < output.Length; i++) embedding[i] = output[i];
        // L2 normalize
        double norm = 0; for (int i = 0; i < embDim; i++) norm += NumOps.ToDouble(embedding[i]) * NumOps.ToDouble(embedding[i]);
        norm = Math.Sqrt(norm);
        if (norm > 1e-8) for (int i = 0; i < embDim; i++) embedding[i] = NumOps.FromDouble(NumOps.ToDouble(embedding[i]) / norm);
        return embedding;
    }

    /// <inheritdoc />
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
        => throw new NotSupportedException("StyleTTS 2 does not support streaming synthesis.");

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultStyleTTS2Layers(
            textEncoderDim: _options.TextEncoderDim, numTextEncoderLayers: _options.NumTextEncoderLayers,
            styleDim: _options.StyleDim, prosodyDim: _options.ProsodyDim,
            numMels: _options.NumMels, numAttentionHeads: _options.NumAttentionHeads,
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
            Name = _useNativeMode ? "StyleTTS2-Native" : "StyleTTS2-ONNX",
            Description = $"StyleTTS 2 {_options.Variant} TTS model (Li et al., 2023)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumTextEncoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["SupportsVoiceCloning"] = "true";
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.TextEncoderDim); w.Write(_options.NumTextEncoderLayers);
        w.Write(_options.StyleDim); w.Write(_options.ProsodyDim);
        w.Write(_options.NumMels); w.Write(_options.NumAttentionHeads);
        w.Write(_options.SpeakerEmbeddingDim); w.Write(_options.IsMultiSpeaker);
        w.Write(_options.NumDiffusionSteps); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.TextEncoderDim = r.ReadInt32(); _options.NumTextEncoderLayers = r.ReadInt32();
        _options.StyleDim = r.ReadInt32(); _options.ProsodyDim = r.ReadInt32();
        _options.NumMels = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.SpeakerEmbeddingDim = r.ReadInt32(); _options.IsMultiSpeaker = r.ReadBoolean();
        _options.NumDiffusionSteps = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new StyleTTS2<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodeText(string text)
    {
        // Simple character-level encoding
        int dim = _options.TextEncoderDim;
        var encoded = new Tensor<T>(new[] { text.Length * dim });
        for (int i = 0; i < text.Length; i++)
        {
            int charIdx = text[i] % dim;
            encoded[i * dim + charIdx] = NumOps.FromDouble(1.0);
        }
        return encoded;
    }

    private Tensor<T> ApplyStyle(Tensor<T> encoded, double speakingRate, double pitch,
        string? emotion = null, double emotionIntensity = 0.5)
    {
        // Scale encoded features by speaking rate and pitch
        int styleDim = _options.StyleDim;
        var styled = new Tensor<T>(new[] { encoded.Length + styleDim });
        for (int i = 0; i < encoded.Length; i++)
            styled[i] = NumOps.FromDouble(NumOps.ToDouble(encoded[i]) * speakingRate);
        // Add style vector
        styled[encoded.Length] = NumOps.FromDouble(speakingRate);
        if (styleDim > 1) styled[encoded.Length + 1] = NumOps.FromDouble(pitch);
        if (emotion is not null && styleDim > 2)
        {
            int emotionHash = emotion.GetHashCode() % (styleDim - 2);
            if (emotionHash < 0) emotionHash += styleDim - 2;
            styled[encoded.Length + 2 + emotionHash] = NumOps.FromDouble(emotionIntensity);
        }
        return styled;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(StyleTTS2<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
