using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// CosyVoice2 scalable streaming TTS model from Alibaba.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CosyVoice2 (Du et al., 2024, Alibaba) uses a finite scalar quantization (FSQ) codec
/// with a flow-matching decoder to achieve natural-sounding speech with very low latency.
/// It supports zero-shot voice cloning from a few seconds of reference audio, cross-lingual
/// synthesis, and fine-grained emotion control.
/// </para>
/// <para>
/// <b>For Beginners:</b> CosyVoice2 converts text into natural-sounding speech. What makes
/// it special is that it can clone anyone's voice from just a few seconds of audio, speak
/// in different languages, and add emotions. It's also fast enough for real-time applications
/// like voice assistants and audiobooks.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 1);
/// var model = new CosyVoice2&lt;float&gt;(arch, "cosyvoice2.onnx");
/// var audio = model.Synthesize("Hello, how are you?");
/// var cloned = model.SynthesizeWithVoiceCloning("Hi!", referenceAudio);
/// </code>
/// </para>
/// </remarks>
public class CosyVoice2<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    #region Fields

    private readonly CosyVoice2Options _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ITextToSpeech Properties

    /// <inheritdoc />
    public IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; } = new[]
    {
        new VoiceInfo<T> { Id = "default", Name = "Default", Language = "zh", Gender = VoiceGender.Neutral, Style = "neutral" }
    };

    /// <inheritdoc />
    public bool SupportsVoiceCloning => true;

    /// <inheritdoc />
    public bool SupportsEmotionControl => true;

    /// <inheritdoc />
    public bool SupportsStreaming => true;

    #endregion

    #region Constructors

    /// <summary>Creates a CosyVoice2 model in ONNX inference mode.</summary>
    public CosyVoice2(NeuralNetworkArchitecture<T> architecture, string modelPath, CosyVoice2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new CosyVoice2Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates a CosyVoice2 model in native training mode.</summary>
    public CosyVoice2(NeuralNetworkArchitecture<T> architecture, CosyVoice2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CosyVoice2Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<CosyVoice2<T>> CreateAsync(CosyVoice2Options? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new CosyVoice2Options();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("cosyvoice2", $"cosyvoice2_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new CosyVoice2<T>(arch, mp, options);
    }

    #endregion

    #region ITextToSpeech Methods

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
    public Tensor<T> SynthesizeWithVoiceCloning(string text, Tensor<T> referenceAudio,
        double speakingRate = 1.0, double pitch = 0.0)
    {
        ThrowIfDisposed();
        var speakerEmb = ExtractSpeakerEmbedding(referenceAudio);
        var encoded = EncodeText(text);
        var combined = new Tensor<T>(new[] { encoded.Length + speakerEmb.Length });
        for (int i = 0; i < encoded.Length; i++) combined[i] = encoded[i];
        for (int i = 0; i < speakerEmb.Length; i++) combined[encoded.Length + i] = speakerEmb[i];
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
        var features = PreprocessAudio(referenceAudio);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        int embDim = _options.SpeakerEmbeddingDim;
        var emb = new Tensor<T>(new[] { embDim });
        for (int i = 0; i < embDim && i < output.Length; i++) emb[i] = output[i];
        // L2 normalize
        double norm = 0; for (int i = 0; i < embDim; i++) norm += NumOps.ToDouble(emb[i]) * NumOps.ToDouble(emb[i]);
        norm = Math.Sqrt(norm);
        if (norm > 1e-8) for (int i = 0; i < embDim; i++) emb[i] = NumOps.FromDouble(NumOps.ToDouble(emb[i]) / norm);
        return emb;
    }

    /// <inheritdoc />
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
        => new CosyVoice2StreamingSession(this, speakingRate);

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultCosyVoice2Layers(
            textEncoderDim: _options.TextEncoderDim, numTextEncoderLayers: _options.NumTextEncoderLayers,
            decoderDim: _options.DecoderDim, numDecoderLayers: _options.NumDecoderLayers,
            numMels: _options.NumMels, speakerEmbeddingDim: _options.SpeakerEmbeddingDim,
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
            Name = _useNativeMode ? "CosyVoice2-Native" : "CosyVoice2-ONNX",
            Description = $"CosyVoice2 {_options.Variant} streaming TTS (Du et al., 2024, Alibaba)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumTextEncoderLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["SupportsVoiceCloning"] = "true";
        m.AdditionalInfo["SupportsStreaming"] = "true";
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.TextEncoderDim); w.Write(_options.NumTextEncoderLayers);
        w.Write(_options.DecoderDim); w.Write(_options.NumDecoderLayers);
        w.Write(_options.NumMels); w.Write(_options.SpeakerEmbeddingDim);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.TextEncoderDim = r.ReadInt32(); _options.NumTextEncoderLayers = r.ReadInt32();
        _options.DecoderDim = r.ReadInt32(); _options.NumDecoderLayers = r.ReadInt32();
        _options.NumMels = r.ReadInt32(); _options.SpeakerEmbeddingDim = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new CosyVoice2<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> EncodeText(string text)
    {
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
        int styleDim = _options.SpeakerEmbeddingDim;
        var styled = new Tensor<T>(new[] { encoded.Length + styleDim });
        for (int i = 0; i < encoded.Length; i++)
            styled[i] = NumOps.FromDouble(NumOps.ToDouble(encoded[i]) * speakingRate);
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

    #region Streaming Session

    private sealed class CosyVoice2StreamingSession : IStreamingSynthesisSession<T>
    {
        private readonly CosyVoice2<T> _model;
        private readonly double _speakingRate;
        private readonly List<Tensor<T>> _pendingAudio = [];
        private string _textBuffer = string.Empty;
        private bool _disposed;

        public CosyVoice2StreamingSession(CosyVoice2<T> model, double speakingRate)
        { _model = model; _speakingRate = speakingRate; }

        public void FeedText(string textChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(CosyVoice2StreamingSession));
            _textBuffer += textChunk;
            // Synthesize when we have a sentence boundary
            int boundary = _textBuffer.LastIndexOfAny(['.', '!', '?', ',', ';']);
            if (boundary >= 0)
            {
                string toSynthesize = _textBuffer[..(boundary + 1)];
                _textBuffer = _textBuffer[(boundary + 1)..];
                _pendingAudio.Add(_model.Synthesize(toSynthesize, null, _speakingRate));
            }
        }

        public IEnumerable<Tensor<T>> GetAvailableAudio()
        {
            var audio = new List<Tensor<T>>(_pendingAudio);
            _pendingAudio.Clear();
            return audio;
        }

        public IEnumerable<Tensor<T>> Finalize()
        {
            if (!string.IsNullOrWhiteSpace(_textBuffer))
            {
                _pendingAudio.Add(_model.Synthesize(_textBuffer, null, _speakingRate));
                _textBuffer = string.Empty;
            }
            var audio = new List<Tensor<T>>(_pendingAudio);
            _pendingAudio.Clear();
            return audio;
        }

        public void Dispose()
        {
            _disposed = true;
            _pendingAudio.Clear();
        }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CosyVoice2<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
