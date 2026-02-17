using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// HiFi-GAN neural vocoder (Kong et al., 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// HiFi-GAN is a GAN-based vocoder that converts mel-spectrograms to high-fidelity waveforms.
/// Using multi-period and multi-scale discriminators, it achieves MOS 4.23 at 13.9x real-time
/// on CPU (V1 variant). HiFi-GAN is the most widely-used vocoder in production TTS systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> A vocoder is the final step in text-to-speech. After a model like
/// Tacotron or StyleTTS creates a spectrogram (a picture of sound), the vocoder turns it
/// into actual audio. HiFi-GAN is like a highly skilled musician who can read sheet music
/// (the spectrogram) and play it perfectly at super speed.
///
/// Why "GAN"? It was trained using two competing networks:
/// - A Generator that creates audio
/// - Discriminators that try to tell real audio from generated audio
/// This competition pushes the generator to make increasingly realistic audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 1);
/// var vocoder = new HiFiGAN&lt;float&gt;(arch, "hifigan_v1.onnx");
/// var audio = vocoder.Synthesize("This text is converted elsewhere; pass mel-spec directly.");
/// // Or use directly with mel-spectrogram:
/// var waveform = vocoder.Predict(melSpectrogram);
/// </code>
/// </para>
/// </remarks>
public class HiFiGAN<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    #region Fields

    private readonly HiFiGANOptions _options;
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
    public bool SupportsVoiceCloning => false;

    /// <inheritdoc />
    public bool SupportsEmotionControl => false;

    /// <inheritdoc />
    public bool SupportsStreaming => true;

    #endregion

    #region Constructors

    /// <summary>Creates a HiFi-GAN vocoder in ONNX inference mode.</summary>
    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, HiFiGANOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new HiFiGANOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates a HiFi-GAN vocoder in native training mode.</summary>
    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, HiFiGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new HiFiGANOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<HiFiGAN<T>> CreateAsync(HiFiGANOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new HiFiGANOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("hifigan", $"hifigan_{options.Variant.ToLowerInvariant()}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new HiFiGAN<T>(arch, mp, options);
    }

    #endregion

    #region ITextToSpeech

    /// <inheritdoc />
    public Tensor<T> Synthesize(string text, string? voiceId = null, double speakingRate = 1.0, double pitch = 0.0)
    {
        ThrowIfDisposed();
        // HiFi-GAN is a vocoder; it expects mel-spectrogram input, not text.
        // For text input, encode as mel-shaped features.
        var melFeatures = TextToMelFeatures(text);
        return VocoderForward(melFeatures);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> SynthesizeAsync(string text, string? voiceId = null, double speakingRate = 1.0,
        double pitch = 0.0, CancellationToken cancellationToken = default)
        => Task.Run(() => Synthesize(text, voiceId, speakingRate, pitch), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoiceCloning(string text, Tensor<T> referenceAudio, double speakingRate = 1.0, double pitch = 0.0)
        => throw new NotSupportedException("HiFi-GAN is a vocoder and does not support voice cloning. Use StyleTTS 2 + HiFi-GAN.");

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithEmotion(string text, string emotion, double emotionIntensity = 0.5,
        string? voiceId = null, double speakingRate = 1.0)
        => throw new NotSupportedException("HiFi-GAN is a vocoder and does not support emotion control.");

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
        => throw new NotSupportedException("HiFi-GAN is a vocoder and does not extract speaker embeddings.");

    /// <inheritdoc />
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
        => new HiFiGANStreamingSession(this);

    /// <summary>
    /// Converts a mel-spectrogram to waveform audio (the primary vocoder function).
    /// </summary>
    /// <param name="melSpectrogram">Mel-spectrogram tensor [frames, numMels] or [numMels, frames].</param>
    /// <returns>Audio waveform tensor [samples].</returns>
    public Tensor<T> VocoderForward(Tensor<T> melSpectrogram)
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultHiFiGANLayers(
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

    protected override Tensor<T> PostprocessOutput(Tensor<T> o)
    {
        // Tanh clamp output to [-1, 1]
        for (int i = 0; i < o.Length; i++)
        {
            double v = NumOps.ToDouble(o[i]);
            o[i] = NumOps.FromDouble(Math.Tanh(v));
        }
        return o;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "HiFi-GAN-Native" : "HiFi-GAN-ONNX",
            Description = $"HiFi-GAN {_options.Variant} neural vocoder (Kong et al., 2020)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.UpsampleRates.Length
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.NumMels); w.Write(_options.HopLength);
        w.Write(_options.UpsampleInitialChannel); w.Write(_options.NumResBlocks);
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
        int numRates = r.ReadInt32();
        _options.UpsampleRates = new int[numRates];
        for (int i = 0; i < numRates; i++) _options.UpsampleRates[i] = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new HiFiGAN<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> TextToMelFeatures(string text)
    {
        // Placeholder: map text to mel-shaped features for vocoder
        int numFrames = text.Length * 5; // rough estimate
        int numMels = _options.NumMels;
        var mel = new Tensor<T>(new[] { numFrames * numMels });
        for (int i = 0; i < text.Length; i++)
        {
            int frame = i * 5;
            int bin = text[i] % numMels;
            if (frame * numMels + bin < mel.Length)
                mel[frame * numMels + bin] = NumOps.FromDouble(1.0);
        }
        return mel;
    }

    #endregion

    #region Streaming Session

    private sealed class HiFiGANStreamingSession : IStreamingSynthesisSession<T>
    {
        private readonly HiFiGAN<T> _model;
        private readonly List<Tensor<T>> _pendingMels = new();
        private bool _disposed;

        public HiFiGANStreamingSession(HiFiGAN<T> model) { _model = model; }

        public void FeedText(string textChunk)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HiFiGANStreamingSession));
            var mel = _model.TextToMelFeatures(textChunk);
            _pendingMels.Add(mel);
        }

        public IEnumerable<Tensor<T>> GetAvailableAudio()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HiFiGANStreamingSession));
            foreach (var mel in _pendingMels) yield return _model.VocoderForward(mel);
            _pendingMels.Clear();
        }

        public IEnumerable<Tensor<T>> Finalize()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(HiFiGANStreamingSession));
            var results = new List<Tensor<T>>(GetAvailableAudio());
            _disposed = true;
            return results;
        }

        public void Dispose() { _disposed = true; }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(HiFiGAN<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
