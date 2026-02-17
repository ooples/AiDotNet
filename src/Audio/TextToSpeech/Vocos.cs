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
/// Vocos frequency-domain neural vocoder (Siuzdak, 2023, Charactr).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Vocos (Siuzdak, 2023) replaces traditional time-domain waveform generation with
/// frequency-domain ISTFT reconstruction. It predicts STFT magnitude and phase, then
/// applies inverse STFT to get the waveform. This approach is more efficient and avoids
/// phase-related artifacts. Vocos supports both mel-spectrogram and neural codec (EnCodec)
/// inputs, making it versatile for TTS and audio generation pipelines.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional vocoders generate audio one tiny sample at a time
/// (like filling in pixels one by one). Vocos works differently:
///
/// 1. It predicts the frequencies present in each audio frame (like describing a chord)
/// 2. It predicts the timing/phase of those frequencies
/// 3. It uses math (inverse FFT) to turn frequencies back into audio
///
/// Benefits:
/// - No buzzy artifacts (common in time-domain vocoders)
/// - Computationally efficient (fewer operations)
/// - Works with mel-spectrograms AND neural codec tokens (EnCodec)
/// - Smaller model for similar quality
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 100, outputSize: 1);
/// var vocoder = new Vocos&lt;float&gt;(arch, "vocos_mel.onnx");
/// var waveform = vocoder.Predict(melSpectrogram); // Direct vocoding
/// </code>
/// </para>
/// </remarks>
public class Vocos<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    #region Fields

    private readonly VocosOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
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

    /// <summary>Creates a Vocos model in ONNX inference mode.</summary>
    public Vocos(NeuralNetworkArchitecture<T> architecture, string modelPath, VocosOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VocosOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        InitializeLayers();
    }

    /// <summary>Creates a Vocos model in native training mode.</summary>
    public Vocos(NeuralNetworkArchitecture<T> architecture, VocosOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VocosOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<Vocos<T>> CreateAsync(VocosOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new VocosOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("vocos", $"vocos_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: 1);
        return new Vocos<T>(arch, mp, options);
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
        => throw new NotSupportedException("Vocos is a vocoder; use it with Predict(melSpectrogram) for mel-to-waveform conversion.");

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithEmotion(string text, string emotion, double emotionIntensity = 0.5,
        string? voiceId = null, double speakingRate = 1.0)
        => throw new NotSupportedException("Vocos is a vocoder and does not support emotion control.");

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
        => throw new NotSupportedException("Vocos is a vocoder and does not support speaker embedding extraction.");

    /// <inheritdoc />
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
        => new VocosStreamingSession(this);

    /// <summary>
    /// Converts a mel-spectrogram to a waveform via ISTFT reconstruction (vocoder mode).
    /// </summary>
    /// <param name="melSpectrogram">Mel-spectrogram tensor [frames, numMels] or [numMels].</param>
    /// <returns>Audio waveform tensor [samples].</returns>
    public Tensor<T> Vocode(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(melSpectrogram) : Predict(melSpectrogram);
    }

    /// <summary>
    /// Reconstructs audio from EnCodec tokens (codec vocoder mode).
    /// </summary>
    /// <param name="codecTokens">EnCodec discrete token tensor.</param>
    /// <returns>Audio waveform tensor [samples].</returns>
    public Tensor<T> DecodeCodecTokens(Tensor<T> codecTokens)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(codecTokens) : Predict(codecTokens);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultVocosLayers(
            numMels: _options.NumMels, hiddenDim: _options.HiddenDim,
            numBackboneBlocks: _options.NumBackboneBlocks, intermediateDim: _options.IntermediateDim,
            numFrequencyBins: _options.NumFrequencyBins, dropoutRate: _options.DropoutRate));
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
        _optimizer?.UpdateParameters(Layers);
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
            Name = _useNativeMode ? "Vocos-Native" : "Vocos-ONNX",
            Description = $"Vocos {_options.Variant} ISTFT vocoder (Siuzdak, 2023, Charactr)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumBackboneBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.NumMels); w.Write(_options.HiddenDim);
        w.Write(_options.NumBackboneBlocks); w.Write(_options.IntermediateDim);
        w.Write(_options.ConvKernelSize); w.Write(_options.FFTSize);
        w.Write(_options.HopLength); w.Write(_options.NumFrequencyBins);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.NumMels = r.ReadInt32(); _options.HiddenDim = r.ReadInt32();
        _options.NumBackboneBlocks = r.ReadInt32(); _options.IntermediateDim = r.ReadInt32();
        _options.ConvKernelSize = r.ReadInt32(); _options.FFTSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.NumFrequencyBins = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new Vocos<T>(Architecture, _options);

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

    private sealed class VocosStreamingSession : IStreamingSynthesisSession<T>
    {
        private readonly Vocos<T> _model;
        private readonly List<string> _textChunks = new();
        private bool _disposed;

        public VocosStreamingSession(Vocos<T> model) { _model = model; }

        public void FeedText(string textChunk) { if (_disposed) throw new ObjectDisposedException(nameof(VocosStreamingSession)); _textChunks.Add(textChunk); }

        public IEnumerable<Tensor<T>> GetAvailableAudio()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(VocosStreamingSession));
            foreach (var chunk in _textChunks) yield return _model.Synthesize(chunk);
            _textChunks.Clear();
        }

        public IEnumerable<Tensor<T>> Finalize()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(VocosStreamingSession));
            var result = GetAvailableAudio().ToList();
            _disposed = true;
            return result;
        }

        public void Dispose() { _disposed = true; }
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Vocos<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
