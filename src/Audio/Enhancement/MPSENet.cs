using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// MP-SENet (Multi-Path Speech Enhancement Network) model (Lu et al., 2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MP-SENet predicts both magnitude and phase of the complex spectrogram using parallel
/// estimation paths with cross-domain fusion. It achieves PESQ 3.60 on VoiceBank+DEMAND,
/// surpassing prior single-channel speech enhancement methods.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sound has two components: loudness (magnitude) and timing (phase).
/// Most enhancers only fix loudness, leaving timing artifacts. MP-SENet fixes both at once
/// using two parallel paths that communicate, producing cleaner, more natural audio.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 257, outputSize: 257);
/// var model = new MPSENet&lt;float&gt;(arch, "mpsenet.onnx");
/// var clean = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// </remarks>
public class MPSENet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly MPSENetOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private Tensor<T>? _noiseProfile;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    public MPSENet(NeuralNetworkArchitecture<T> architecture, string modelPath, MPSENetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MPSENetOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        int nFft = NextPowerOfTwo(_options.FFTSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FFTSize <= nFft ? _options.FFTSize : null);
        InitializeLayers();
    }

    public MPSENet(NeuralNetworkArchitecture<T> architecture, MPSENetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MPSENetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FFTSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FFTSize <= nFft ? _options.FFTSize : null);
        InitializeLayers();
    }

    internal static async Task<MPSENet<T>> CreateAsync(MPSENetOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new MPSENetOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("mpsenet", $"mpsenet_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins);
        return new MPSENet<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => _options.FFTSize;

    #endregion

    #region IAudioEnhancer Methods

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> noisyAudio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(noisyAudio);
        var enhanced = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return PostprocessOutput(enhanced);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> EnhanceAsync(Tensor<T> noisyAudio, CancellationToken cancellationToken = default)
        => Task.Run(() => Enhance(noisyAudio), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        EstimateNoiseProfile(reference);
        return Enhance(audio);
    }

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk) => Enhance(audioChunk);

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        _stft.MagnitudeAndPhase(noiseOnlyAudio, out var magnitude, out _);
        _noiseProfile = magnitude;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultMPSENetLayers(
            hiddenDim: _options.HiddenDim, numLayers: _options.NumLayers,
            numAttentionHeads: _options.NumAttentionHeads, feedForwardDim: _options.FeedForwardDim,
            numFreqBins: _options.NumFreqBins, dropoutRate: _options.DropoutRate));
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
        _stft.MagnitudeAndPhase(rawAudio, out var magnitude, out var phase);
        _lastPhase = phase;
        return magnitude;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> enhancedMagnitude)
    {
        if (_lastPhase is null)
            return enhancedMagnitude;
        return _stft.InverseFromMagnitudeAndPhase(enhancedMagnitude, _lastPhase);
    }

    private static int NextPowerOfTwo(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MP-SENet-Native" : "MP-SENet-ONNX",
            Description = $"MP-SENet {_options.Variant} speech enhancement (Lu et al., 2023)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumFreqBins, Complexity = _options.NumLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.HiddenDim); w.Write(_options.NumLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.FeedForwardDim);
        w.Write(_options.NumFreqBins); w.Write(_options.FFTSize);
        w.Write(_options.HopLength); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.HiddenDim = r.ReadInt32(); _options.NumLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.FeedForwardDim = r.ReadInt32();
        _options.NumFreqBins = r.ReadInt32(); _options.FFTSize = r.ReadInt32();
        _options.HopLength = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
        int nFft = NextPowerOfTwo(_options.FFTSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FFTSize <= nFft ? _options.FFTSize : null);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new MPSENet<T>(Architecture, mp, _options);
        return new MPSENet<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MPSENet<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
