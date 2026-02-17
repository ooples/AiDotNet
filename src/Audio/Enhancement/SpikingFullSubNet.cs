using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Spiking-FullSubNet speech enhancement model using spiking neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Spiking-FullSubNet (2024) replaces conventional RNNs with spiking neural network (SNN)
/// layers in the FullSubNet architecture. SNNs use biologically-inspired binary spike events
/// instead of continuous activations, achieving comparable speech enhancement quality with
/// significantly reduced energy consumption, making it suitable for neuromorphic hardware.
/// </para>
/// <para>
/// <b>For Beginners:</b> Spiking-FullSubNet cleans up noisy audio using a brain-inspired approach.
/// Instead of traditional neural networks that pass numbers between layers, it uses "spikes"
/// (on/off signals) like real neurons. This means it can run much more efficiently on
/// specialized hardware while still producing clean, clear speech.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 257, outputSize: 257);
/// var model = new SpikingFullSubNet&lt;float&gt;(arch, "spiking_fullsubnet.onnx");
/// var clean = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// </remarks>
public class SpikingFullSubNet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly SpikingFullSubNetOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;

    #endregion

    #region Constructors

    /// <summary>Creates a Spiking-FullSubNet model in ONNX inference mode.</summary>
    public SpikingFullSubNet(NeuralNetworkArchitecture<T> architecture, string modelPath, SpikingFullSubNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SpikingFullSubNetOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>Creates a Spiking-FullSubNet model in native training mode.</summary>
    public SpikingFullSubNet(NeuralNetworkArchitecture<T> architecture, SpikingFullSubNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SpikingFullSubNetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    internal static async Task<SpikingFullSubNet<T>> CreateAsync(SpikingFullSubNetOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new SpikingFullSubNetOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("spiking_fullsubnet", "spiking_fullsubnet.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins);
        return new SpikingFullSubNet<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => _options.FftSize;

    #endregion

    #region IAudioEnhancer Methods

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        EnhancementStrength = _options.EnhancementStrength;
        var stft = ComputeSTFT(audio);
        Tensor<T> mask;
        if (IsOnnxMode && OnnxEncoder is not null)
            mask = OnnxEncoder.Run(stft);
        else
            mask = Predict(stft);
        var enhanced = ApplyMask(stft, mask);
        return ComputeISTFT(enhanced, audio.Length);
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
        => Enhance(audio);

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        ThrowIfDisposed();
        _streamingBuffer ??= [];
        for (int i = 0; i < audioChunk.Length; i++)
            _streamingBuffer.Add(audioChunk[i]);

        int frameSize = _options.FftSize;
        if (_streamingBuffer.Count < frameSize)
            return new Tensor<T>([0]);

        var frame = new Tensor<T>([frameSize]);
        for (int i = 0; i < frameSize; i++)
            frame[i] = _streamingBuffer[i];
        _streamingBuffer.RemoveRange(0, _options.HopLength);
        return Enhance(frame);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        base.ResetState();
        _streamingBuffer = null;
    }

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio) { }

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpikingFullSubNetLayers(
                numFreqBins: _options.NumFreqBins,
                fullBandHiddenSize: _options.FullBandHiddenSize,
                subBandHiddenSize: _options.SubBandHiddenSize,
                fullBandLayers: _options.NumFullBandLayers,
                subBandLayers: _options.NumSubBandLayers,
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
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
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
        int idx = 0;
        foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => ComputeSTFT(rawAudio);
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SpikingFullSubNet-Native" : "SpikingFullSubNet-ONNX",
            Description = "Spiking-FullSubNet speech enhancement with spiking neural networks (2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumFreqBins,
            Complexity = _options.NumFullBandLayers + _options.NumSubBandLayers
        };
        m.AdditionalInfo["SpikingThreshold"] = _options.SpikingThreshold.ToString();
        m.AdditionalInfo["TimeConstant"] = _options.TimeConstant.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.NumFreqBins); w.Write(_options.NumFullBandLayers); w.Write(_options.FullBandHiddenSize);
        w.Write(_options.NumSubBandLayers); w.Write(_options.SubBandHiddenSize);
        w.Write(_options.SpikingThreshold); w.Write(_options.TimeConstant);
        w.Write(_options.EnhancementStrength); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.NumFreqBins = r.ReadInt32(); _options.NumFullBandLayers = r.ReadInt32(); _options.FullBandHiddenSize = r.ReadInt32();
        _options.NumSubBandLayers = r.ReadInt32(); _options.SubBandHiddenSize = r.ReadInt32();
        _options.SpikingThreshold = r.ReadDouble(); _options.TimeConstant = r.ReadDouble();
        _options.EnhancementStrength = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new SpikingFullSubNet<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> ComputeSTFT(Tensor<T> audio)
    {
        _stft.MagnitudeAndPhase(audio, out var magnitude, out var phase);
        _lastPhase = phase;
        return magnitude;
    }

    private Tensor<T> ApplyMask(Tensor<T> stft, Tensor<T> mask)
    {
        var result = new Tensor<T>(stft.Shape);
        for (int i = 0; i < Math.Min(stft.Length, mask.Length); i++)
            result[i] = NumOps.Multiply(stft[i], mask[i]);
        return result;
    }

    private Tensor<T> ComputeISTFT(Tensor<T> magnitude, int originalLength)
    {
        if (_lastPhase is null)
            throw new InvalidOperationException("Phase not available. Call ComputeSTFT first.");
        return _stft.InverseFromMagnitudeAndPhase(magnitude, _lastPhase, originalLength);
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

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SpikingFullSubNet<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
