using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) for speech enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FullSubNet+ (Chen et al., ICASSP 2022) improves upon FullSubNet by using channel-attention-based
/// full-band models and redesigned sub-band inputs. It achieves PESQ 3.25 and STOI 0.96 on the
/// DNS Challenge dataset at 16 kHz.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>Full-band model</b>: LSTM with channel attention across all frequency bins</item>
/// <item><b>Sub-band model</b>: LSTM processing local frequency neighborhoods</item>
/// <item><b>Fusion</b>: Full-band output guides sub-band processing</item>
/// <item><b>Complex mask estimation</b>: Predicts both magnitude and phase corrections</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> FullSubNet+ is a two-part system for cleaning up noisy speech:
/// one part looks at the big picture (all frequencies), and another focuses on fine details
/// (small frequency groups). Together they produce cleaner audio than either alone.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 257, outputSize: 257);
/// var model = new FullSubNetPlus&lt;float&gt;(arch, "fullsubnet_plus.onnx");
/// var clean = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms" (Chen et al., ICASSP 2022)</item>
/// <item>Repository: https://github.com/hit-thusz-RookieCJ/FullSubNet-plus</item>
/// </list>
/// </para>
/// </remarks>
public class FullSubNetPlus<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly FullSubNetPlusOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a FullSubNet+ model in ONNX inference mode.
    /// </summary>
    public FullSubNetPlus(NeuralNetworkArchitecture<T> architecture, string modelPath, FullSubNetPlusOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FullSubNetPlusOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FullSubNet+ model in native training mode.
    /// </summary>
    public FullSubNetPlus(NeuralNetworkArchitecture<T> architecture, FullSubNetPlusOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FullSubNetPlusOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>
    /// Downloads and creates a FullSubNet+ model asynchronously.
    /// </summary>
    internal static async Task<FullSubNetPlus<T>> CreateAsync(
        FullSubNetPlusOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new FullSubNetPlusOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("fullsubnetplus", "fullsubnet_plus_16k.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins);
        return new FullSubNetPlus<T>(arch, mp, options);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFullSubNetPlusLayers(
                numFreqBins: _options.NumFreqBins,
                fullBandHiddenSize: _options.FullBandHiddenSize,
                subBandHiddenSize: _options.SubBandHiddenSize,
                fullBandLayers: _options.FullBandLayers,
                subBandLayers: _options.SubBandLayers,
                dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null)
            return OnnxEncoder.Run(input);
        var c = input;
        foreach (var l in Layers) c = l.Forward(c);
        return c;
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
            Name = _useNativeMode ? "FullSubNet+-Native" : "FullSubNet+-ONNX",
            Description = "FullSubNet+ Channel Attention Speech Enhancement (Chen et al., ICASSP 2022)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.NumFreqBins,
            Complexity = _options.FullBandLayers + _options.SubBandLayers
        };
        m.AdditionalInfo["Architecture"] = "FullSubNet+";
        m.AdditionalInfo["FullBandHidden"] = _options.FullBandHiddenSize.ToString();
        m.AdditionalInfo["SubBandHidden"] = _options.SubBandHiddenSize.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.NumFreqBins); w.Write(_options.FullBandLayers); w.Write(_options.FullBandHiddenSize);
        w.Write(_options.SubBandLayers); w.Write(_options.SubBandHiddenSize);
        w.Write(_options.EnhancementStrength); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.NumFreqBins = r.ReadInt32(); _options.FullBandLayers = r.ReadInt32(); _options.FullBandHiddenSize = r.ReadInt32();
        _options.SubBandLayers = r.ReadInt32(); _options.SubBandHiddenSize = r.ReadInt32();
        _options.EnhancementStrength = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new FullSubNetPlus<T>(Architecture, _options);

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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FullSubNetPlus<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
