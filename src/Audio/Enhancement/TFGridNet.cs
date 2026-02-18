using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// TF-GridNet (Time-Frequency GridNet) for speech enhancement and separation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TF-GridNet (Wang et al., ICASSP 2023) applies alternating attention along the time and frequency
/// axes in a grid pattern, achieving 23.4 dB SI-SNRi on WSJ0-2mix and PESQ 3.41 on DNS Challenge.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>Input embedding</b>: Maps complex STFT to per-bin embeddings</item>
/// <item><b>Grid blocks</b>: Each block has intra-frame (frequency) and inter-frame (time) modules</item>
/// <item><b>Intra-frame</b>: LSTM/attention across frequency bins for each time frame</item>
/// <item><b>Inter-frame</b>: LSTM/attention across time frames for each frequency bin</item>
/// <item><b>Output</b>: Reconstructs complex STFT for synthesis</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> TF-GridNet cleans audio by processing a time-frequency grid in two
/// alternating directions: across frequencies (understanding harmonic structure) and across
/// time (tracking how sounds evolve). This grid approach captures both local and global patterns.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 257, outputSize: 257);
/// var model = new TFGridNet&lt;float&gt;(arch, "tfgridnet_16k.onnx");
/// var clean = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "TF-GridNet: Making Time-Frequency Domain Models Great Again" (Wang et al., ICASSP 2023)</item>
/// <item>Repository: https://github.com/espnet/espnet</item>
/// </list>
/// </para>
/// </remarks>
public class TFGridNet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly TFGridNetOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private Tensor<T>? _noiseProfile;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TF-GridNet model in ONNX inference mode.
    /// </summary>
    public TFGridNet(NeuralNetworkArchitecture<T> architecture, string modelPath, TFGridNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TFGridNetOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a TF-GridNet model in native training mode.
    /// </summary>
    public TFGridNet(NeuralNetworkArchitecture<T> architecture, TFGridNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new TFGridNetOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>
    /// Downloads and creates a TF-GridNet model asynchronously.
    /// </summary>
    internal static async Task<TFGridNet<T>> CreateAsync(
        TFGridNetOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new TFGridNetOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("tfgridnet", "tfgridnet_16k.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins);
        return new TFGridNet<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength
    {
        get => _options.EnhancementStrength;
        set => _options.EnhancementStrength = Math.Max(0, Math.Min(1, value));
    }

    /// <inheritdoc />
    public int LatencySamples => _options.FftSize;

    #endregion

    #region IAudioEnhancer Methods

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var stft = ComputeSTFT(audio);

        // Apply spectral subtraction if noise profile is available (per-bin comparison)
        if (_noiseProfile is not null)
        {
            int len = Math.Min(stft.Length, _noiseProfile.Length);
            for (int i = 0; i < len; i++)
            {
                T subtracted = NumOps.Subtract(stft[i], _noiseProfile[i]);
                stft[i] = NumOps.GreaterThan(subtracted, NumOps.Zero) ? subtracted : NumOps.Zero;
            }
        }

        Tensor<T> output;
        if (IsOnnxMode && OnnxEncoder is not null)
            output = OnnxEncoder.Run(stft);
        else
            output = Predict(stft);

        // Apply enhancement strength blending
        var result = ComputeISTFT(output, audio.Length);
        double strength = EnhancementStrength;
        if (strength < 1.0)
        {
            T s = NumOps.FromDouble(strength);
            T inv = NumOps.FromDouble(1.0 - strength);
            for (int i = 0; i < result.Length && i < audio.Length; i++)
            {
                result[i] = NumOps.Add(NumOps.Multiply(s, result[i]), NumOps.Multiply(inv, audio[i]));
            }
        }

        return result;
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        EstimateNoiseProfile(reference);
        return Enhance(audio);
    }

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        ThrowIfDisposed();
        _streamingBuffer ??= [];
        for (int i = 0; i < audioChunk.Length; i++) _streamingBuffer.Add(audioChunk[i]);
        int frameSize = _options.FftSize;
        if (_streamingBuffer.Count < frameSize) return new Tensor<T>([0]);
        var frame = new Tensor<T>([frameSize]);
        for (int i = 0; i < frameSize; i++) frame[i] = _streamingBuffer[i];
        _streamingBuffer.RemoveRange(0, _options.HopLength);
        return Enhance(frame);
    }

    /// <inheritdoc />
    public override void ResetState() { base.ResetState(); _streamingBuffer = null; }

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        _stft.MagnitudeAndPhase(noiseOnlyAudio, out var magnitude, out _);
        _noiseProfile = magnitude;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultTFGridNetLayers(
                numFreqBins: _options.NumFreqBins,
                hiddenDim: _options.HiddenDim,
                embeddingDim: _options.EmbeddingDim,
                numBlocks: _options.NumBlocks,
                numAttentionHeads: _options.NumAttentionHeads,
                dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
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
        _optimizer?.UpdateParameters(Layers);
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
            Name = _useNativeMode ? "TF-GridNet-Native" : "TF-GridNet-ONNX",
            Description = "TF-GridNet Time-Frequency Grid Enhancement (Wang et al., ICASSP 2023)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.NumFreqBins,
            Complexity = _options.NumBlocks
        };
        m.AdditionalInfo["Architecture"] = "TF-GridNet";
        m.AdditionalInfo["HiddenDim"] = _options.HiddenDim.ToString();
        m.AdditionalInfo["NumBlocks"] = _options.NumBlocks.ToString();
        m.AdditionalInfo["EmbeddingDim"] = _options.EmbeddingDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.NumFreqBins); w.Write(_options.HiddenDim); w.Write(_options.EmbeddingDim);
        w.Write(_options.NumBlocks); w.Write(_options.NumAttentionHeads);
        w.Write(_options.EnhancementStrength); w.Write(_options.DropoutRate); w.Write(_options.NumSources);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.NumFreqBins = r.ReadInt32(); _options.HiddenDim = r.ReadInt32(); _options.EmbeddingDim = r.ReadInt32();
        _options.NumBlocks = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.EnhancementStrength = r.ReadDouble(); _options.DropoutRate = r.ReadDouble(); _options.NumSources = r.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new TFGridNet<T>(Architecture, mp, _options);
        return new TFGridNet<T>(Architecture, _options);
    }

    #endregion

    #region Private Helpers

    private Tensor<T> ComputeSTFT(Tensor<T> audio)
    {
        _stft.MagnitudeAndPhase(audio, out var magnitude, out var phase);
        _lastPhase = phase;
        return magnitude;
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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TFGridNet<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
