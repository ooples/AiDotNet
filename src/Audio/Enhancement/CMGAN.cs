using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// CMGAN (Conformer-based Metric GAN) for speech enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CMGAN (Cao et al., INTERSPEECH 2022) combines a conformer-based generator with a metric
/// discriminator for high-quality speech enhancement, achieving PESQ 3.41 and STOI 0.97
/// on the VoiceBank-DEMAND dataset.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>U-Net encoder</b>: Compresses the noisy spectrogram with convolutional blocks</item>
/// <item><b>Conformer bottleneck</b>: Self-attention + convolution for global context</item>
/// <item><b>U-Net decoder</b>: Reconstructs clean spectrogram with skip connections</item>
/// <item><b>Metric discriminator</b>: Judges enhancement quality during training</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> CMGAN uses a competition between two networks: a "generator" that
/// cleans audio and a "discriminator" that judges quality. The generator uses Conformer
/// layers that combine attention (understanding context) with convolution (local patterns).
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 201, outputSize: 201);
/// var model = new CMGAN&lt;float&gt;(arch, "cmgan_voicebank.onnx");
/// var clean = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "CMGAN: Conformer-based Metric GAN for Speech Enhancement" (Cao et al., INTERSPEECH 2022)</item>
/// <item>Repository: https://github.com/ruizhecao96/CMGAN</item>
/// </list>
/// </para>
/// </remarks>
public class CMGAN<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly CMGANOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ShortTimeFourierTransform<T> _stft;
    private Tensor<T>? _lastPhase;
    private Tensor<T>? _noiseProfile;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a CMGAN model in ONNX inference mode.
    /// </summary>
    public CMGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, CMGANOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CMGANOptions();
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
    /// Creates a CMGAN model in native training mode.
    /// </summary>
    public CMGAN(NeuralNetworkArchitecture<T> architecture, CMGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CMGANOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        int nFft = NextPowerOfTwo(_options.FftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _options.HopLength,
            windowLength: _options.FftSize <= nFft ? _options.FftSize : null);
        InitializeLayers();
    }

    /// <summary>
    /// Downloads and creates a CMGAN model asynchronously.
    /// </summary>
    internal static async Task<CMGAN<T>> CreateAsync(
        CMGANOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new CMGANOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("cmgan", "cmgan_voicebank.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFreqBins, outputSize: options.NumFreqBins);
        return new CMGAN<T>(arch, mp, options);
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

        // Apply spectral subtraction if noise profile is available
        if (_noiseProfile is not null && _noiseProfile.Length == stft.Length)
        {
            for (int i = 0; i < stft.Length; i++)
            {
                T subtracted = NumOps.Subtract(stft[i], _noiseProfile[i]);
                stft[i] = NumOps.GreaterThan(subtracted, NumOps.Zero) ? subtracted : NumOps.Zero;
            }
        }

        Tensor<T> mask;
        if (IsOnnxMode && OnnxEncoder is not null)
            mask = OnnxEncoder.Run(stft);
        else
            mask = Predict(stft);
        var enhanced = ApplyMask(stft, mask);

        // Apply enhancement strength blending: output = strength * enhanced + (1 - strength) * original
        var result = ComputeISTFT(enhanced, audio.Length);
        double strength = _options.EnhancementStrength;
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
        // Use the reference signal as a noise profile estimate before enhancement
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
        // Compute STFT of noise-only audio to get spectral noise floor
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCMGANLayers(
                numFreqBins: _options.NumFreqBins,
                conformerDim: _options.ConformerDim,
                numConformerLayers: _options.NumConformerLayers,
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
            Name = _useNativeMode ? "CMGAN-Native" : "CMGAN-ONNX",
            Description = "CMGAN Conformer-based Metric GAN (Cao et al., INTERSPEECH 2022)",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.NumFreqBins,
            Complexity = _options.NumConformerLayers
        };
        m.AdditionalInfo["Architecture"] = "CMGAN";
        m.AdditionalInfo["ConformerDim"] = _options.ConformerDim.ToString();
        m.AdditionalInfo["NumConformerLayers"] = _options.NumConformerLayers.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.FftSize); w.Write(_options.HopLength);
        w.Write(_options.NumFreqBins); w.Write(_options.ConformerDim); w.Write(_options.NumConformerLayers);
        w.Write(_options.NumAttentionHeads); w.Write(_options.EnhancementStrength); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.FftSize = r.ReadInt32(); _options.HopLength = r.ReadInt32();
        _options.NumFreqBins = r.ReadInt32(); _options.ConformerDim = r.ReadInt32(); _options.NumConformerLayers = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32(); _options.EnhancementStrength = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new CMGAN<T>(Architecture, mp, _options);
        return new CMGAN<T>(Architecture, _options);
    }

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

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CMGAN<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
