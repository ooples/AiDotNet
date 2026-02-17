using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Demucs for Noise - real-time noise suppression using the Demucs architecture (Defossez et al., 2020, Meta).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Demucs for Noise adapts the Demucs U-Net encoder-decoder architecture for real-time speech
/// denoising. Operating in the time domain with skip connections and LSTM bottleneck, it
/// achieves high-quality noise removal at 40ms latency. It was developed at Meta for
/// real-time communication applications.
/// </para>
/// <para>
/// <b>For Beginners:</b> Demucs for Noise works like a music separator, but instead of
/// separating instruments, it separates clean speech from background noise. Feed it a noisy
/// recording and it outputs just the clean speech.
///
/// How it works:
/// 1. Encoder: Progressively compresses audio into a compact representation
/// 2. LSTM bottleneck: Captures temporal patterns in the compressed audio
/// 3. Decoder: Reconstructs clean audio with skip connections from the encoder
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 1);
/// var model = new DemucsNoise&lt;float&gt;(arch, "demucs_noise.onnx");
/// var clean = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// </remarks>
public class DemucsNoise<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly DemucsNoiseOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels => _options.NumChannels;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => (int)Math.Pow(_options.Stride, _options.Depth);

    #endregion

    #region Constructors

    /// <summary>Creates a Demucs Noise model in ONNX inference mode.</summary>
    public DemucsNoise(NeuralNetworkArchitecture<T> architecture, string modelPath, DemucsNoiseOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DemucsNoiseOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates a Demucs Noise model in native training mode.</summary>
    public DemucsNoise(NeuralNetworkArchitecture<T> architecture, DemucsNoiseOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DemucsNoiseOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<DemucsNoise<T>> CreateAsync(DemucsNoiseOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new DemucsNoiseOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("demucs_noise", $"demucs_noise_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: 1);
        return new DemucsNoise<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference) => Enhance(audio);

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk) => Enhance(audioChunk);

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio) { /* Demucs learns noise implicitly */ }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultDemucsNoiseLayers(
            hiddenChannels: _options.HiddenChannels, depth: _options.Depth,
            lstmHiddenSize: _options.LSTMHiddenSize, numLSTMLayers: _options.NumLSTMLayers,
            channelGrowth: _options.ChannelGrowth, dropoutRate: _options.DropoutRate));
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

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "DemucsNoise-Native" : "DemucsNoise-ONNX",
            Description = $"Demucs for Noise {_options.Variant} (Defossez et al., 2020, Meta)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = 1, Complexity = _options.Depth
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["Depth"] = _options.Depth.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.HiddenChannels); w.Write(_options.Depth);
        w.Write(_options.LSTMHiddenSize); w.Write(_options.NumLSTMLayers);
        w.Write(_options.ChannelGrowth); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.HiddenChannels = r.ReadInt32(); _options.Depth = r.ReadInt32();
        _options.LSTMHiddenSize = r.ReadInt32(); _options.NumLSTMLayers = r.ReadInt32();
        _options.ChannelGrowth = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new DemucsNoise<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DemucsNoise<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
