using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Audio Super-Resolution model for upsampling low-resolution audio to high-resolution
/// (Kuleshov et al., 2017; Li et al., 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio Super-Resolution uses deep neural networks to predict missing high-frequency
/// content in low-resolution audio. Given input at a low sample rate (e.g., 8 kHz telephone
/// quality), it reconstructs audio at a higher sample rate (e.g., 44.1 kHz studio quality)
/// by predicting the missing frequency bands. The architecture uses residual blocks with
/// attention modules to capture both local and global spectral patterns.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio Super-Resolution is like AI-powered upscaling for sound.
/// Just as image super-resolution makes blurry photos sharper, this model makes low-quality
/// audio sound clearer and more detailed.
///
/// Common uses:
/// - Upscaling old telephone recordings (8 kHz to 44.1 kHz)
/// - Recovering quality from heavily compressed audio (MP3 at 64 kbps)
/// - Enhancing voice recordings from cheap microphones
/// - Restoring bandwidth-limited historical recordings
///
/// How it works:
/// 1. Takes a low-resolution audio waveform as input
/// 2. Passes through residual blocks that learn to predict missing high-frequency content
/// 3. Outputs a high-resolution waveform with restored detail
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 1);
/// var model = new AudioSuperResolution&lt;float&gt;(arch, "audio_sr.onnx");
/// var highRes = model.Enhance(lowResAudio);
/// </code>
/// </para>
/// </remarks>
public class AudioSuperResolution<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly AudioSuperResolutionOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => _options.InputSampleRate; // one second of input latency

    #endregion

    #region Constructors

    /// <summary>Creates an Audio Super-Resolution model in ONNX inference mode.</summary>
    public AudioSuperResolution(NeuralNetworkArchitecture<T> architecture, string modelPath, AudioSuperResolutionOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AudioSuperResolutionOptions();
        _useNativeMode = false;
        base.SampleRate = _options.OutputSampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates an Audio Super-Resolution model in native training mode.</summary>
    public AudioSuperResolution(NeuralNetworkArchitecture<T> architecture, AudioSuperResolutionOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AudioSuperResolutionOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.OutputSampleRate;
        InitializeLayers();
    }

    internal static async Task<AudioSuperResolution<T>> CreateAsync(AudioSuperResolutionOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new AudioSuperResolutionOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("audio_super_resolution", $"audio_sr_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: 1);
        return new AudioSuperResolution<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return ApplyStrength(audio, output);
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference) => Enhance(audio);

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk) => Enhance(audioChunk);

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio) { /* Not applicable for super-resolution */ }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultAudioSuperResolutionLayers(
            hiddenDim: _options.HiddenDim, numResBlocks: _options.NumResBlocks,
            numHeads: _options.NumHeads, numAttentionLayers: _options.NumAttentionLayers,
            upsampleFactor: _options.UpsampleFactor, dropoutRate: _options.DropoutRate));
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
            Name = _useNativeMode ? "AudioSuperResolution-Native" : "AudioSuperResolution-ONNX",
            Description = $"Audio Super-Resolution {_options.Variant} ({_options.InputSampleRate / 1000}kHz -> {_options.OutputSampleRate / 1000}kHz)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = 1, Complexity = _options.NumResBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["UpsampleFactor"] = _options.UpsampleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.InputSampleRate); w.Write(_options.OutputSampleRate);
        w.Write(_options.UpsampleFactor); w.Write(_options.Variant);
        w.Write(_options.HiddenDim); w.Write(_options.NumResBlocks);
        w.Write(_options.NumHeads); w.Write(_options.NumAttentionLayers);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.InputSampleRate = r.ReadInt32(); _options.OutputSampleRate = r.ReadInt32();
        _options.UpsampleFactor = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.HiddenDim = r.ReadInt32(); _options.NumResBlocks = r.ReadInt32();
        _options.NumHeads = r.ReadInt32(); _options.NumAttentionLayers = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new AudioSuperResolution<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> ApplyStrength(Tensor<T> original, Tensor<T> enhanced)
    {
        double strength = EnhancementStrength;
        if (Math.Abs(strength - 1.0) < 1e-9) return enhanced;

        // Blend original and enhanced based on strength
        var result = new Tensor<T>(enhanced.Shape);
        int len = Math.Min(original.Length, enhanced.Length);
        for (int i = 0; i < enhanced.Length; i++)
        {
            double orig = i < original.Length ? NumOps.ToDouble(original[i % original.Length]) : 0.0;
            double enh = NumOps.ToDouble(enhanced[i]);
            result[i] = NumOps.FromDouble(orig + (enh - orig) * strength);
        }
        return result;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioSuperResolution<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
