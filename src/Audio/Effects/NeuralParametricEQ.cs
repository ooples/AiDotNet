using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Neural Parametric EQ model for automatic equalization (Steinmetz et al., 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Neural Parametric EQ uses a neural network to automatically predict optimal parametric
/// EQ settings (gain, frequency, Q for each band) to match a target frequency response.
/// It analyzes input audio and outputs the parameters for a cascaded biquad filter bank,
/// enabling automatic mastering, hearing aid fitting, and frequency matching.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you have audio that sounds too bassy or too bright.
/// Instead of manually turning EQ knobs, this model listens to the audio and automatically
/// figures out the right EQ settings. It predicts gain (louder/quieter), frequency (which
/// range to adjust), and Q (how narrow the adjustment is) for each band.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1025, outputSize: 18);
/// var model = new NeuralParametricEQ&lt;float&gt;(arch, "neural_eq.onnx");
/// var enhanced = model.Enhance(audioWaveform);
/// </code>
/// </para>
/// </remarks>
public class NeuralParametricEQ<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly NeuralParametricEQOptions _options;
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
    public int LatencySamples => _options.FFTSize;

    #endregion

    #region Constructors

    /// <summary>Creates a Neural Parametric EQ model in ONNX inference mode.</summary>
    public NeuralParametricEQ(NeuralNetworkArchitecture<T> architecture, string modelPath, NeuralParametricEQOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new NeuralParametricEQOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates a Neural Parametric EQ model in native training mode.</summary>
    public NeuralParametricEQ(NeuralNetworkArchitecture<T> architecture, NeuralParametricEQOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new NeuralParametricEQOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<NeuralParametricEQ<T>> CreateAsync(NeuralParametricEQOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new NeuralParametricEQOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("neural_parametric_eq", $"neural_eq_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        int numFreqBins = options.FFTSize / 2 + 1;
        int outputSize = options.NumBands * 3; // gain, freq, Q per band
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: numFreqBins, outputSize: outputSize);
        return new NeuralParametricEQ<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var eqParams = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return ApplyEQ(audio, eqParams);
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference) => Enhance(audio);

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk) => Enhance(audioChunk);

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio) { /* Not applicable for EQ */ }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultNeuralParametricEQLayers(
            encoderDim: _options.EncoderDim, numEncoderLayers: _options.NumEncoderLayers,
            numBands: _options.NumBands, fftSize: _options.FFTSize,
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
            Name = _useNativeMode ? "NeuralParametricEQ-Native" : "NeuralParametricEQ-ONNX",
            Description = $"Neural Parametric EQ {_options.Variant} (Steinmetz et al., 2022)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.FFTSize / 2 + 1, Complexity = _options.NumEncoderLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["NumBands"] = _options.NumBands.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Variant);
        w.Write(_options.EncoderDim); w.Write(_options.NumEncoderLayers);
        w.Write(_options.NumBands); w.Write(_options.FFTSize);
        w.Write(_options.GainRange); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.EncoderDim = r.ReadInt32(); _options.NumEncoderLayers = r.ReadInt32();
        _options.NumBands = r.ReadInt32(); _options.FFTSize = r.ReadInt32();
        _options.GainRange = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new NeuralParametricEQ<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> ApplyEQ(Tensor<T> audio, Tensor<T> eqParams)
    {
        // Apply predicted EQ parameters to audio (simplified frequency-domain application)
        var result = new Tensor<T>(audio.Shape);
        double strength = EnhancementStrength;
        for (int i = 0; i < audio.Length; i++)
        {
            double sample = NumOps.ToDouble(audio[i]);
            // Apply parametric EQ as gain scaling (simplified)
            double eqGain = eqParams.Length > 0 ? NumOps.ToDouble(eqParams[i % eqParams.Length]) : 0.0;
            double gain = 1.0 + eqGain * strength * 0.1;
            result[i] = NumOps.FromDouble(sample * gain);
        }
        return result;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(NeuralParametricEQ<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}
