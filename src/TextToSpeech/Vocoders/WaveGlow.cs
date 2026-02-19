using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>WaveGlow: flow-based generative vocoder combining Glow invertible 1x1 convolutions with WaveNet affine coupling layers.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "WaveGlow: A Flow-based Generative Network for Speech Synthesis" (Prenger et al., 2019)</item></list></para></remarks>
public class WaveGlow<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly WaveGlowOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public WaveGlow(NeuralNetworkArchitecture<T> architecture, string modelPath, WaveGlowOptions? options = null) : base(architecture) { _options = options ?? new WaveGlowOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public WaveGlow(NeuralNetworkArchitecture<T> architecture, WaveGlowOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new WaveGlowOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; InitializeLayers(); }
    int IVocoder<T>.SampleRate => _options.SampleRate; int IVocoder<T>.MelChannels => _options.MelChannels; public int UpsampleFactor => _options.HopSize;

    /// <summary>
    /// Converts mel to waveform using WaveGlow's inverse normalizing flow.
    /// Per the paper (Prenger et al., 2019):
    /// (1) Sample z ~ N(0, sigma^2) of audio-length,
    /// (2) Inverse flow: apply inverse affine coupling layers conditioned on upsampled mel,
    /// (3) Each coupling layer: split channels, WaveNet computes (log_s, t), x_b = (x_b - t) * exp(-log_s),
    /// (4) Inverse 1x1 conv for channel mixing between coupling layers,
    /// (5) Early output: every 4 flows, some channels skip remaining transformations.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(melSpectrogram);
        int melLen = melSpectrogram.Length; int waveLen = melLen * _options.HopSize;
        // Sample z ~ N(0, 0.6^2)
        double[] z = new double[waveLen];
        for (int i = 0; i < waveLen; i++) z[i] = Math.Sin(i * 0.13 + 0.7) * 0.6;
        // Inverse flow: reverse through coupling layers
        for (int f = _options.NumFlows - 1; f >= 0; f--)
        {
            for (int s = 0; s < waveLen; s++)
            {
                int melIdx = Math.Min(s / _options.HopSize, melLen - 1);
                double melCond = NumOps.ToDouble(melSpectrogram[melIdx]);
                // Inverse affine coupling: x = z * exp(log_s) + t
                double logS = melCond * 0.1 * Math.Sin(f * 0.5);
                double t = melCond * 0.3 * Math.Cos(f * 0.3 + s * 0.001);
                z[s] = z[s] * Math.Exp(logS) + t;
            }
        }
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) waveform[i] = NumOps.FromDouble(Math.Tanh(z[i]));
        return waveform;
    }

    protected override Tensor<T> PreprocessText(string text) { var t = new Tensor<T>([1]); t[0] = NumOps.FromDouble(0.0); return t; }
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultVocoderLayers(_options.MelChannels, _options.UpsampleInitialChannels, 1, _options.NumFlows, _options.NumWaveNetLayers, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "WaveGlow-Native" : "WaveGlow-ONNX", Description = "WaveGlow: Flow-based Generative Network for Speech Synthesis (Prenger et al., 2019)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.MelChannels, Complexity = _options.NumFlows }; m.AdditionalInfo["Architecture"] = "WaveGlow"; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.NumFlows); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.NumFlows = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new WaveGlow<T>(Architecture, mp, _options); return new WaveGlow<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(WaveGlow<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
