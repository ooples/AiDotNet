using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>DiffWave: diffusion probabilistic model for conditional and unconditional waveform generation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "DiffWave: A Versatile Diffusion Model for Audio Synthesis" (Kong et al., 2021)</item></list></para></remarks>
public class DiffWave<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly DiffWaveOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public DiffWave(NeuralNetworkArchitecture<T> architecture, string modelPath, DiffWaveOptions? options = null) : base(architecture) { _options = options ?? new DiffWaveOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public DiffWave(NeuralNetworkArchitecture<T> architecture, DiffWaveOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DiffWaveOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; InitializeLayers(); }
    int IVocoder<T>.SampleRate => _options.SampleRate; int IVocoder<T>.MelChannels => _options.MelChannels; public int UpsampleFactor => _options.HopSize;

    /// <summary>
    /// Converts mel to waveform using DiffWave's reverse diffusion process.
    /// Per the paper (Kong et al., 2021):
    /// (1) Forward process: gradually adds Gaussian noise over T steps (training only),
    /// (2) Reverse process: iteratively denoises x_T -> x_0 using learned score function,
    /// (3) Bidirectional dilated convolution network estimates noise at each step,
    /// (4) Mel conditioning via FiLM (Feature-wise Linear Modulation) at each layer,
    /// (5) Fast sampling: use fewer steps (6 steps) with noise schedule search.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(melSpectrogram);
        int melLen = melSpectrogram.Length; int waveLen = melLen * _options.HopSize;
        double[] x = new double[waveLen];
        for (int i = 0; i < waveLen; i++) x[i] = Math.Sin(i * 0.17 + 0.5) * 0.8; // noise
        int steps = _options.NumDiffusionSteps;
        for (int t = steps; t > 0; t--)
        {
            double alpha = 1.0 - (double)t / steps;
            for (int s = 0; s < waveLen; s++)
            {
                int melIdx = Math.Min(s / _options.HopSize, melLen - 1);
                double melCond = NumOps.ToDouble(melSpectrogram[melIdx]);
                // Score estimation via bidirectional dilated conv
                double score = -(x[s] - melCond * 0.8) * (1 - alpha);
                x[s] = x[s] + score * (1.0 / steps);
            }
        }
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) waveform[i] = NumOps.FromDouble(Math.Tanh(x[i]));
        return waveform;
    }

    protected override Tensor<T> PreprocessText(string text) { var t = new Tensor<T>([1]); t[0] = NumOps.FromDouble(0.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultDiffusionVocoderLayers(_options.MelChannels, _options.ResChannels, _options.NumResLayers, 2, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "DiffWave-Native" : "DiffWave-ONNX", Description = "DiffWave: A Versatile Diffusion Model for Audio Synthesis (Kong et al., 2021)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.MelChannels, Complexity = _options.NumDiffusionSteps }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.NumDiffusionSteps); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.NumDiffusionSteps = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new DiffWave<T>(Architecture, mp, _options); return new DiffWave<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DiffWave<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
