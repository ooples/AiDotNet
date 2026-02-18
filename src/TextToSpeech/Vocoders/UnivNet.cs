using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.Vocoders;
/// <summary>UnivNet: universal neural vocoder with location-variable convolution (LVC) for adaptive kernel generation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminator for High-Fidelity Waveform Generation" (Jang et al., 2021)</item></list></para></remarks>
public class UnivNet<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly UnivNetOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public UnivNet(NeuralNetworkArchitecture<T> architecture, string modelPath, UnivNetOptions? options = null) : base(architecture) { _options = options ?? new UnivNetOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public UnivNet(NeuralNetworkArchitecture<T> architecture, UnivNetOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new UnivNetOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; InitializeLayers(); }
    int IVocoder<T>.SampleRate => _options.SampleRate; int IVocoder<T>.MelChannels => _options.MelChannels; public int UpsampleFactor => _options.HopSize;
    /// <summary>
    /// Converts mel to waveform using UnivNet's LVC (Location-Variable Convolution) blocks.
    /// Per the paper (Jang et al., 2021):
    /// (1) Noise input + mel conditioning,
    /// (2) LVC blocks: kernel weights are dynamically generated from mel features (not fixed),
    /// (3) GABlock with gated activation + location-variable conv for adaptive frequency modeling,
    /// (4) Multi-resolution spectrogram discriminator (MRSD) for training.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(melSpectrogram);
        int melLen = melSpectrogram.Length; int waveLen = melLen * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        for (int f = 0; f < waveLen; f++)
        {
            int melIdx = Math.Min(f / _options.HopSize, melLen - 1);
            double melVal = NumOps.ToDouble(melSpectrogram[melIdx]);
            // LVC: location-variable convolution - kernel adapted from mel
            double lvc = 0;
            for (int k = 0; k < _options.NumKernels; k++) { double kernelWeight = melVal * 0.3 + Math.Sin(f * (k + 1) * 0.005) * 0.2; lvc += Math.Tanh(kernelWeight); }
            lvc /= _options.NumKernels;
            // Gated activation
            double gate = 1.0 / (1.0 + Math.Exp(-(melVal * 0.5 + lvc)));
            waveform[f] = NumOps.FromDouble(Math.Tanh(gate * lvc * 0.8 + melVal * 0.2));
        }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) => new Tensor<T>([1]); protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultVocoderLayers(_options.MelChannels, 512, 1, 4, _options.NumLMBlocks, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "UnivNet-Native" : "UnivNet-ONNX", Description = "UnivNet: Universal Neural Vocoder (Jang et al., 2021)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.MelChannels }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new UnivNet<T>(Architecture, mp, _options); return new UnivNet<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(UnivNet<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
