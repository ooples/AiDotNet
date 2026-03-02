using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// HiFi-GAN: high-fidelity neural vocoder with multi-receptive field fusion for parallel waveform generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (Kong et al., 2020)</item></list></para>
/// <para><b>For Beginners:</b> /// HiFi-GAN: high-fidelity neural vocoder with multi-receptive field fusion for parallel waveform generation.
///. This model converts text input into speech audio output.</para>
/// </remarks>
public class HiFiGAN<T> : TtsModelBase<T>, IVocoder<T>
{
    private readonly HiFiGANOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode; private bool _disposed;

    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, HiFiGANOptions? options = null) : base(architecture) { _options = options ?? new HiFiGANOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public HiFiGAN(NeuralNetworkArchitecture<T> architecture, HiFiGANOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new HiFiGANOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; InitializeLayers(); }

    int IVocoder<T>.SampleRate => _options.SampleRate; int IVocoder<T>.MelChannels => _options.MelChannels; public int UpsampleFactor { get { int f = 1; foreach (var r in _options.UpsampleRates) f *= r; return f; } }

    /// <summary>
    /// Converts mel-spectrogram to waveform using HiFi-GAN's generator.
    /// Per the paper (Kong et al., 2020):
    /// (1) Initial conv: mel channels -> upsample_initial_channels,
    /// (2) Transposed conv upsampling blocks: each increases temporal resolution by upsample_rate,
    /// (3) Multi-Receptive Field Fusion (MRF): parallel residual blocks with different kernel sizes (3,7,11),
    /// (4) Final conv: channels -> 1 with tanh activation for [-1,1] waveform output.
    /// Discriminators (MPD + MSD) used only during training.
    /// </summary>
    public Tensor<T> MelToWaveform(Tensor<T> melSpectrogram) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(melSpectrogram); return Predict(melSpectrogram); }

    protected override Tensor<T> PreprocessText(string text) { var t = new Tensor<T>([1]); t[0] = NumOps.FromDouble(0.0); return t; }
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultVocoderLayers(_options.MelChannels, _options.UpsampleInitialChannels, 1, _options.UpsampleRates.Length, _options.ResblockKernelSizes.Length, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); try { var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); } finally { SetTrainingMode(false); } }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "HiFiGAN-Native" : "HiFiGAN-ONNX", Description = "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., 2020)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.MelChannels, Complexity = _options.UpsampleRates.Length + _options.ResblockKernelSizes.Length }; m.AdditionalInfo["Architecture"] = "HiFiGAN"; m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.UpsampleInitialChannels); writer.Write(_options.UpsampleRates.Length); foreach (var r in _options.UpsampleRates) writer.Write(r); writer.Write(_options.DropoutRate); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.UpsampleInitialChannels = reader.ReadInt32(); int n = reader.ReadInt32(); _options.UpsampleRates = new int[n]; for (int i = 0; i < n; i++) _options.UpsampleRates[i] = reader.ReadInt32();  _options.DropoutRate = reader.ReadDouble();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new HiFiGAN<T>(Architecture, mp, _options); return new HiFiGAN<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(HiFiGAN<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
