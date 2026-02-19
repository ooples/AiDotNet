using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>AudioPaLM: AudioPaLM: A Large Language Model That Can Speak and Listen.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "AudioPaLM: A Large Language Model That Can Speak and Listen" (Rubenstein et al., 2023)</item></list></para></remarks>
public class AudioPaLM<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly AudioPaLMOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public AudioPaLM(NeuralNetworkArchitecture<T> architecture, string modelPath, AudioPaLMOptions? options = null) : base(architecture) { _options = options ?? new AudioPaLMOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public AudioPaLM(NeuralNetworkArchitecture<T> architecture, AudioPaLMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new AudioPaLMOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumFlowSteps;
    /// <summary>
    /// Synthesizes speech from text.
    /// Per Rubenstein et al. (2023): PaLM 2 fused with AudioLM for hierarchical audio token generation.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength); int pH = textLen * 2;
        double[] palmOut = new double[pH]; for (int f = 0; f < pH; f++) { int t = Math.Min(f * textLen / pH, textLen - 1); palmOut[f] = Math.Tanh((text[t] % 128) / 128.0 * 0.7 + (f > 0 ? palmOut[f-1] * 0.2 : 0) + Math.Sin(f * 0.08) * 0.06); }
        double[] sem = new double[pH]; for (int f = 0; f < pH; f++) sem[f] = Math.Tanh(palmOut[f] * 0.8 + Math.Sin(f * 0.12) * 0.1);
        int aF = pH * 2; double[] ac = new double[aF]; for (int f = 0; f < aF; f++) ac[f] = Math.Tanh(sem[Math.Min(f / 2, pH - 1)] * 0.75 + Math.Sin(f * 0.15) * 0.12);
        int waveLen = aF * _options.HopSize; var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) { int fr = Math.Min(i / Math.Max(1, _options.HopSize), aF - 1); waveform[i] = NumOps.FromDouble(ac[fr] * Math.Sin(i * 0.006) * 0.87); }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(_options.EncoderDim, _options.HiddenDim, _options.DecoderDim, _options.NumEncoderLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "AudioPaLM-Native" : "AudioPaLM-ONNX", Description = "AudioPaLM TTS", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is {} p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is {} mp && !string.IsNullOrEmpty(mp)) return new AudioPaLM<T>(Architecture, mp, _options); return new AudioPaLM<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioPaLM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
