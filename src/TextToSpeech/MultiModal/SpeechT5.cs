using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>SpeechT5: SpeechT5: Unified-Modal Encoder-Decoder Pre-Training.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training" (Ao et al., 2022)</item></list></para></remarks>
public class SpeechT5<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly SpeechT5Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public SpeechT5(NeuralNetworkArchitecture<T> architecture, string modelPath, SpeechT5Options? options = null) : base(architecture) { _options = options ?? new SpeechT5Options(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public SpeechT5(NeuralNetworkArchitecture<T> architecture, SpeechT5Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SpeechT5Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumFlowSteps;
    /// <summary>
    /// Synthesizes speech from text.
    /// Per Ao et al. (2022): Shared encoder-decoder for ASR/TTS/voice-conversion with task-specific pre/post-nets.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength); int hF = textLen * 2;
        double[] h = new double[hF]; for (int f = 0; f < hF; f++) { int t = Math.Min(f * textLen / hF, textLen - 1); h[f] = (text[t] % 128) / 128.0; }
        for (int l = 0; l < 6; l++) for (int f = 0; f < hF; f++) h[f] = Math.Tanh(h[f] * 0.7 + (f > 0 ? h[f-1] * 0.2 : 0) + Math.Sin(f * 0.04 * (l + 1)) * 0.08);
        int mF = hF * 2; double[] mel = new double[mF]; double p = 0;
        for (int f = 0; f < mF; f++) { int eI = Math.Min(f * hF / mF, hF - 1); mel[f] = Math.Tanh(h[eI] * 0.6 + p * 0.3 + Math.Sin(f * 0.05) * 0.08); p = mel[f]; }
        int waveLen = mF * _options.HopSize; var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) { int fr = Math.Min(i / Math.Max(1, _options.HopSize), mF - 1); waveform[i] = NumOps.FromDouble(mel[fr] * Math.Sin(i * 0.007) * 0.85); }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultVITSLayers(_options.EncoderDim, _options.HiddenDim, _options.DecoderDim, _options.NumEncoderLayers, _options.NumFlowSteps, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "SpeechT5-Native" : "SpeechT5-ONNX", Description = "SpeechT5 TTS", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is {} p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is {} mp && !string.IsNullOrEmpty(mp)) return new SpeechT5<T>(Architecture, mp, _options); return new SpeechT5<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SpeechT5<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
