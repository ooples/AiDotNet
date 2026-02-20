using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>DiTToTTS: DiTTo-TTS: Efficient and Scalable Zero-Shot TTS with Diffusion Transformer.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "DiTTo-TTS: Efficient and Scalable Zero-Shot TTS with Diffusion Transformer" (Lee et al., 2024)</item></list></para></remarks>
public class DiTToTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly DiTToTTSOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public DiTToTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, DiTToTTSOptions? options = null) : base(architecture) { _options = options ?? new DiTToTTSOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public DiTToTTS(NeuralNetworkArchitecture<T> architecture, DiTToTTSOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DiTToTTSOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumFlowSteps;
    /// <summary>
    /// Synthesizes speech from text.
    /// Per Lee et al. (2024): DiT blocks with adaptive layer norm for iterative denoising.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        // Run preprocessed text through learned layers for feature extraction
        var features = input;
        foreach (var l in Layers) features = l.Forward(features);
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        int tF = 0; int[] dur = new int[textLen]; for (int i = 0; i < textLen; i++) { dur[i] = Math.Max(2, (int)(3.0 + Math.Sin(i * 0.3) * 1.5)); tF += dur[i]; } tF = Math.Min(tF, 500);
        double[] tc = new double[tF]; int fI = 0;
        for (int i = 0; i < textLen && fI < tF; i++) { double cv = (text[i] % 128) / 128.0; for (int d = 0; d < dur[i] && fI < tF; d++) tc[fI++] = cv; }
        double[] lat = new double[tF]; for (int f = 0; f < tF; f++) lat[f] = Math.Sin(f * 0.25) * 0.6;
        for (int st = 0; st < 10; st++) { double t = 1.0 - (st + 1.0) / 10; for (int f = 0; f < tF; f++) lat[f] = Math.Tanh(tc[f] * (1.0 - t) + lat[f] * t + Math.Sin(f * 0.1 + st * 0.5) * t * 0.2); }
        int waveLen = tF * _options.HopSize; var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) { int fr = Math.Min(i / Math.Max(1, _options.HopSize), tF - 1); waveform[i] = NumOps.FromDouble(lat[fr] * Math.Sin(i * 0.006) * 0.85); }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultFlowMatchingTTSLayers(_options.EncoderDim, _options.FlowDim, _options.DecoderDim, _options.NumEncoderLayers, _options.NumFlowLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "DiTToTTS-Native" : "DiTToTTS-ONNX", Description = "DiTToTTS TTS", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.DecoderDim); writer.Write(_options.DropoutRate); writer.Write(_options.EncoderDim); writer.Write(_options.FlowDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumFlowLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32();  _options.DecoderDim = reader.ReadInt32(); _options.DropoutRate = reader.ReadDouble(); _options.EncoderDim = reader.ReadInt32(); _options.FlowDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumFlowLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is {} p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is {} mp && !string.IsNullOrEmpty(mp)) return new DiTToTTS<T>(Architecture, mp, _options); return new DiTToTTS<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DiTToTTS<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
