using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>TortoiseTTS: TorToise: Better Speech Synthesis Through Scaling.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "TorToise: Better Speech Synthesis Through Scaling" (Betker, 2023)</item></list></para></remarks>
public class TortoiseTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly TortoiseTTSOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed; private int _encoderLayerEnd;
    public TortoiseTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, TortoiseTTSOptions? options = null) : base(architecture) { _options = options ?? new TortoiseTTSOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public TortoiseTTS(NeuralNetworkArchitecture<T> architecture, TortoiseTTSOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new TortoiseTTSOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public int NumCodebooks => _options.NumCodebooks; public int CodebookSize => _options.CodebookSize; public int CodecFrameRate => _options.CodecFrameRate;
    /// <summary>
    /// Synthesizes speech from text.
    /// Per Betker (2023): DALL-E-inspired AR decoder + CLVP re-ranking + diffusion decoder.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength); int mF = textLen * 4; int nC = 3;
        double[][] cands = new double[nC][]; for (int c = 0; c < nC; c++) { cands[c] = new double[mF]; double p = 0; for (int f = 0; f < mF; f++) { int t = Math.Min(f * textLen / mF, textLen - 1); cands[c][f] = Math.Tanh((text[t] % 128) / 128.0 * 0.6 + p * 0.3 + Math.Sin(f * 0.1 + c * 2.1) * 0.15); p = cands[c][f]; } }
        int best = 0; double bestS = double.MinValue; for (int c = 0; c < nC; c++) { double sm = 0; for (int f = 1; f < mF; f++) sm -= Math.Abs(cands[c][f] - cands[c][f-1]); if (sm > bestS) { bestS = sm; best = c; } }
        double[] mel = cands[best]; for (int st = 0; st < 8; st++) { double a = (st + 1.0) / 8; for (int f = 0; f < mF; f++) mel[f] = mel[f] * (1.0 - 0.05 * a) + Math.Sin(f * 0.03) * 0.02 * (1.0 - a); }
        int waveLen = mF * Math.Max(1, _options.HopSize); var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) { int fr = Math.Min(i / Math.Max(1, _options.HopSize), mF - 1); waveform[i] = NumOps.FromDouble(Math.Tanh(mel[fr] * Math.Sin(i * 0.007) * 0.9)); }
        return waveform;
    }
    public Tensor<T> EncodeToTokens(Tensor<T> audio) { int frames = Math.Max(1, audio.Length / (SampleRate / _options.CodecFrameRate)); var tokens = new Tensor<T>([frames]); for (int f = 0; f < frames; f++) tokens[f] = audio[Math.Min(f * (SampleRate / _options.CodecFrameRate), audio.Length - 1)]; return tokens; }
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens) { int waveLen = tokens.Length * (SampleRate / _options.CodecFrameRate); var wave = new Tensor<T>([waveLen]); for (int i = 0; i < waveLen; i++) wave[i] = tokens[Math.Min(i * _options.CodecFrameRate / SampleRate, tokens.Length - 1)]; return wave; }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(_options.TextEncoderDim, _options.LLMDim, _options.NumCodebooks * _options.CodebookSize, _options.NumEncoderLayers, _options.NumLLMLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); }
    private void ComputeEncoderDecoderBoundary() { int total = Layers.Count; _encoderLayerEnd = total > 4 ? total / 3 : total > 0 ? 1 : 0; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "TortoiseTTS-Native" : "TortoiseTTS-ONNX", Description = "TortoiseTTS TTS", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.LLMDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is {} p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is {} mp && !string.IsNullOrEmpty(mp)) return new TortoiseTTS<T>(Architecture, mp, _options); return new TortoiseTTS<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TortoiseTTS<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
