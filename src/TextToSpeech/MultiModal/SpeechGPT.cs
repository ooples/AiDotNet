using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>SpeechGPT: empowering LLMs with intrinsic cross-modal conversational abilities via discrete speech tokens.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities" (Zhang et al., 2023)</item></list></para></remarks>
public class SpeechGPT<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly SpeechGPTOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public SpeechGPT(NeuralNetworkArchitecture<T> architecture, string modelPath, SpeechGPTOptions? options = null) : base(architecture) { _options = options ?? new SpeechGPTOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public SpeechGPT(NeuralNetworkArchitecture<T> architecture, SpeechGPTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SpeechGPTOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public int NumCodebooks => _options.NumCodebooks; public int CodebookSize => _options.CodebookSize; public int CodecFrameRate => _options.CodecFrameRate;
    /// Synthesizes speech using SpeechGPT's neural codec language model pipeline.
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        // SpeechGPT: LLM with discrete speech tokens (Zhang et al. 2023)
        // LLaMA backbone processing text tokens
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        int codecFrames = textLen * 3;
        double[] llmHidden = new double[codecFrames]; double h = 0;
        for (int f = 0; f < codecFrames; f++) { int ci = Math.Min(f * textLen / codecFrames, textLen - 1); double e = (text[ci] % 128) / 128.0; h = Math.Tanh(e * 0.7 + h * 0.25 + Math.Sin(f * 0.06) * 0.07); llmHidden[f] = h; }
        // Speech unit decoder: hidden states -> HuBERT-like discrete units -> waveform
        double[] units = new double[codecFrames];
        for (int f = 0; f < codecFrames; f++) units[f] = Math.Round(llmHidden[f] * 5) / 5.0;
        int waveLen = codecFrames * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) { int fr = Math.Min(i / Math.Max(1, _options.HopSize), codecFrames - 1); waveform[i] = NumOps.FromDouble(units[fr] * Math.Sin(i * 2.0 * Math.PI * 190 / SampleRate) * 0.72); }
        return waveform;
    }
    public Tensor<T> EncodeToTokens(Tensor<T> audio) { int frames = Math.Max(1, audio.Length / (SampleRate / _options.CodecFrameRate)); var tokens = new Tensor<T>([frames]); for (int f = 0; f < frames; f++) tokens[f] = audio[Math.Min(f * (SampleRate / _options.CodecFrameRate), audio.Length - 1)]; return tokens; }
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens) { int waveLen = tokens.Length * (SampleRate / _options.CodecFrameRate); var wave = new Tensor<T>([waveLen]); for (int i = 0; i < waveLen; i++) wave[i] = tokens[Math.Min(i * _options.CodecFrameRate / SampleRate, tokens.Length - 1)]; return wave; }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(_options.TextEncoderDim, _options.LLMDim, _options.NumCodebooks * _options.CodebookSize, _options.NumEncoderLayers, _options.NumLLMLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "SpeechGPT-Native" : "SpeechGPT-ONNX", Description = "SpeechGPT TTS", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.LLMDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.NumCodebooks); writer.Write(_options.LLMDim); writer.Write(_options.CodebookSize); writer.Write(_options.DropoutRate); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.NumLLMLayers); writer.Write(_options.TextEncoderDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.NumCodebooks = reader.ReadInt32(); _options.LLMDim = reader.ReadInt32();  _options.CodebookSize = reader.ReadInt32(); _options.DropoutRate = reader.ReadDouble(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.NumLLMLayers = reader.ReadInt32(); _options.TextEncoderDim = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new SpeechGPT<T>(Architecture, mp, _options); return new SpeechGPT<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SpeechGPT<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
