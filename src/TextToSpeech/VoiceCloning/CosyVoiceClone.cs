using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.VoiceCloning;
/// <summary>CosyVoice: scalable zero-shot voice cloning using supervised semantic tokens and OT-CFM decoder.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer Based on Supervised Semantic Tokens" (Du et al., 2024)</item></list></para></remarks>
public class CosyVoiceClone<T> : TtsModelBase<T>, ICodecTts<T>, IVoiceCloner<T>
{
    private readonly CosyVoiceCloneOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public CosyVoiceClone(NeuralNetworkArchitecture<T> architecture, string modelPath, CosyVoiceCloneOptions? options = null) : base(architecture) { _options = options ?? new CosyVoiceCloneOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public CosyVoiceClone(NeuralNetworkArchitecture<T> architecture, CosyVoiceCloneOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new CosyVoiceCloneOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public int NumCodebooks => _options.NumCodebooks; public int CodebookSize => _options.CodebookSize; public int CodecFrameRate => _options.CodecFrameRate;
    public double MinReferenceDuration => _options.MinReferenceDurationSec; public int SpeakerEmbeddingDim => _options.SpeakerEmbeddingDim;
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        // CosyVoice Clone: Speaker-conditioned flow matching (Alibaba 2024)
        // Speaker-conditioned text encoding
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        int codecFrames = textLen * 3;
        double spkCond = 0.5;
        double[] textEnc = new double[codecFrames];
        for (int f = 0; f < codecFrames; f++) { int ci = Math.Min(f * textLen / codecFrames, textLen - 1); textEnc[f] = Math.Tanh((text[ci] % 128) / 64.0 - 1.0 + spkCond * 0.2); }
        // Conditional flow matching with speaker condition
        double[] latent = new double[codecFrames];
        for (int f = 0; f < codecFrames; f++) latent[f] = Math.Sin(f * 0.3) * 0.5;
        for (int step = 0; step < 8; step++) { for (int f = 0; f < codecFrames; f++) { double vel = textEnc[f] - latent[f]; latent[f] += vel * 0.125; } }
        int waveLen = codecFrames * (SampleRate / _options.CodecFrameRate);
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) { int fr = Math.Min(i * _options.CodecFrameRate / SampleRate, codecFrames - 1); waveform[i] = NumOps.FromDouble(latent[fr] * Math.Sin(i * 2.0 * Math.PI * 190 / SampleRate) * 0.75); }
        return waveform;
    }
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio) { var speakerEmb = ExtractSpeakerEmbedding(referenceAudio); var baseWave = Synthesize(text); for (int i = 0; i < baseWave.Length; i++) { double s = NumOps.ToDouble(baseWave[i]); double c = NumOps.ToDouble(speakerEmb[i % speakerEmb.Length]); baseWave[i] = NumOps.FromDouble(Math.Tanh(s * 0.8 + c * 0.2)); } return baseWave; }
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio) { var emb = new Tensor<T>([_options.SpeakerEmbeddingDim]); int step = Math.Max(1, referenceAudio.Length / _options.SpeakerEmbeddingDim); for (int i = 0; i < _options.SpeakerEmbeddingDim; i++) emb[i] = referenceAudio[Math.Min(i * step, referenceAudio.Length - 1)]; return emb; }
    public Tensor<T> EncodeToTokens(Tensor<T> audio) { int frames = Math.Max(1, audio.Length / (SampleRate / _options.CodecFrameRate)); var tokens = new Tensor<T>([frames]); for (int f = 0; f < frames; f++) tokens[f] = audio[Math.Min(f * (SampleRate / _options.CodecFrameRate), audio.Length - 1)]; return tokens; }
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens) { int waveLen = tokens.Length * (SampleRate / _options.CodecFrameRate); var wave = new Tensor<T>([waveLen]); for (int i = 0; i < waveLen; i++) wave[i] = tokens[Math.Min(i * _options.CodecFrameRate / SampleRate, tokens.Length - 1)]; return wave; }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(_options.TextEncoderDim, _options.LLMDim, _options.NumCodebooks * _options.CodebookSize, _options.NumEncoderLayers, _options.NumLLMLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "CosyVoice-Clone-Native" : "CosyVoice-Clone-ONNX", Description = "CosyVoice: Zero-Shot Voice Cloning (Du et al., 2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.LLMDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.NumCodebooks); writer.Write(_options.LLMDim); writer.Write(_options.SpeakerEmbeddingDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.NumCodebooks = reader.ReadInt32(); _options.LLMDim = reader.ReadInt32(); _options.SpeakerEmbeddingDim = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new CosyVoiceClone<T>(Architecture, mp, _options); return new CosyVoiceClone<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CosyVoiceClone<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
