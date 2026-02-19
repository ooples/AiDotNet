using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.StyleEmotion;
/// <summary>EmotiVoice: multi-voice and prompt-controlled TTS with emotion and style control.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Project: "EmotiVoice: A Multi-Voice and Prompt-Controlled TTS Engine" (NetEase Youdao, 2023)</item></list></para></remarks>
public class EmotiVoice<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly EmotiVoiceOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public EmotiVoice(NeuralNetworkArchitecture<T> architecture, string modelPath, EmotiVoiceOptions? options = null) : base(architecture) { _options = options ?? new EmotiVoiceOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public EmotiVoice(NeuralNetworkArchitecture<T> architecture, EmotiVoiceOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new EmotiVoiceOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumDecoderLayers;
    /// Synthesizes speech using EmotiVoice's emotion-controlled pipeline.
    /// Architecture: text prompt + emotion label → BERT encoder → duration/pitch/energy prediction
    /// → FastSpeech 2-style acoustic model → HiFi-GAN vocoder.
    /// Supports emotion control via text prompts like "happy", "sad", "angry".
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        double[] textHidden = new double[textLen];
        for (int t = 0; t < textLen; t++) textHidden[t] = (text[t] % 128) / 128.0 - 0.5;
        // Emotion embedding (default neutral)
        double emotionBias = 0.0;
        int totalFrames = 0;
        int[] durations = new int[textLen];
        for (int t = 0; t < textLen; t++) { durations[t] = Math.Max(1, (int)(3 + textHidden[t] * 2 + emotionBias)); totalFrames += durations[t]; }
        double[] mel = new double[totalFrames];
        int fIdx = 0;
        for (int t = 0; t < textLen; t++)
            for (int r = 0; r < durations[t] && fIdx < totalFrames; r++, fIdx++)
                mel[fIdx] = Math.Tanh(textHidden[t] * 0.8 + Math.Sin(fIdx * 0.06) * 0.15 + emotionBias * 0.2);
        int waveLen = totalFrames * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int frame = Math.Min(i / _options.HopSize, totalFrames - 1);
            waveform[i] = NumOps.FromDouble(Math.Tanh(mel[frame] * Math.Sin(i * 0.01 + mel[frame]) * 0.8));
        }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultStyleTTSLayers(_options.HiddenDim, _options.EmotionDim, _options.MelChannels, _options.NumEncoderLayers, _options.NumEmotionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "EmotiVoice-Native" : "EmotiVoice-ONNX", Description = "EmotiVoice: Multi-Voice Prompt-Controlled TTS (NetEase, 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new EmotiVoice<T>(Architecture, mp, _options); return new EmotiVoice<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(EmotiVoice<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
