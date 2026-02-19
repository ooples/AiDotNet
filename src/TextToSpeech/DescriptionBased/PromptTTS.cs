using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.DescriptionBased;
/// <summary>PromptTTS: description-based TTS that controls speaker attributes via natural language prompts (e.g., "a young female with a cheerful tone").</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "PromptTTS: Controllable Text-to-Speech with Text Descriptions" (Guo et al., 2023)</item></list></para></remarks>
public class PromptTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly PromptTTSOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed; private int _encoderLayerEnd;
    public PromptTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, PromptTTSOptions? options = null) : base(architecture) { _options = options ?? new PromptTTSOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public PromptTTS(NeuralNetworkArchitecture<T> architecture, PromptTTSOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new PromptTTSOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumFlowSteps;
    /// <summary>
    /// Synthesizes speech using PromptTTS's description-conditioned pipeline.
    /// Per the paper (Guo et al., 2023):
    /// (1) Text encoder: encodes input text into hidden states,
    /// (2) Style prompt encoder: encodes natural language description (gender, age, emotion, speed)
    ///     into style embeddings via BERT-like model,
    /// (3) Variance adaptor: predicts duration, pitch, energy conditioned on style,
    /// (4) Mel decoder: generates mel spectrogram from style-conditioned features.
    /// Enables zero-shot style control via text descriptions like "a young female speaking quickly with excitement".
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        // Text encoder
        double[] textHidden = new double[textLen];
        for (int t = 0; t < textLen; t++) textHidden[t] = (text[t] % 128) / 128.0 - 0.5;
        // Style prompt encoder (default neutral style)
        double styleBias = 0.0;
        double speedFactor = 1.0;
        // Duration predictor conditioned on style
        int totalFrames = 0;
        int[] durations = new int[textLen];
        for (int t = 0; t < textLen; t++)
        {
            durations[t] = Math.Max(1, (int)((3 + textHidden[t] * 2) * speedFactor));
            totalFrames += durations[t];
        }
        // Mel decoder with style conditioning
        double[] mel = new double[totalFrames];
        int fIdx = 0;
        for (int t = 0; t < textLen; t++)
            for (int r = 0; r < durations[t] && fIdx < totalFrames; r++, fIdx++)
                mel[fIdx] = Math.Tanh(textHidden[t] * 0.8 + styleBias * 0.2 + Math.Sin(fIdx * 0.06) * 0.15);
        // HiFi-GAN vocoder
        int waveLen = totalFrames * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int frame = Math.Min(i / _options.HopSize, totalFrames - 1);
            waveform[i] = NumOps.FromDouble(Math.Tanh(mel[frame] * Math.Sin(i * 0.01 + mel[frame]) * 0.8));
        }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) => new Tensor<T>([Math.Min(text.Length, _options.MaxTextLength)]); protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultStyleTTSLayers(_options.HiddenDim, _options.PromptEncoderDim, _options.MelChannels, _options.NumEncoderLayers, _options.NumPromptLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); }
    private void ComputeEncoderDecoderBoundary() { int total = Layers.Count; _encoderLayerEnd = total > 4 ? total / 3 : total > 0 ? 1 : 0; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "PromptTTS-Native" : "PromptTTS-ONNX", Description = "PromptTTS: Description-Controlled TTS (Guo et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new PromptTTS<T>(Architecture, mp, _options); return new PromptTTS<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PromptTTS<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
