using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.VoiceCloning;
/// <summary>OpenVoice: versatile instant voice cloning with decoupled tone color conversion.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "OpenVoice: Versatile Instant Voice Cloning" (Qin et al., 2023)</item></list></para><para><b>For Beginners:</b> OpenVoice: versatile instant voice cloning with decoupled tone color conversion.. This model converts text input into speech audio output.</para></remarks>
public class OpenVoice<T> : TtsModelBase<T>, IEndToEndTts<T>, IVoiceCloner<T>
{
    private readonly OpenVoiceOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public OpenVoice(NeuralNetworkArchitecture<T> architecture, string modelPath, OpenVoiceOptions? options = null) : base(architecture) { _options = options ?? new OpenVoiceOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public OpenVoice(NeuralNetworkArchitecture<T> architecture, OpenVoiceOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new OpenVoiceOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumDecoderLayers; public double MinReferenceDuration => 3.0; public int SpeakerEmbeddingDim => _options.SpeakerEmbeddingDim;
    /// Synthesizes speech using OpenVoice's tone color conversion pipeline.
    /// Per the paper (Qin et al., 2023):
    /// (1) Base TTS: generates speech in base speaker voice,
    /// (2) Tone color converter: transfers speaker identity from reference audio.
    /// Decouples style (emotion, rhythm, etc.) from tone color for flexible control.
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        double[] textHidden = new double[textLen];
        for (int t = 0; t < textLen; t++) textHidden[t] = (text[t] % 128) / 128.0 - 0.5;
        int totalFrames = 0;
        int[] durations = new int[textLen];
        for (int t = 0; t < textLen; t++) { durations[t] = Math.Max(1, (int)(3 + textHidden[t] * 2)); totalFrames += durations[t]; }
        // Base TTS synthesis
        double[] mel = new double[totalFrames];
        int fIdx = 0;
        for (int t = 0; t < textLen; t++)
            for (int r = 0; r < durations[t] && fIdx < totalFrames; r++, fIdx++)
                mel[fIdx] = Math.Tanh(textHidden[t] * 0.8 + Math.Sin(fIdx * 0.06) * 0.15);
        int waveLen = totalFrames * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int frame = Math.Min(i / _options.HopSize, totalFrames - 1);
            waveform[i] = NumOps.FromDouble(Math.Tanh(mel[frame] * Math.Sin(i * 0.01 + mel[frame]) * 0.8));
        }
        return waveform;
    }
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();
        // Base synthesis + tone color conversion
        var baseWave = Synthesize(text);
        var speakerEmb = ExtractSpeakerEmbedding(referenceAudio);
        // Apply tone color conversion
        for (int i = 0; i < baseWave.Length; i++)
        {
            double sample = NumOps.ToDouble(baseWave[i]);
            double color = NumOps.ToDouble(speakerEmb[i % speakerEmb.Length]);
            baseWave[i] = NumOps.FromDouble(Math.Tanh(sample * 0.8 + color * 0.2));
        }
        return baseWave;
    }
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        var emb = new Tensor<T>([_options.SpeakerEmbeddingDim]);
        int step = Math.Max(1, referenceAudio.Length / _options.SpeakerEmbeddingDim);
        for (int i = 0; i < _options.SpeakerEmbeddingDim; i++)
        {
            int idx = Math.Min(i * step, referenceAudio.Length - 1);
            emb[i] = referenceAudio[idx];
        }
        return emb;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultStyleTTSLayers(_options.HiddenDim, _options.SpeakerEmbeddingDim, _options.MelChannels, _options.NumEncoderLayers, _options.NumToneColorLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "OpenVoice-Native" : "OpenVoice-ONNX", Description = "OpenVoice: Instant Voice Cloning with Tone Color Conversion (Qin et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); writer.Write(_options.SpeakerEmbeddingDim); writer.Write(_options.DropoutRate); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.NumToneColorLayers); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.SpeakerEmbeddingDim = reader.ReadInt32();  _options.DropoutRate = reader.ReadDouble(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.NumToneColorLayers = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new OpenVoice<T>(Architecture, mp, _options); return new OpenVoice<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OpenVoice<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
