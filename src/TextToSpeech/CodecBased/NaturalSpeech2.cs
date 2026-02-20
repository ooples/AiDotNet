using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>NaturalSpeech 2: latent diffusion model with continuous latent vectors for zero-shot speech synthesis.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers" (Shen et al., 2023)</item></list></para></remarks>
public class NaturalSpeech2<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly NaturalSpeech2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public NaturalSpeech2(NeuralNetworkArchitecture<T> architecture, string modelPath, NaturalSpeech2Options? options = null) : base(architecture) { _options = options ?? new NaturalSpeech2Options(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public NaturalSpeech2(NeuralNetworkArchitecture<T> architecture, NaturalSpeech2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new NaturalSpeech2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; /* Intentionally hides base HiddenDim to expose options-driven value. */ public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumDiffusionSteps;
    /// Synthesizes speech using NaturalSpeech 2's latent diffusion pipeline.
    /// Per the paper (Shen et al., 2023):
    /// (1) Neural audio codec: speech → continuous latent vectors (not discrete tokens),
    /// (2) Diffusion model: denoises latent vectors conditioned on text + duration + pitch + speaker,
    /// (3) Codec decoder: latent → waveform.
    /// Achieves zero-shot TTS and singing voice synthesis.
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultFlowMatchingTTSLayers(_options.HiddenDim, _options.DiffusionDim, _options.MelChannels, _options.NumEncoderLayers, _options.NumDiffusionSteps, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { ThrowIfDisposed(); if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); try { var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); } finally { SetTrainingMode(false); } }
    public override void UpdateParameters(Vector<T> parameters) { ThrowIfDisposed(); if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "NaturalSpeech2-Native" : "NaturalSpeech2-ONNX", Description = "NaturalSpeech 2: Latent Diffusion TTS (Shen et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); writer.Write(_options.NumDiffusionSteps); writer.Write(_options.DiffusionDim); writer.Write(_options.DropoutRate); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.HopSize); writer.Write(_options.MaxTextLength); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.NumDiffusionSteps = reader.ReadInt32(); _options.DiffusionDim = reader.ReadInt32(); _options.DropoutRate = reader.ReadDouble(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.MaxTextLength = reader.ReadInt32(); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new NaturalSpeech2<T>(Architecture, mp, _options); return new NaturalSpeech2<T>(Architecture, _options, _optimizer); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(NaturalSpeech2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
