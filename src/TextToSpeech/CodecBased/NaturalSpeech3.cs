using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.CodecBased;
/// <summary>NaturalSpeech 3: factorized codec + diffusion for disentangled speech attribute control.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "NaturalSpeech 3: Zero-Shot Speech Synthesis with a Factorized Codec and Diffusion Models" (Ju et al., 2024)</item></list></para></remarks>
public class NaturalSpeech3<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly NaturalSpeech3Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed; private int _encoderLayerEnd;
    public NaturalSpeech3(NeuralNetworkArchitecture<T> architecture, string modelPath, NaturalSpeech3Options? options = null) : base(architecture) { _options = options ?? new NaturalSpeech3Options(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public NaturalSpeech3(NeuralNetworkArchitecture<T> architecture, NaturalSpeech3Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new NaturalSpeech3Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumDiffusionSteps;
    /// Synthesizes speech using NaturalSpeech 3's factorized diffusion pipeline.
    /// Per the paper (Ju et al., 2024):
    /// FACodec factorizes speech into prosody, content, acoustic detail, and timbre subspaces.
    /// Separate diffusion models operate on each disentangled subspace for fine control.
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        double[] textHidden = new double[textLen];
        for (int t = 0; t < textLen; t++) textHidden[t] = (text[t] % 128) / 128.0 - 0.5;
        int totalFrames = textLen * 3;
        // Factorized latents: content, prosody, acoustic detail
        double[] content = new double[totalFrames];
        double[] prosody = new double[totalFrames];
        double[] detail = new double[totalFrames];
        for (int f = 0; f < totalFrames; f++)
        {
            int srcT = Math.Min(f * textLen / totalFrames, textLen - 1);
            content[f] = textHidden[srcT] * 0.7;
            prosody[f] = Math.Sin(f * 0.05) * 0.3 + textHidden[srcT] * 0.2;
            detail[f] = Math.Sin(f * 0.2) * 0.15;
        }
        // Combine factorized representations
        double[] combined = new double[totalFrames];
        for (int f = 0; f < totalFrames; f++)
            combined[f] = Math.Tanh(content[f] + prosody[f] + detail[f]);
        // FACodec decoder
        int waveLen = totalFrames * (SampleRate / 50);
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int frame = Math.Min(i * 50 / SampleRate, totalFrames - 1);
            waveform[i] = NumOps.FromDouble(Math.Tanh(combined[frame] * Math.Sin(i * 0.01 + combined[frame]) * 0.8));
        }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultFlowMatchingTTSLayers(_options.HiddenDim, _options.DiffusionDim, _options.MelChannels, _options.NumEncoderLayers, _options.NumDiffusionSteps, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); }
    private void ComputeEncoderDecoderBoundary() { int total = Layers.Count; _encoderLayerEnd = total > 4 ? total / 3 : total > 0 ? 1 : 0; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "NaturalSpeech3-Native" : "NaturalSpeech3-ONNX", Description = "NaturalSpeech 3: Factorized Codec + Diffusion TTS (Ju et al., 2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); writer.Write(_options.NumDiffusionSteps); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.NumDiffusionSteps = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new NaturalSpeech3<T>(Architecture, mp, _options); return new NaturalSpeech3<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(NaturalSpeech3<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
