using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.MultiModal;
/// <summary>LLaMA-Omni: seamless speech interaction with LLM, enabling low-latency speech-to-speech conversation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "LLaMA-Omni: Seamless Speech Interaction with Large Language Model" (Fang et al., 2024)</item></list></para></remarks>
public class LlamaOmni<T> : TtsModelBase<T>, ICodecTts<T>, IStreamingTts<T>
{
    private readonly LlamaOmniOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public LlamaOmni(NeuralNetworkArchitecture<T> architecture, string modelPath, LlamaOmniOptions? options = null) : base(architecture) { _options = options ?? new LlamaOmniOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public LlamaOmni(NeuralNetworkArchitecture<T> architecture, LlamaOmniOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new LlamaOmniOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public int NumCodebooks => _options.NumCodebooks; public int CodebookSize => _options.CodebookSize; public int CodecFrameRate => _options.CodecFrameRate;
    private int _streamPosition; private string _streamText = string.Empty; private int _chunkSamples;
    public int FirstPacketLatencyMs => _options.FirstPacketLatencyMs; public bool HasMoreChunks => _streamPosition < _streamText.Length;
    /// Synthesizes speech using LlamaOmni's neural codec language model pipeline.
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }
    public Tensor<T> EncodeToTokens(Tensor<T> audio) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(audio); return Predict(audio); }
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens); return Predict(tokens); }
    public Tensor<T> SynthesizeFirstChunk(string text, int chunkSize) { _streamText = text; _streamPosition = 0; _chunkSamples = chunkSize; return SynthesizeNextChunk(); }
    public Tensor<T> SynthesizeNextChunk()
    {
        if (_streamPosition >= _streamText.Length) return new Tensor<T>([0]);
        int chunkTextLen = Math.Min(20, _streamText.Length - _streamPosition);
        string chunk = _streamText.Substring(_streamPosition, chunkTextLen);
        _streamPosition += chunkTextLen;
        var audio = Synthesize(chunk);
        if (audio.Length > _chunkSamples) { var trimmed = new Tensor<T>([_chunkSamples]); for (int i = 0; i < _chunkSamples; i++) trimmed[i] = audio[i]; return trimmed; }
        return audio;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(_options.TextEncoderDim, _options.LLMDim, _options.NumCodebooks * _options.CodebookSize, _options.NumEncoderLayers, _options.NumLLMLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); try { var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); } finally { SetTrainingMode(false); } }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "LlamaOmni-Native" : "LlamaOmni-ONNX", Description = "LlamaOmni TTS", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.LLMDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.NumCodebooks); writer.Write(_options.LLMDim); writer.Write(_options.CodebookSize); writer.Write(_options.DropoutRate); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.NumLLMLayers); writer.Write(_options.TextEncoderDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.NumCodebooks = reader.ReadInt32(); _options.LLMDim = reader.ReadInt32();  _options.CodebookSize = reader.ReadInt32(); _options.DropoutRate = reader.ReadDouble(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.NumLLMLayers = reader.ReadInt32(); _options.TextEncoderDim = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new LlamaOmni<T>(Architecture, mp, _options); return new LlamaOmni<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LlamaOmni<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
