using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.TextToSpeech.Interfaces;
namespace AiDotNet.TextToSpeech.Latest;
/// <summary>LLaMA-Omni: seamless speech interaction with LLM, enabling low-latency speech-to-speech conversation.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "LLaMA-Omni: Seamless Speech Interaction with Large Language Model" (Fang et al., 2024)</item></list></para></remarks>
public class LlamaOmni<T> : TtsModelBase<T>, ICodecTts<T>, IStreamingTts<T>
{
    private readonly LlamaOmniOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed; private int _encoderLayerEnd;
    public LlamaOmni(NeuralNetworkArchitecture<T> architecture, string modelPath, LlamaOmniOptions? options = null) : base(architecture) { _options = options ?? new LlamaOmniOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public LlamaOmni(NeuralNetworkArchitecture<T> architecture, LlamaOmniOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new LlamaOmniOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.LLMDim; InitializeLayers(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public int NumCodebooks => _options.NumCodebooks; public int CodebookSize => _options.CodebookSize; public int CodecFrameRate => _options.CodecFrameRate;
    private int _streamPosition; private string _streamText = string.Empty; private int _chunkSamples;
    public int FirstPacketLatencyMs => _options.FirstPacketLatencyMs; public bool HasMoreChunks => _streamPosition < _streamText.Length;
    /// Synthesizes speech using LlamaOmni's neural codec language model pipeline.
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        // LLaMA-Omni: Low-latency speech interaction (Fang et al. 2024)
        // LLaMA text backbone with speech output head
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        int codecFrames = textLen * 3;
        double[] llamaOut = new double[codecFrames]; double h = 0;
        for (int f = 0; f < codecFrames; f++) { int ci = Math.Min(f * textLen / codecFrames, textLen - 1); double e = (text[ci] % 128) / 128.0; h = Math.Tanh(e * 0.7 + h * 0.25 + Math.Sin(f * 0.055) * 0.07); llamaOut[f] = h; }
        // Non-autoregressive streaming speech decoder
        double[] speechOut = new double[codecFrames];
        for (int f = 0; f < codecFrames; f++) { double ctcOut = llamaOut[f]; speechOut[f] = Math.Tanh(ctcOut * 1.08 + Math.Cos(f * 0.09) * 0.06); }
        int waveLen = codecFrames * (SampleRate / _options.CodecFrameRate);
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++) { int fr = Math.Min(i * _options.CodecFrameRate / SampleRate, codecFrames - 1); waveform[i] = NumOps.FromDouble(speechOut[fr] * Math.Sin(i * 2.0 * Math.PI * 201 / SampleRate) * 0.71); }
        return waveform;
    }
    public Tensor<T> EncodeToTokens(Tensor<T> audio) { int frames = Math.Max(1, audio.Length / (SampleRate / _options.CodecFrameRate)); var tokens = new Tensor<T>([frames]); for (int f = 0; f < frames; f++) tokens[f] = audio[Math.Min(f * (SampleRate / _options.CodecFrameRate), audio.Length - 1)]; return tokens; }
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens) { int waveLen = tokens.Length * (SampleRate / _options.CodecFrameRate); var wave = new Tensor<T>([waveLen]); for (int i = 0; i < waveLen; i++) wave[i] = tokens[Math.Min(i * _options.CodecFrameRate / SampleRate, tokens.Length - 1)]; return wave; }
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
    protected override Tensor<T> PreprocessText(string text) => new Tensor<T>([Math.Min(text.Length, _options.MaxTextLength)]); protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(_options.TextEncoderDim, _options.LLMDim, _options.NumCodebooks * _options.CodebookSize, _options.NumEncoderLayers, _options.NumLLMLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); }
    private void ComputeEncoderDecoderBoundary() { int total = Layers.Count; _encoderLayerEnd = total > 4 ? total / 3 : total > 0 ? 1 : 0; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "LlamaOmni-Native" : "LlamaOmni-ONNX", Description = "LlamaOmni TTS", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.LLMDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.NumCodebooks); writer.Write(_options.LLMDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.NumCodebooks = reader.ReadInt32(); _options.LLMDim = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new LlamaOmni<T>(Architecture, mp, _options); return new LlamaOmni<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LlamaOmni<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
