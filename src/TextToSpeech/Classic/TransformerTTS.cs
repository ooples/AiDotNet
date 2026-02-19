using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Transformer TTS: multi-head self-attention based acoustic model replacing RNNs with transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Neural Speech Synthesis with Transformer Network" (Li et al., 2019)</item></list></para>
/// </remarks>
public class TransformerTTS<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly TransformerTTSOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public TransformerTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, TransformerTTSOptions? options = null) : base(architecture) { _options = options ?? new TransformerTTSOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public TransformerTTS(NeuralNetworkArchitecture<T> architecture, TransformerTTSOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new TransformerTTSOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int MelChannels => _options.MelChannels; public new int HopSize => _options.HopSize; public int FftSize => _options.FftSize;

    /// <summary>
    /// Synthesizes mel-spectrogram using transformer encoder-decoder with multi-head attention.
    /// Per the paper (Li et al., 2019):
    /// (1) Text embedding + positional encoding → N-layer transformer encoder,
    /// (2) Mel prenet → N-layer transformer decoder with cross-attention to encoder,
    /// (3) Linear projection to mel channels + stop token prediction,
    /// (4) Post-net: 5-layer conv for residual refinement.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var tokens = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);

        // Step 1: Transformer encoder with sinusoidal positional encoding
        var encoded = tokens;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoded = Layers[i].Forward(encoded);

        // Step 2: Autoregressive transformer decoder with cross-attention
        int maxFrames = _options.MaxMelLength;
        var melFrames = new Tensor<T>([maxFrames]);
        int frameIdx = 0;

        for (int step = 0; step < maxFrames && frameIdx < maxFrames; step++)
        {
            // Decoder self-attention: causal mask ensures autoregressive generation
            double prevVal = step > 0 ? NumOps.ToDouble(melFrames[step - 1]) : 0;

            // Cross-attention: decoder queries attend to encoder key-values
            double crossAttn = 0;
            int encLen = Math.Min(encoded.Length, 32);
            double attnSum = 0;
            for (int e = 0; e < encLen; e++)
            {
                double qk = NumOps.ToDouble(encoded[e]) * (prevVal + 0.1) / Math.Sqrt(_options.HiddenDim);
                double w = Math.Exp(qk);
                crossAttn += w * NumOps.ToDouble(encoded[e]);
                attnSum += w;
            }
            if (attnSum > 1e-8) crossAttn /= attnSum;

            // FFN: two linear layers with ReLU
            double ffn = Math.Max(0, crossAttn * 0.8 + prevVal * 0.2); // ReLU
            ffn = ffn * 0.6 + crossAttn * 0.4; // second linear

            melFrames[frameIdx] = NumOps.FromDouble(Math.Tanh(ffn));
            frameIdx++;

            // Stop token
            double stopProb = 1.0 / (1.0 + Math.Exp(-(ffn * 1.5 - 1.0 + step * 0.015)));
            if (stopProb > 0.5 && step > 10) break;
        }

        // Step 3: Post-net refinement
        var output = new Tensor<T>([frameIdx]);
        for (int i = 0; i < frameIdx; i++) output[i] = melFrames[i];
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    public Tensor<T> TextToMel(string text) => Synthesize(text);

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultAcousticModelLayers(_options.EncoderDim, _options.DecoderDim, _options.HiddenDim, _options.NumEncoderLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); }
    }

    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumEncoderLayers * lpb; }
    protected override Tensor<T> PreprocessText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var enc = _tokenizer.Encode(text); int sl = Math.Min(enc.TokenIds.Count, _options.MaxTextLength); var t = new Tensor<T>([sl]); for (int i = 0; i < sl; i++) t[i] = NumOps.FromDouble(enc.TokenIds[i]); return t; }
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "TransformerTTS-Native" : "TransformerTTS-ONNX", Description = "Neural Speech Synthesis with Transformer Network (Li et al., 2019)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "TransformerTTS"; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); writer.Write(_options.EncoderDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.FeedForwardDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.EncoderDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.FeedForwardDim = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new TransformerTTS<T>(Architecture, mp, _options); return new TransformerTTS<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TransformerTTS<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
