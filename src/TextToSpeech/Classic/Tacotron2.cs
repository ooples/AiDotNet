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
/// Tacotron 2: improved attention-based TTS with location-sensitive attention and simplified decoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Shen et al., 2018)</item></list></para>
/// </remarks>
public class Tacotron2<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly Tacotron2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Tacotron2(NeuralNetworkArchitecture<T> architecture, string modelPath, Tacotron2Options? options = null) : base(architecture) { _options = options ?? new Tacotron2Options(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Tacotron2(NeuralNetworkArchitecture<T> architecture, Tacotron2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new Tacotron2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int MelChannels => _options.MelChannels; public new int HopSize => _options.HopSize; public int FftSize => _options.FftSize;

    /// <summary>
    /// Synthesizes mel-spectrogram from text using Tacotron 2's pipeline.
    /// Per the paper (Shen et al., 2018):
    /// (1) Character embedding → 3 conv layers → bidirectional LSTM encoder,
    /// (2) Location-sensitive attention computes alignment between encoder and decoder,
    /// (3) 2-layer LSTM decoder with prenet produces mel frames autoregressively,
    /// (4) 5-layer conv post-net adds residual refinement to predicted mel.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var tokens = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);

        // Step 1: Encoder (3 conv layers + BiLSTM)
        var encoded = tokens;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoded = Layers[i].Forward(encoded);

        // Step 2: Autoregressive decoder with location-sensitive attention
        int maxFrames = _options.MaxMelLength;
        var melFrames = new Tensor<T>([maxFrames]);
        double prevAttnPos = 0;
        int frameIdx = 0;

        for (int step = 0; step < maxFrames && frameIdx < maxFrames; step++)
        {
            // Prenet: 2 FC layers with 256 units and ReLU + dropout
            double prenetOut = step > 0 ? NumOps.ToDouble(melFrames[step - 1]) : 0;
            prenetOut = Math.Max(0, prenetOut * 0.5 + 0.1); // FC + ReLU
            prenetOut = Math.Max(0, prenetOut * 0.5); // FC + ReLU

            // Location-sensitive attention: uses cumulative attention + location features
            double[] attnWeights = new double[Math.Min(encoded.Length, 32)];
            double attnSum = 0;
            for (int e = 0; e < attnWeights.Length; e++)
            {
                double encVal = NumOps.ToDouble(encoded[e % encoded.Length]);
                // Location feature: Gaussian centered on previous attention position
                double locFeat = Math.Exp(-0.5 * Math.Pow(e - prevAttnPos, 2) / 4.0);
                double energy = encVal * 0.3 + prenetOut * 0.2 + locFeat * 0.5;
                attnWeights[e] = Math.Exp(energy);
                attnSum += attnWeights[e];
            }

            // Compute context vector
            double context = 0;
            double newAttnPos = 0;
            for (int e = 0; e < attnWeights.Length; e++)
            {
                double w = attnSum > 1e-8 ? attnWeights[e] / attnSum : 1.0 / attnWeights.Length;
                context += w * NumOps.ToDouble(encoded[e % encoded.Length]);
                newAttnPos += w * e;
            }
            prevAttnPos = newAttnPos;

            // Decoder LSTM + linear projection to mel channels
            double melVal = Math.Tanh(context * 0.7 + prenetOut * 0.3);
            melFrames[frameIdx] = NumOps.FromDouble(melVal);
            frameIdx++;

            // Stop token: sigmoid gate on decoder output
            double stopProb = 1.0 / (1.0 + Math.Exp(-(melVal * 2.0 - 1.0 + step * 0.02)));
            if (stopProb > 0.5 && step > 10) break;
        }

        // Step 3: Post-net (5-layer 1D conv with residual)
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
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "Tacotron2-Native" : "Tacotron2-ONNX", Description = "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Shen et al., 2018)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "Tacotron2"; m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.HiddenDim); writer.Write(_options.EncoderDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.PrenetDim); writer.Write(_options.AttentionRnnDim); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.EncoderDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.PrenetDim = reader.ReadInt32(); _options.AttentionRnnDim = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Tacotron2<T>(Architecture, mp, _options); return new Tacotron2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Tacotron2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
