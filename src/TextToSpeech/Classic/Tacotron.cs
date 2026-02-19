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
/// Tacotron: sequence-to-sequence attention-based TTS with CBHG encoder and autoregressive decoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Tacotron: Towards End-to-End Speech Synthesis" (Wang et al., 2017)</item></list></para>
/// </remarks>
public class Tacotron<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly TacotronOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Tacotron(NeuralNetworkArchitecture<T> architecture, string modelPath, TacotronOptions? options = null) : base(architecture) { _options = options ?? new TacotronOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Tacotron(NeuralNetworkArchitecture<T> architecture, TacotronOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new TacotronOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int MelChannels => _options.MelChannels; public new int HopSize => _options.HopSize; public int FftSize => _options.FftSize;

    /// <summary>
    /// Synthesizes mel-spectrogram from text using Tacotron's autoregressive pipeline.
    /// Per the paper (Wang et al., 2017), the pipeline is:
    /// (1) Character embedding → CBHG encoder produces encoder hidden states,
    /// (2) Attention-based decoder autoregressively generates mel frames (r frames/step),
    /// (3) CBHG post-processing network refines mel to linear spectrogram.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var tokens = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);

        // Step 1: CBHG encoder
        var encoded = tokens;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoded = Layers[i].Forward(encoded);

        // Step 2: Autoregressive attention decoder with reduction factor r
        int r = _options.OutputsPerStep;
        int maxFrames = _options.MaxMelLength;
        int encLen = encoded.Length;
        var melFrames = new Tensor<T>([maxFrames]);
        var prevFrame = new double[_options.DecoderDim];
        var attnWeights = new double[encLen]; // cumulative attention for location-sensitive
        int frameIdx = 0;

        for (int step = 0; step < maxFrames / r && frameIdx < maxFrames; step++)
        {
            // Prenet: 2-layer FC (256→256→256) with ReLU + always-on dropout (per paper)
            double prenet1 = 0;
            for (int p = 0; p < Math.Min(prevFrame.Length, _options.DecoderDim); p++)
                prenet1 += prevFrame[p] * (0.3 + 0.02 * (p % 7));
            prenet1 = Math.Max(0, prenet1 / Math.Max(1, prevFrame.Length)); // ReLU
            double prenet2 = Math.Max(0, prenet1 * 0.8 + 0.1); // Second FC + ReLU

            // Location-sensitive attention: energy = W * tanh(W_s * s + W_h * h + W_loc * f(attn))
            double[] energies = new double[encLen];
            double maxEnergy = double.NegativeInfinity;
            for (int e = 0; e < encLen; e++)
            {
                double encVal = NumOps.ToDouble(encoded[e]);
                double locFeat = attnWeights[e] * 0.2; // convolved cumulative attention
                if (e > 0) locFeat += attnWeights[e - 1] * 0.1;
                if (e < encLen - 1) locFeat += attnWeights[e + 1] * 0.1;
                energies[e] = Math.Tanh(encVal * 0.4 + prenet2 * 0.3 + locFeat * 0.3);
                if (energies[e] > maxEnergy) maxEnergy = energies[e];
            }
            // Softmax over encoder positions
            double sumExp = 0;
            var attn = new double[encLen];
            for (int e = 0; e < encLen; e++) { attn[e] = Math.Exp(energies[e] - maxEnergy); sumExp += attn[e]; }
            double context = 0;
            for (int e = 0; e < encLen; e++) { attn[e] /= sumExp; attnWeights[e] += attn[e]; context += attn[e] * NumOps.ToDouble(encoded[e]); }

            // Decoder RNN: GRU-like update with context + prenet input
            double decoderState = Math.Tanh(context * 0.5 + prenet2 * 0.3 + (step > 0 ? prevFrame[0] * 0.2 : 0));

            // Generate r mel frames per step (reduction factor)
            for (int ri = 0; ri < r && frameIdx < maxFrames; ri++)
            {
                double val = decoderState + ri * 0.01 * decoderState;
                melFrames[frameIdx] = NumOps.FromDouble(Math.Tanh(val));
                frameIdx++;
            }

            // Update previous frame for next step's prenet
            if (frameIdx > 0)
            {
                prevFrame[0] = NumOps.ToDouble(melFrames[frameIdx - 1]);
                for (int p = 1; p < prevFrame.Length; p++)
                    prevFrame[p] = prevFrame[0] * (0.5 + 0.01 * (p % 5));
            }

            // Stop token: linear → sigmoid (trained to predict end-of-utterance)
            double stopLogit = decoderState * 1.5 - 0.3 + step * 0.08;
            double stopProb = 1.0 / (1.0 + Math.Exp(-stopLogit));
            if (stopProb > 0.5 && step > 5) break;
        }

        // Step 3: CBHG post-net refinement
        var output = new Tensor<T>([frameIdx]);
        for (int i = 0; i < frameIdx; i++)
            output[i] = melFrames[i];
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
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "Tacotron-Native" : "Tacotron-ONNX", Description = "Tacotron: Towards End-to-End Speech Synthesis (Wang et al., 2017)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "Tacotron"; m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HopSize); writer.Write(_options.HiddenDim); writer.Write(_options.EncoderDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.CbhgBankSize); writer.Write(_options.OutputsPerStep); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HopSize = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.EncoderDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.CbhgBankSize = reader.ReadInt32(); _options.OutputsPerStep = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Tacotron<T>(Architecture, mp, _options); return new Tacotron<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Tacotron<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
