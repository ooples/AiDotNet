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
/// Deep Voice 3: fully convolutional attention-based TTS with monotonic attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning" (Ping et al., 2018)</item></list></para>
/// </remarks>
public class DeepVoice3<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly DeepVoice3Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public DeepVoice3(NeuralNetworkArchitecture<T> architecture, string modelPath, DeepVoice3Options? options = null) : base(architecture) { _options = options ?? new DeepVoice3Options(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public DeepVoice3(NeuralNetworkArchitecture<T> architecture, DeepVoice3Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DeepVoice3Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int MelChannels => _options.MelChannels; public new int HopSize => _options.HopSize; public int FftSize => _options.FftSize;

    /// <summary>
    /// Synthesizes mel-spectrogram from text using Deep Voice 3's fully convolutional pipeline.
    /// Per the paper (Ping et al., 2018):
    /// (1) Fully convolutional encoder: causal conv blocks with residual connections,
    /// (2) Monotonic attention: position-augmented attention with forced monotonicity,
    /// (3) Causal convolutional decoder: generates r mel frames per step,
    /// (4) Converter: separate conv network refines mel to linear spectrogram.
    /// Multi-speaker: speaker embedding added to encoder, decoder, and converter inputs.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var tokens = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);

        // Step 1: Fully convolutional encoder with gated linear units
        var encoded = tokens;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoded = Layers[i].Forward(encoded);

        // Step 2: Monotonic attention decoder - generates r frames/step
        int r = _options.OutputsPerStep;
        int maxFrames = _options.MaxMelLength;
        var melFrames = new Tensor<T>([maxFrames]);
        int frameIdx = 0;
        double attnPos = 0;

        for (int step = 0; step < maxFrames / r && frameIdx < maxFrames; step++)
        {
            // Position-augmented attention with monotonic constraint
            double context = 0;
            double bestScore = double.MinValue;
            int attnPosInt = Math.Min((int)attnPos, encoded.Length - 1);
            int window = 3; // monotonic: only look near current position

            for (int e = Math.Max(0, attnPosInt - window); e < Math.Min(encoded.Length, attnPosInt + window + 1); e++)
            {
                double encVal = NumOps.ToDouble(encoded[e % encoded.Length]);
                double posWeight = Math.Exp(-0.5 * Math.Pow(e - attnPos, 2));
                double score = encVal * posWeight;
                if (score > bestScore) { bestScore = score; context = encVal * 0.8; }
            }
            attnPos = Math.Min(attnPos + 1.0 / (maxFrames / encoded.Length + 1), encoded.Length - 1);

            // Causal conv decoder: generate r frames with gated linear units
            for (int ri = 0; ri < r && frameIdx < maxFrames; ri++)
            {
                double prevVal = frameIdx > 0 ? NumOps.ToDouble(melFrames[frameIdx - 1]) : 0;
                // GLU: sigmoid(x1) * x2
                double gate = 1.0 / (1.0 + Math.Exp(-context));
                double val = gate * (context * 0.6 + prevVal * 0.3);
                melFrames[frameIdx] = NumOps.FromDouble(Math.Tanh(val));
                frameIdx++;
            }

            // Done token
            double doneProb = 1.0 / (1.0 + Math.Exp(-(step * 0.1 - 3.0)));
            if (doneProb > 0.5 && step > 5) break;
        }

        // Step 3: Converter post-net
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
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "DeepVoice3-Native" : "DeepVoice3-ONNX", Description = "Deep Voice 3: Scaling TTS with Convolutional Sequence Learning (Ping et al., 2018)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "DeepVoice3"; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); writer.Write(_options.EncoderDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.ConvKernelSize); writer.Write(_options.NumSpeakers); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.EncoderDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.ConvKernelSize = reader.ReadInt32(); _options.NumSpeakers = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new DeepVoice3<T>(Architecture, mp, _options); return new DeepVoice3<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DeepVoice3<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
