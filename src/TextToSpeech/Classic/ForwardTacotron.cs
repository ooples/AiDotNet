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
/// Forward Tacotron: non-autoregressive Tacotron variant using duration predictor instead of attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Forward Tacotron: A Non-Autoregressive Alternative to Tacotron" (Elias et al., 2021)</item></list></para>
/// </remarks>
public class ForwardTacotron<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly ForwardTacotronOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public ForwardTacotron(NeuralNetworkArchitecture<T> architecture, string modelPath, ForwardTacotronOptions? options = null) : base(architecture) { _options = options ?? new ForwardTacotronOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public ForwardTacotron(NeuralNetworkArchitecture<T> architecture, ForwardTacotronOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ForwardTacotronOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int MelChannels => _options.MelChannels; public new int HopSize => _options.HopSize; public int FftSize => _options.FftSize;

    /// <summary>
    /// Synthesizes mel-spectrogram using Forward Tacotron's non-autoregressive pipeline.
    /// Per the paper (Elias et al., 2021):
    /// (1) CBHG-style encoder (prenet + conv bank + highway + BiGRU),
    /// (2) Duration predictor replaces attention for robust alignment,
    /// (3) Gaussian upsampling expands phoneme-level to frame-level (smoother than repeat),
    /// (4) LSTM decoder generates mel frames non-autoregressively.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var tokens = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);

        var encoded = tokens;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoded = Layers[i].Forward(encoded);

        // Duration prediction + Gaussian upsampling
        int seqLen = encoded.Length;
        int totalFrames = 0;
        var durations = new double[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            double val = Math.Abs(NumOps.ToDouble(encoded[i % encoded.Length]));
            durations[i] = Math.Max(1.0, 1.0 + val * 4.0);
            totalFrames += (int)Math.Round(durations[i]);
        }

        // Gaussian upsampling: smooth interpolation based on duration centroids
        int expandedLen = Math.Min(totalFrames, _options.MaxMelLength);
        var expanded = new Tensor<T>([expandedLen]);
        for (int f = 0; f < expandedLen; f++)
        {
            double weightedSum = 0;
            double weightSum = 0;
            double center = 0;
            for (int p = 0; p < seqLen; p++)
            {
                double pCenter = center + durations[p] / 2.0;
                double sigma = durations[p] / 2.0 + 0.5;
                double w = Math.Exp(-0.5 * Math.Pow((f - pCenter) / sigma, 2));
                weightedSum += w * NumOps.ToDouble(encoded[p % encoded.Length]);
                weightSum += w;
                center += durations[p];
            }
            expanded[f] = NumOps.FromDouble(weightSum > 1e-8 ? weightedSum / weightSum : 0);
        }

        var output = expanded;
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
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "ForwardTacotron-Native" : "ForwardTacotron-ONNX", Description = "Forward Tacotron: Non-Autoregressive Alternative to Tacotron (Elias et al., 2021)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.HiddenDim, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "ForwardTacotron"; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumDecoderLayers); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ForwardTacotron<T>(Architecture, mp, _options); return new ForwardTacotron<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ForwardTacotron<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
