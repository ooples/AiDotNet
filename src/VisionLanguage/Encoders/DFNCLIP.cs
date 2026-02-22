using AiDotNet.Helpers; using AiDotNet.Interfaces; using AiDotNet.Models.Options; using AiDotNet.NeuralNetworks; using AiDotNet.Onnx; using AiDotNet.Optimizers; using AiDotNet.Tokenization; using AiDotNet.Tokenization.Interfaces; using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// DFN-CLIP (Data Filtering Networks for CLIP) model using filtered high-quality training data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DFN-CLIP (Fang et al., 2023) uses a small CLIP model to score and filter image-text pairs
/// from a large noisy pool, then trains a larger model on only high-quality data. Achieves 83.0%
/// zero-shot on ImageNet with ViT-H/14.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Data Filtering Networks" (Fang et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> DFNCLIP is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class DFNCLIP<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    private readonly DFNCLIPOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed; private int _visionLayerEnd;

    public DFNCLIP(NeuralNetworkArchitecture<T> architecture, string imageEncoderModelPath, DFNCLIPOptions? options = null) : base(architecture) { _options = options ?? new DFNCLIPOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.VisionEmbeddingDim; if (string.IsNullOrWhiteSpace(imageEncoderModelPath)) throw new ArgumentException("Image encoder model path cannot be null or empty.", nameof(imageEncoderModelPath)); if (!File.Exists(imageEncoderModelPath)) throw new FileNotFoundException($"ONNX model not found: {imageEncoderModelPath}", imageEncoderModelPath); _options.ImageEncoderModelPath = imageEncoderModelPath; OnnxImageEncoder = new OnnxModel<T>(imageEncoderModelPath, _options.OnnxOptions); if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp)) { if (!File.Exists(tp)) throw new FileNotFoundException($"Text ONNX not found: {tp}", tp); OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions); } _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public DFNCLIP(NeuralNetworkArchitecture<T> architecture, DFNCLIPOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DFNCLIPOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.VisionEmbeddingDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.VisionEmbeddingDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxSequenceLength => _options.MaxSequenceLength; public int TextEmbeddingDimension => _options.TextEmbeddingDim; public int ProjectionDimension => _options.ProjectionDim; public T Temperature => NumOps.FromDouble(_options.Temperature);
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxImageEncoder is not null) return L2Normalize(OnnxImageEncoder.Run(p)); return L2Normalize(ForwardVisionEncoder(p)); }
    public Tensor<T> EncodeText(string text) { ThrowIfDisposed(); var t = TokenizeText(text); if (IsOnnxMode && OnnxTextEncoder is not null) return L2Normalize(OnnxTextEncoder.Run(t)); return L2Normalize(ForwardTextEncoder(t)); }
    public Tensor<T>[] EncodeTexts(string[] texts) { var e = new Tensor<T>[texts.Length]; for (int i = 0; i < texts.Length; i++) e[i] = EncodeText(texts[i]); return e; }
    public T ComputeSimilarity(Tensor<T> image, string text) => CosineSimilarity(EncodeImage(image), EncodeText(text));
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels) { var ie = EncodeImage(image); var te = EncodeTexts(labels); var logits = new Tensor<T>([labels.Length]); double temp = _options.Temperature; for (int i = 0; i < labels.Length; i++) logits[i] = NumOps.FromDouble(NumOps.ToDouble(CosineSimilarity(ie, te[i])) / temp); var probs = Softmax(logits); var r = new Dictionary<string, T>(); for (int i = 0; i < labels.Length; i++) r[labels[i]] = probs[i]; return r; }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultOpenCLIPLayers(_options.VisionEmbeddingDim, _options.TextEmbeddingDim, _options.ProjectionDim, _options.NumVisionLayers, _options.NumTextLayers, _options.NumVisionHeads, _options.NumTextHeads, _options.DropoutRate)); int lpb = _options.DropoutRate > 0 ? 6 : 5; _visionLayerEnd = 2 + _options.NumVisionLayers * lpb; } }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxImageEncoder is not null) return OnnxImageEncoder.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "DFN-CLIP-Native" : "DFN-CLIP-ONNX", Description = "DFN-CLIP: Data Filtering Networks (Fang et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.ProjectionDim, Complexity = _options.NumVisionLayers + _options.NumTextLayers }; m.AdditionalInfo["Architecture"] = "DFN-CLIP"; m.AdditionalInfo["FilteringThreshold"] = _options.FilteringThreshold.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ImageEncoderModelPath ?? string.Empty); writer.Write(_options.TextEncoderModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionEmbeddingDim); writer.Write(_options.TextEmbeddingDim); writer.Write(_options.ProjectionDim); writer.Write(_options.Temperature); writer.Write(_options.FilteringThreshold); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string ip = reader.ReadString(); if (!string.IsNullOrEmpty(ip)) _options.ImageEncoderModelPath = ip; string tp = reader.ReadString(); if (!string.IsNullOrEmpty(tp)) _options.TextEncoderModelPath = tp; _options.ImageSize = reader.ReadInt32(); _options.VisionEmbeddingDim = reader.ReadInt32(); _options.TextEmbeddingDim = reader.ReadInt32(); _options.ProjectionDim = reader.ReadInt32(); _options.Temperature = reader.ReadDouble(); _options.FilteringThreshold = reader.ReadDouble(); if (!_useNativeMode && _options.ImageEncoderModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxImageEncoder = new OnnxModel<T>(p, _options.OnnxOptions); if (_options.TextEncoderModelPath is { } t2 && !string.IsNullOrEmpty(t2)) OnnxTextEncoder = new OnnxModel<T>(t2, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ImageEncoderModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new DFNCLIP<T>(Architecture, mp, _options); return new DFNCLIP<T>(Architecture, _options); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var enc = _tokenizer.Encode(text); int sl = Math.Min(enc.TokenIds.Count, _options.MaxSequenceLength); var tk = new Tensor<T>([sl]); for (int i = 0; i < sl; i++) tk[i] = NumOps.FromDouble(enc.TokenIds[i]); return tk; }
    private Tensor<T> ForwardVisionEncoder(Tensor<T> input) { var c = input; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return c; }
    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens) { var c = tokens; for (int i = _visionLayerEnd; i < Layers.Count; i++) c = Layers[i].Forward(c); return c; }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DFNCLIP<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; if (disposing) { OnnxImageEncoder?.Dispose(); OnnxTextEncoder?.Dispose(); } base.Dispose(disposing); }
}
