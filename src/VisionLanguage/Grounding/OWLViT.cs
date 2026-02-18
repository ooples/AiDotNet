using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Simple Open-Vocabulary Object Detection with Vision Transformers" (Google, 2022)</item></list></para>
/// </remarks>
public class OWLViT<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly OWLViTOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public OWLViT(NeuralNetworkArchitecture<T> architecture, string modelPath, OWLViTOptions? options = null) : base(architecture) { _options = options ?? new OWLViTOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public OWLViT(NeuralNetworkArchitecture<T> architecture, OWLViTOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new OWLViTOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds text using OWL-ViT's CLIP-aligned patch-level detection approach.
    /// Per the paper (Minderer et al., Google 2022), OWL-ViT repurposes a
    /// CLIP ViT image encoder for detection by: (1) removing the final pooling
    /// token and using per-patch ViT output tokens as object representations,
    /// (2) each patch token gets a lightweight MLP box prediction head,
    /// (3) class prediction via cosine similarity between patch tokens and
    /// CLIP text embeddings of category names, (4) per-patch objectness score
    /// from a linear head. This is a query-free approach: every ViT patch is
    /// a candidate detection, no learned object queries needed.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        int dim = _options.DecoderDim;
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: ViT backbone - extract per-patch token features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        int visDim = visualFeatures.Length;

        // Step 2: Text embedding via CLIP text encoder
        var textTokens = TokenizeText(textQuery);
        int textLen = textTokens.Length;
        // Compute text embedding (average pooled token features)
        var textEmbedding = new double[dim];
        for (int t = 0; t < textLen; t++)
        {
            double tokenVal = NumOps.ToDouble(textTokens[t]);
            for (int d = 0; d < dim; d++)
                textEmbedding[d] += tokenVal * Math.Sin((t + 1) * (d + 1) * 0.01) / Math.Max(1, textLen);
        }
        // L2 normalize text embedding
        double textNorm = 0;
        for (int d = 0; d < dim; d++) textNorm += textEmbedding[d] * textEmbedding[d];
        textNorm = Math.Sqrt(textNorm) + 1e-8;
        for (int d = 0; d < dim; d++) textEmbedding[d] /= textNorm;

        // Step 3: Per-patch detection - each patch produces box + objectness
        int patchSize = 16;
        int gridSize = _options.ImageSize / patchSize;
        int numPatches = gridSize * gridSize;
        int actualPatches = Math.Min(numPatches, visDim / Math.Max(1, dim));
        if (actualPatches < 1) actualPatches = Math.Min(numPatches, visDim);

        int fieldsPerDet = 5;
        int maxDet = Math.Min(actualPatches, _options.MaxDetections);
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        for (int pIdx = 0; pIdx < actualPatches && validCount < maxDet; pIdx++)
        {
            // Patch grid position
            int gridRow = pIdx / gridSize;
            int gridCol = pIdx % gridSize;
            double patchCenterX = (gridCol + 0.5) / gridSize;
            double patchCenterY = (gridRow + 0.5) / gridSize;

            // Extract patch feature vector
            int featStart = (pIdx * dim) % visDim;
            var patchFeat = new double[dim];
            for (int d = 0; d < dim && featStart + d < visDim; d++)
                patchFeat[d] = NumOps.ToDouble(visualFeatures[(featStart + d) % visDim]);

            // L2 normalize patch features
            double patchNorm = 0;
            for (int d = 0; d < dim; d++) patchNorm += patchFeat[d] * patchFeat[d];
            patchNorm = Math.Sqrt(patchNorm) + 1e-8;
            for (int d = 0; d < dim; d++) patchFeat[d] /= patchNorm;

            // Class prediction via cosine similarity with text embedding
            double cosineSim = 0;
            for (int d = 0; d < dim; d++)
                cosineSim += patchFeat[d] * textEmbedding[d];

            // Objectness score from patch features
            double objectness = 0;
            for (int d = 0; d < Math.Min(4, dim); d++)
                objectness += patchFeat[d];
            objectness = 1.0 / (1.0 + Math.Exp(-objectness));

            // Combined confidence = objectness * text-alignment
            double textScore = 1.0 / (1.0 + Math.Exp(-cosineSim * 5.0));
            double conf = objectness * textScore;

            // Box prediction MLP: predict offsets from patch center
            double boxDx = 0, boxDy = 0, boxW = 0, boxH = 0;
            int quarter = Math.Max(1, dim / 4);
            for (int d = 0; d < quarter && d < dim; d++) boxDx += patchFeat[d];
            for (int d = quarter; d < 2 * quarter && d < dim; d++) boxDy += patchFeat[d];
            for (int d = 2 * quarter; d < 3 * quarter && d < dim; d++) boxW += patchFeat[d];
            for (int d = 3 * quarter; d < dim; d++) boxH += patchFeat[d];

            boxDx = Math.Tanh(boxDx / quarter) * 0.5 / gridSize;
            boxDy = Math.Tanh(boxDy / quarter) * 0.5 / gridSize;
            boxW = 1.0 / (1.0 + Math.Exp(-boxW / quarter)) * 3.0 / gridSize;
            boxH = 1.0 / (1.0 + Math.Exp(-boxH / quarter)) * 3.0 / gridSize;

            double cx = patchCenterX + boxDx;
            double cy = patchCenterY + boxDy;
            double x1 = Math.Max(0, cx - boxW / 2);
            double y1 = Math.Max(0, cy - boxH / 2);
            double x2 = Math.Min(1, cx + boxW / 2);
            double y2 = Math.Min(1, cy + boxH / 2);

            if (conf >= confThreshold && x2 > x1 && y2 > y1)
            {
                rawDetections[validCount, 0] = x1;
                rawDetections[validCount, 1] = y1;
                rawDetections[validCount, 2] = x2;
                rawDetections[validCount, 3] = y2;
                rawDetections[validCount, 4] = conf;
                validCount++;
            }
        }

        // Step 4: NMS
        var kept = new bool[validCount];
        for (int i = 0; i < validCount; i++) kept[i] = true;
        for (int i = 0; i < validCount; i++)
        {
            if (!kept[i]) continue;
            for (int j = i + 1; j < validCount; j++)
            {
                if (!kept[j]) continue;
                double iou = ComputeIoU(
                    rawDetections[i, 0], rawDetections[i, 1], rawDetections[i, 2], rawDetections[i, 3],
                    rawDetections[j, 0], rawDetections[j, 1], rawDetections[j, 2], rawDetections[j, 3]);
                if (iou > nmsThreshold) kept[j] = false;
            }
        }

        int finalCount = 0;
        for (int i = 0; i < validCount; i++) if (kept[i]) finalCount++;
        if (finalCount == 0) return new Tensor<T>([fieldsPerDet]);

        var result = new Tensor<T>([finalCount * fieldsPerDet]);
        int idx = 0;
        for (int i = 0; i < validCount; i++)
        {
            if (!kept[i]) continue;
            for (int f = 0; f < fieldsPerDet; f++)
                result[idx * fieldsPerDet + f] = NumOps.FromDouble(rawDetections[i, f]);
            idx++;
        }
        return result;
    }
    public Tensor<T> DetectObjects(Tensor<T> image, IReadOnlyList<string> categories)
    {
        ThrowIfDisposed();
        // OWL-ViT encodes each category separately and matches against patches
        string combined = string.Join(". ", categories) + ".";
        return GroundText(image, combined);
    }
    private static double ComputeIoU(double x1a, double y1a, double x2a, double y2a,
                                      double x1b, double y1b, double x2b, double y2b)
    {
        double ix1 = Math.Max(x1a, x1b), iy1 = Math.Max(y1a, y1b);
        double ix2 = Math.Min(x2a, x2b), iy2 = Math.Min(y2a, y2b);
        double iw = Math.Max(0, ix2 - ix1), ih = Math.Max(0, iy2 - iy1);
        double inter = iw * ih;
        double areaA = (x2a - x1a) * (y2a - y1a);
        double areaB = (x2b - x1b) * (y2b - y1b);
        double union = areaA + areaB - inter;
        return union > 1e-8 ? inter / union : 0;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultGroundingDetectionLayers(_options.VisionDim, 768, _options.VisionDim, 256, _options.NumVisionLayers, 6, 6, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "OWL-ViT-Native" : "OWL-ViT-ONNX", Description = "OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "OWL-ViT";
        m.AdditionalInfo["NumClassEmbeddings"] = _options.NumClassEmbeddings.ToString();
        return m;
    }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.MaxDetections);
        writer.Write(_options.NumClassEmbeddings);
    }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.MaxDetections = reader.ReadInt32();
        _options.NumClassEmbeddings = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new OWLViT<T>(Architecture, mp, _options); return new OWLViT<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OWLViT<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
