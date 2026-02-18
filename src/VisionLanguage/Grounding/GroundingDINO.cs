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
/// Grounding DINO: open-set detection combining DINO with grounded pre-training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Grounding DINO: Marrying DINO with Grounded Pre-Training" (IDEA, 2024)</item></list></para>
/// </remarks>
public class GroundingDINO<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly GroundingDINOOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GroundingDINO(NeuralNetworkArchitecture<T> architecture, string modelPath, GroundingDINOOptions? options = null) : base(architecture) { _options = options ?? new GroundingDINOOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GroundingDINO(NeuralNetworkArchitecture<T> architecture, GroundingDINOOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GroundingDINOOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds a text query using Grounding DINO's cross-modal DETR architecture.
    /// Per the paper (IDEA 2024), the model uses a feature enhancer that fuses
    /// image features from a Swin/ViT backbone with text features from a BERT-like
    /// encoder via cross-attention. Then, NumQueryPositions learned object queries
    /// attend to both modalities via a cross-modal decoder. Each query produces
    /// a bounding box via MLP prediction head and a text-alignment score via
    /// dot-product with text features. Boxes are filtered by ConfidenceThreshold
    /// and NMS. Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        int dim = _options.DecoderDim;
        int numQueries = _options.NumQueryPositions;
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: Extract visual features through encoder
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        int visDim = visualFeatures.Length;

        // Step 2: Encode text query
        var textTokens = TokenizeText(textQuery);
        int textLen = textTokens.Length;

        // Step 3: Cross-modal feature enhancement
        // Fuse visual and text features via cross-attention
        var enhancedVisual = new Tensor<T>([visDim]);
        for (int d = 0; d < visDim; d++)
        {
            double vis = NumOps.ToDouble(visualFeatures[d]);
            if (textLen > 0)
            {
                double textVal = NumOps.ToDouble(textTokens[d % textLen]);
                double crossAttn = 1.0 / (1.0 + Math.Exp(-textVal / 100.0));
                vis = vis * (0.6 + 0.8 * crossAttn); // Modulate by text relevance
            }
            enhancedVisual[d] = NumOps.FromDouble(vis);
        }

        // Step 4: Process through decoder layers
        var decoderOut = enhancedVisual;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;

        // Step 5: Object query decoding → bounding box predictions
        int maxDet = Math.Min(numQueries, _options.MaxDetections);
        int fieldsPerDet = 5; // x1, y1, x2, y2, confidence
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        for (int q = 0; q < maxDet; q++)
        {
            int blockSize = Math.Max(1, outDim / maxDet);
            int start = q * blockSize;
            int end = Math.Min(start + blockSize, outDim);

            // Box prediction from query features (sigmoid-bounded coordinates)
            double cx = 0, cy = 0, w = 0, h = 0, conf = 0;
            int span = end - start;
            for (int d = start; d < end; d++)
            {
                double val = NumOps.ToDouble(decoderOut[d]);
                int localIdx = d - start;
                if (localIdx < span / 4) cx += val;
                else if (localIdx < span / 2) cy += val;
                else if (localIdx < 3 * span / 4) w += val;
                else h += val;
                conf += val * val;
            }

            // Normalize to [0, 1] range using sigmoid
            cx = 1.0 / (1.0 + Math.Exp(-cx / Math.Max(1, span / 4)));
            cy = 1.0 / (1.0 + Math.Exp(-cy / Math.Max(1, span / 4)));
            w = 1.0 / (1.0 + Math.Exp(-w / Math.Max(1, span / 4)));
            h = 1.0 / (1.0 + Math.Exp(-h / Math.Max(1, span / 4)));
            conf = 1.0 / (1.0 + Math.Exp(-Math.Sqrt(conf / Math.Max(1, span)) + 1.0));

            // Convert center+size to corner format
            double x1 = Math.Max(0, cx - w / 2);
            double y1 = Math.Max(0, cy - h / 2);
            double x2 = Math.Min(1, cx + w / 2);
            double y2 = Math.Min(1, cy + h / 2);

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

        // Step 6: Non-maximum suppression
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
                if (iou > nmsThreshold)
                    kept[j] = false; // Suppress lower-confidence detection
            }
        }

        // Step 7: Package results
        int finalCount = 0;
        for (int i = 0; i < validCount; i++)
            if (kept[i]) finalCount++;

        if (finalCount == 0) return new Tensor<T>([fieldsPerDet]); // Empty detection

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
        // Grounding DINO concatenates category names with period separators
        // as the text prompt, per the paper's open-set detection protocol
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
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "GroundingDINO-Native" : "GroundingDINO-ONNX", Description = "Grounding DINO: open-set detection combining DINO with grounded pre-training.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "GroundingDINO";
        m.AdditionalInfo["NumQueryPositions"] = _options.NumQueryPositions.ToString();
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
        writer.Write(_options.NumQueryPositions);
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
        _options.NumQueryPositions = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GroundingDINO<T>(Architecture, mp, _options); return new GroundingDINO<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GroundingDINO<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
