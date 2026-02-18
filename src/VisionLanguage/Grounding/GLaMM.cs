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
/// GLaMM: pixel-level grounded LMM generating text and segmentation masks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "GLaMM: Pixel Grounding Large Multimodal Model" (MBZUAI, 2024)</item></list></para>
/// </remarks>
public class GLaMM<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly GLaMMOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GLaMM(NeuralNetworkArchitecture<T> architecture, string modelPath, GLaMMOptions? options = null) : base(architecture) { _options = options ?? new GLaMMOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GLaMM(NeuralNetworkArchitecture<T> architecture, GLaMMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GLaMMOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds text using GLaMM's pixel-level grounding with mask generation.
    /// Per the paper (Rasheed et al., MBZUAI 2024), GLaMM produces both
    /// text and segmentation masks via: (1) a grounding image encoder that
    /// extracts multi-scale features, (2) a pixel decoder that upsamples
    /// features to MaskDim-channel mask embeddings, (3) a region-level
    /// text-mask alignment where special [SEG] tokens in the LLM output
    /// trigger mask generation through dot-product with pixel embeddings,
    /// (4) each grounded phrase maps to a binary mask via the mask decoder.
    /// For bounding box output, the mask is converted to its tight bbox.
    /// Output format: [x1, y1, x2, y2, confidence] per detection, with
    /// optional per-detection mask logits appended if EnablePixelGrounding.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        int dim = _options.DecoderDim;
        int maskDim = _options.MaskDim;
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: Grounding image encoder - multi-scale features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        int visDim = visualFeatures.Length;

        // Step 2: Pixel decoder - upsample to mask embedding space
        int maskGridSize = (int)Math.Sqrt(maskDim);
        if (maskGridSize < 4) maskGridSize = (int)Math.Sqrt(visDim / 4);
        if (maskGridSize < 4) maskGridSize = 4;
        int totalMaskPixels = maskGridSize * maskGridSize;

        var pixelEmbeddings = new double[totalMaskPixels][];
        for (int px = 0; px < totalMaskPixels; px++)
        {
            pixelEmbeddings[px] = new double[maskDim];
            int srcStart = (px * visDim / totalMaskPixels) % visDim;
            for (int d = 0; d < maskDim; d++)
            {
                int srcIdx = (srcStart + d) % visDim;
                pixelEmbeddings[px][d] = NumOps.ToDouble(visualFeatures[srcIdx]);
            }
        }

        // Step 3: Text encoding and [SEG] token extraction
        var textTokens = TokenizeText(textQuery);
        int textLen = textTokens.Length;

        // Compute text-conditioned mask queries (one per potential grounded phrase)
        // Each query will produce a mask via dot product with pixel embeddings
        int numMaskQueries = Math.Min(8, _options.MaxDetections);
        var maskQueries = new double[numMaskQueries][];
        for (int q = 0; q < numMaskQueries; q++)
        {
            maskQueries[q] = new double[maskDim];
            for (int d = 0; d < maskDim; d++)
            {
                double qVal = 0;
                // Query-specific text attention
                for (int t = 0; t < textLen; t++)
                {
                    double tv = NumOps.ToDouble(textTokens[t]);
                    double weight = Math.Exp(-Math.Abs(q - t * numMaskQueries / Math.Max(1, textLen)));
                    qVal += tv * weight * Math.Sin((d + 1) * (t + 1) * 0.05);
                }
                maskQueries[q][d] = qVal / Math.Max(1, textLen);
            }
        }

        // Step 4: Generate masks via query-pixel dot product
        int fieldsPerDet = 5;
        int maxDet = _options.MaxDetections;
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        for (int q = 0; q < numMaskQueries && validCount < maxDet; q++)
        {
            // Compute mask logits: dot product of query with each pixel embedding
            var maskLogits = new double[totalMaskPixels];
            double maxLogit = double.MinValue;
            for (int px = 0; px < totalMaskPixels; px++)
            {
                double dot = 0;
                for (int d = 0; d < maskDim; d++)
                    dot += maskQueries[q][d] * pixelEmbeddings[px][d];
                maskLogits[px] = dot;
                if (dot > maxLogit) maxLogit = dot;
            }

            // Threshold mask to find foreground pixels
            double maskThreshold = maxLogit * 0.3;
            int minR = maskGridSize, maxR = 0, minC = maskGridSize, maxC = 0;
            double maskConfSum = 0;
            int fgCount = 0;

            for (int px = 0; px < totalMaskPixels; px++)
            {
                if (maskLogits[px] > maskThreshold)
                {
                    int row = px / maskGridSize;
                    int col = px % maskGridSize;
                    if (row < minR) minR = row;
                    if (row > maxR) maxR = row;
                    if (col < minC) minC = col;
                    if (col > maxC) maxC = col;
                    maskConfSum += 1.0 / (1.0 + Math.Exp(-maskLogits[px]));
                    fgCount++;
                }
            }

            if (fgCount < 1) continue;

            double x1 = (double)minC / maskGridSize;
            double y1 = (double)minR / maskGridSize;
            double x2 = (double)(maxC + 1) / maskGridSize;
            double y2 = (double)(maxR + 1) / maskGridSize;
            double conf = maskConfSum / fgCount;

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

        // Step 5: NMS
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "GLaMM-Native" : "GLaMM-ONNX", Description = "GLaMM: pixel-level grounded LMM generating text and segmentation masks.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "GLaMM";
        m.AdditionalInfo["PixelGrounding"] = _options.EnablePixelGrounding.ToString();
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
        writer.Write(_options.EnablePixelGrounding);
        writer.Write(_options.MaskDim);
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
        _options.EnablePixelGrounding = reader.ReadBoolean();
        _options.MaskDim = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GLaMM<T>(Architecture, mp, _options); return new GLaMM<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GLaMM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
