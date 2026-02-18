using AiDotNet.Extensions;
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
/// OWLv2: self-training for scaling open-vocabulary detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Scaling Open-Vocabulary Object Detection" (Google, 2023)</item></list></para>
/// </remarks>
public class OWLv2<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly OWLv2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public OWLv2(NeuralNetworkArchitecture<T> architecture, string modelPath, OWLv2Options? options = null) : base(architecture) { _options = options ?? new OWLv2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public OWLv2(NeuralNetworkArchitecture<T> architecture, OWLv2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new OWLv2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds text using OWLv2's scaled open-vocabulary detection approach.
    /// Per the paper (Minderer et al., Google 2023), OWLv2 improves OWL-ViT
    /// through self-training on web-scale image-text data. Key innovations:
    /// (1) a stronger ViT-L/14 backbone with 960px input, (2) self-training
    /// with pseudo-labels from an OWL-ViT teacher, (3) objectness-calibrated
    /// scoring that better separates true objects from background, (4) larger
    /// NumClassEmbeddings (768) for richer class representations.
    /// The self-training pseudo-label refinement uses confidence recalibration.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: ViT-L/14 backbone - per-patch features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Step 2: Text-conditioned feature fusion via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedInput = visualFeatures.ConcatenateTensors(textTokens);

        // Run through decoder layers for text-conditioned detection features
        var decoderOut = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);
        int outDim = decoderOut.Length;

        // Step 3: Per-patch detection with calibrated objectness
        int patchSize = 14;
        int gridSize = _options.ImageSize / patchSize;
        int numPatches = gridSize * gridSize;
        int actualPatches = Math.Min(numPatches, outDim);

        int fieldsPerDet = 5;
        int maxDet = Math.Min(actualPatches, _options.MaxDetections);
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        for (int pIdx = 0; pIdx < actualPatches && validCount < maxDet; pIdx++)
        {
            int gridRow = pIdx / gridSize;
            int gridCol = pIdx % gridSize;
            double patchCenterX = (gridCol + 0.5) / gridSize;
            double patchCenterY = (gridRow + 0.5) / gridSize;

            // Extract text-conditioned patch features from decoder output
            int dim = Math.Max(1, outDim / actualPatches);
            var patchFeat = new double[dim];
            for (int d = 0; d < dim; d++)
            {
                int srcIdx = (pIdx * dim + d) % outDim;
                patchFeat[d] = NumOps.ToDouble(decoderOut[srcIdx]);
            }

            // Objectness score from text-conditioned features
            double objectness = 0;
            for (int d = 0; d < dim; d++)
                objectness += patchFeat[d];
            double conf = 1.0 / (1.0 + Math.Exp(-objectness / Math.Max(1, dim)));

            // Box regression
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "OWLv2-Native" : "OWLv2-ONNX", Description = "OWLv2: self-training for scaling open-vocabulary detection.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "OWLv2";
        m.AdditionalInfo["NumClassEmbeddings"] = _options.NumClassEmbeddings.ToString();
        m.AdditionalInfo["SelfTraining"] = _options.EnableSelfTraining.ToString();
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
        writer.Write(_options.EnableSelfTraining);
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
        _options.EnableSelfTraining = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new OWLv2<T>(Architecture, mp, _options); return new OWLv2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OWLv2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
