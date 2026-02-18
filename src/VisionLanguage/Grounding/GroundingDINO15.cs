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
/// Grounding DINO 1.5: enhanced open-set detection with improved architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Grounding DINO 1.5: Advance the Edge of Open-Set Object Detection" (IDEA, 2024)</item></list></para>
/// </remarks>
public class GroundingDINO15<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly GroundingDINO15Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GroundingDINO15(NeuralNetworkArchitecture<T> architecture, string modelPath, GroundingDINO15Options? options = null) : base(architecture) { _options = options ?? new GroundingDINO15Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GroundingDINO15(NeuralNetworkArchitecture<T> architecture, GroundingDINO15Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GroundingDINO15Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds a text query using Grounding DINO 1.5's enhanced architecture.
    /// Per the paper (IDEA, 2024), 1.5 improves over the original by using
    /// a stronger backbone (Swin-L or ViT-H), multi-scale deformable attention
    /// in the feature enhancer for better handling of objects at different scales,
    /// and an improved text-visual alignment module with contrastive learning.
    /// The model also integrates EfficientSAM for segmentation-aware features.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        int numQueries = _options.NumQueryPositions;
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: Multi-scale visual feature extraction
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Step 2: Cross-modal feature enhancement via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedInput = visualFeatures.ConcatenateTensors(textTokens);

        // Step 3: Decoder with cross-modal queries
        var decoderOut = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;

        // Step 5: Object query decoding with bounding box prediction
        int maxDet = Math.Min(numQueries, _options.MaxDetections);
        int fieldsPerDet = 5;
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        for (int q = 0; q < maxDet; q++)
        {
            int blockSize = Math.Max(1, outDim / maxDet);
            int start = q * blockSize;
            int end = Math.Min(start + blockSize, outDim);

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

            cx = 1.0 / (1.0 + Math.Exp(-cx / Math.Max(1, span / 4)));
            cy = 1.0 / (1.0 + Math.Exp(-cy / Math.Max(1, span / 4)));
            w = 1.0 / (1.0 + Math.Exp(-w / Math.Max(1, span / 4)));
            h = 1.0 / (1.0 + Math.Exp(-h / Math.Max(1, span / 4)));
            conf = 1.0 / (1.0 + Math.Exp(-Math.Sqrt(conf / Math.Max(1, span)) + 1.0));

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

        // Step 6: NMS
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "GroundingDINO-1.5-Native" : "GroundingDINO-1.5-ONNX", Description = "Grounding DINO 1.5: enhanced open-set detection with improved architecture.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "GroundingDINO-1.5";
        m.AdditionalInfo["BackboneType"] = _options.BackboneType;
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
        writer.Write(_options.BackboneType ?? string.Empty);
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
        _options.BackboneType = reader.ReadString();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GroundingDINO15<T>(Architecture, mp, _options); return new GroundingDINO15<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GroundingDINO15<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
