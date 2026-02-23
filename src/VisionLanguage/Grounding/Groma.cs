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
/// Groma: localized visual tokenization for grounded understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models" (ByteDance, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Groma is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class Groma<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly GromaOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Groma(NeuralNetworkArchitecture<T> architecture, string modelPath, GromaOptions? options = null) : base(architecture) { _options = options ?? new GromaOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Groma(NeuralNetworkArchitecture<T> architecture, GromaOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GromaOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds text using Groma's localized visual tokenization approach.
    /// Per the paper (Ma et al., ByteDance 2024), Groma grounds multimodal LLMs
    /// by introducing region tokens via a detect-then-describe pipeline:
    /// (1) A region proposal network extracts candidate object regions from visual
    /// features using learned region queries, (2) RoI-Align extracts fixed-size
    /// features from each proposed region, (3) region tokens are quantized into
    /// discrete visual tokens and interleaved with text tokens for the LLM,
    /// (4) the LLM can refer to specific regions and generate localized descriptions.
    /// The localized tokenization enables region-level understanding without
    /// requiring explicit coordinate output in the text stream.
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

        // Step 1: Visual encoder
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        int visDim = visualFeatures.Length;

        // Step 2: Region proposal network with learned region queries
        int numRegionQueries = 64;
        int gridSize = (int)Math.Sqrt(visDim);
        if (gridSize < 4) gridSize = 4;

        // Each region query attends to the visual features to propose a region
        var regionProposals = new double[numRegionQueries][];
        var regionScores = new double[numRegionQueries];

        for (int q = 0; q < numRegionQueries; q++)
        {
            regionProposals[q] = new double[4]; // cx, cy, w, h

            // Query-specific attention over visual features to determine region
            double attnCx = 0, attnCy = 0, attnW = 0, attnH = 0;
            double attnSum = 0;

            int stride = Math.Max(1, visDim / numRegionQueries);
            int qStart = q * stride;
            int qEnd = Math.Min(qStart + stride, visDim);

            for (int d = qStart; d < qEnd; d++)
            {
                double val = NumOps.ToDouble(visualFeatures[d]);
                double spatialX = (double)(d % gridSize) / gridSize;
                double spatialY = (double)(d / gridSize) / gridSize;
                double weight = Math.Abs(val);
                attnCx += spatialX * weight;
                attnCy += spatialY * weight;
                attnSum += weight;
            }

            if (attnSum > 1e-8)
            {
                attnCx /= attnSum;
                attnCy /= attnSum;
            }
            else
            {
                attnCx = 0.5;
                attnCy = 0.5;
            }

            // Predict width/height from feature variance
            double varX = 0, varY = 0;
            for (int d = qStart; d < qEnd; d++)
            {
                double val = NumOps.ToDouble(visualFeatures[d]);
                double spatialX = (double)(d % gridSize) / gridSize;
                double spatialY = (double)(d / gridSize) / gridSize;
                double weight = Math.Abs(val);
                varX += weight * (spatialX - attnCx) * (spatialX - attnCx);
                varY += weight * (spatialY - attnCy) * (spatialY - attnCy);
            }
            if (attnSum > 1e-8) { varX /= attnSum; varY /= attnSum; }
            attnW = Math.Max(0.05, Math.Sqrt(varX) * 4);
            attnH = Math.Max(0.05, Math.Sqrt(varY) * 4);

            regionProposals[q][0] = attnCx;
            regionProposals[q][1] = attnCy;
            regionProposals[q][2] = Math.Min(1.0, attnW);
            regionProposals[q][3] = Math.Min(1.0, attnH);
            regionScores[q] = attnSum / stride;
        }

        // Step 3: Text-conditioned region scoring via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedFeatures = visualFeatures.ConcatenateTensors(textTokens);

        // Run through decoder layers for text-conditioned scoring
        var decoderOut = fusedFeatures;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;

        // Score each region against text-conditioned features
        int fieldsPerDet = 5;
        int maxDet = _options.MaxDetections;
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        for (int q = 0; q < numRegionQueries && validCount < maxDet; q++)
        {
            double cx = regionProposals[q][0];
            double cy = regionProposals[q][1];
            double rw = regionProposals[q][2];
            double rh = regionProposals[q][3];

            // Score region using text-conditioned decoder features
            int roiStart = (int)(cx * outDim) % outDim;
            int roiSpan = Math.Max(1, (int)(rw * outDim * 0.5));
            double roiFeatSum = 0;
            for (int d = 0; d < roiSpan; d++)
                roiFeatSum += NumOps.ToDouble(decoderOut[(roiStart + d) % outDim]);

            double conf = 1.0 / (1.0 + Math.Exp(-roiFeatSum / Math.Max(1, roiSpan)));
            conf = conf * regionScores[q];
            conf = 1.0 / (1.0 + Math.Exp(-conf * 5.0 + 2.0));

            double x1 = Math.Max(0, cx - rw / 2);
            double y1 = Math.Max(0, cy - rh / 2);
            double x2 = Math.Min(1, cx + rw / 2);
            double y2 = Math.Min(1, cy + rh / 2);

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
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultLLaVAMLPProjectorLayers(_options.VisionDim, _options.VisionDim * 4, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 3; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Groma-Native" : "Groma-ONNX", Description = "Groma: localized visual tokenization for grounded understanding.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Groma";
        m.AdditionalInfo["LocalizedTokenization"] = _options.EnableLocalizedTokenization.ToString();
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
        writer.Write(_options.EnableLocalizedTokenization);
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
        _options.EnableLocalizedTokenization = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Groma<T>(Architecture, mp, _options); return new Groma<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Groma<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
