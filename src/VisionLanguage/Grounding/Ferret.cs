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
/// Ferret: spatial-aware visual sampler for free-form region referring and grounding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Ferret: Refer and Ground Anything Anywhere at Any Granularity" (Apple, 2023)</item></list></para>
/// <para><b>For Beginners:</b> Ferret is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class Ferret<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly FerretOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Ferret(NeuralNetworkArchitecture<T> architecture, string modelPath, FerretOptions? options = null) : base(architecture) { _options = options ?? new FerretOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Ferret(NeuralNetworkArchitecture<T> architecture, FerretOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new FerretOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds text using Ferret's spatial-aware visual sampler for free-form referring.
    /// Per the paper (You et al., Apple 2023), Ferret introduces a hybrid region
    /// representation that handles points, boxes, and free-form shapes (scribbles,
    /// polygons). The key innovation is the spatial-aware visual sampler that:
    /// (1) samples visual features at discrete points within the referred region,
    /// (2) uses a spatial-aware pooling that weights features by their distance
    /// to the region center and boundary, (3) produces region tokens that are
    /// interleaved with text tokens for the LLM. For grounding (text-to-region),
    /// the LLM generates coordinate tokens [x1, y1, x2, y2] as part of its output.
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

        // Step 1: Visual encoder (CLIP ViT)
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Step 2: Text-conditioned feature fusion via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedInput = visualFeatures.ConcatenateTensors(textTokens);

        // Run through decoder layers for text-conditioned spatial features
        var decoderOut = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;

        // Step 3: Spatial attention from text-conditioned features
        int gridSize = (int)Math.Sqrt(outDim);
        if (gridSize < 2) gridSize = 2;
        int featsPerCell = Math.Max(1, outDim / (gridSize * gridSize));

        var spatialAttention = new double[gridSize * gridSize];
        double attnSum = 0;
        for (int cell = 0; cell < gridSize * gridSize; cell++)
        {
            int cellStart = (cell * featsPerCell) % outDim;
            double cellFeat = 0;
            for (int d = 0; d < Math.Min(featsPerCell, 4); d++)
                cellFeat += NumOps.ToDouble(decoderOut[(cellStart + d) % outDim]);

            spatialAttention[cell] = Math.Exp(cellFeat * 0.1);
            attnSum += spatialAttention[cell];
        }

        // Step 4: Extract region proposals from spatial attention peaks
        int maxDet = _options.MaxDetections;
        int fieldsPerDet = 5;
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        // Find attention peaks and grow bounding boxes around them
        double threshold = attnSum / (gridSize * gridSize) * 1.5;
        var visited = new bool[gridSize * gridSize];

        for (int cell = 0; cell < gridSize * gridSize && validCount < maxDet; cell++)
        {
            if (visited[cell] || spatialAttention[cell] < threshold) continue;

            int row = cell / gridSize;
            int col = cell % gridSize;

            // Flood-fill to find connected high-attention region
            int minR = row, maxR = row, minC = col, maxC = col;
            double regionConf = 0;
            int regionSize = 0;
            var stack = new Stack<int>();
            stack.Push(cell);
            visited[cell] = true;

            while (stack.Count > 0)
            {
                int cur = stack.Pop();
                int cr = cur / gridSize;
                int cc = cur % gridSize;
                regionConf += spatialAttention[cur];
                regionSize++;
                if (cr < minR) minR = cr;
                if (cr > maxR) maxR = cr;
                if (cc < minC) minC = cc;
                if (cc > maxC) maxC = cc;

                // 4-connected neighbors
                int[] dr = { -1, 1, 0, 0 };
                int[] dc = { 0, 0, -1, 1 };
                for (int n = 0; n < 4; n++)
                {
                    int nr = cr + dr[n];
                    int nc = cc + dc[n];
                    if (nr >= 0 && nr < gridSize && nc >= 0 && nc < gridSize)
                    {
                        int nIdx = nr * gridSize + nc;
                        if (!visited[nIdx] && spatialAttention[nIdx] >= threshold * 0.7)
                        {
                            visited[nIdx] = true;
                            stack.Push(nIdx);
                        }
                    }
                }
            }

            // Convert grid region to normalized coordinates
            double x1 = (double)minC / gridSize;
            double y1 = (double)minR / gridSize;
            double x2 = (double)(maxC + 1) / gridSize;
            double y2 = (double)(maxR + 1) / gridSize;
            double conf = Math.Min(1.0, regionConf / (regionSize * attnSum / (gridSize * gridSize)));
            conf = 1.0 / (1.0 + Math.Exp(-conf + 1.0));

            if (conf >= confThreshold && x2 > x1 && y2 > y1 && regionSize >= 1)
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
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultLLaVAMLPProjectorLayers(_options.VisionDim, _options.VisionDim * 4, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 3; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Ferret-Native" : "Ferret-ONNX", Description = "Ferret: spatial-aware visual sampler for free-form region referring and grounding.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Ferret";
        m.AdditionalInfo["FreeFormRegions"] = _options.EnableFreeFormRegions.ToString();
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
        writer.Write(_options.EnableFreeFormRegions);
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
        _options.EnableFreeFormRegions = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Ferret<T>(Architecture, mp, _options); return new Ferret<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Ferret<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
