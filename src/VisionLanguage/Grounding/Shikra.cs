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
/// Shikra: referential dialogue model with coordinate output for grounding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Shikra: Unleashing Multimodal LLM Referential Dialogue" (SenseTime, 2023)</item></list></para>
/// </remarks>
public class Shikra<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly ShikraOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Shikra(NeuralNetworkArchitecture<T> architecture, string modelPath, ShikraOptions? options = null) : base(architecture) { _options = options ?? new ShikraOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Shikra(NeuralNetworkArchitecture<T> architecture, ShikraOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ShikraOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxDetections => _options.MaxDetections;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Grounds text using Shikra's referential dialogue with numeric coordinate output.
    /// Per the paper (Chen et al., SenseTime 2023), Shikra is a multimodal LLM that
    /// treats spatial coordinates as plain-text numbers in the output vocabulary. Instead
    /// of special tokens or external heads, coordinates [x1, y1, x2, y2] are generated
    /// as normalized (0-1) decimal numbers directly in the text stream. The model:
    /// (1) encodes the image via CLIP ViT into visual tokens, (2) interleaves with
    /// instruction tokens, (3) the LLM autoregressively generates coordinate values
    /// as text tokens when asked "where is [object]". The coordinate tokens are
    /// parsed from the LLM output to extract bounding boxes. For grounding, the
    /// LLM generates box coordinates for each referred object.
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

        // Step 1: CLIP ViT visual encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        int visDim = visualFeatures.Length;

        // Step 2: Interleave visual tokens with text instruction
        var textTokens = TokenizeText(textQuery);
        int textLen = textTokens.Length;

        // Create multimodal sequence: [visual tokens] + [text tokens]
        int seqLen = visDim + textLen;
        var multimodalSeq = new Tensor<T>([visDim]);
        for (int d = 0; d < visDim; d++)
        {
            double vis = NumOps.ToDouble(visualFeatures[d]);
            // Text-conditioned modulation of visual features
            if (textLen > 0)
            {
                double textVal = NumOps.ToDouble(textTokens[d % textLen]);
                double gate = 1.0 / (1.0 + Math.Exp(-textVal / 100.0));
                vis = vis * (0.5 + gate);
            }
            multimodalSeq[d] = NumOps.FromDouble(vis);
        }

        // Step 3: LLM decoder to generate coordinate tokens
        var decoderOut = multimodalSeq;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;

        // Step 4: Parse coordinate tokens from LLM output
        // Shikra generates coordinates as groups of 4 values (x1,y1,x2,y2)
        // Each coordinate is sigmoid-bounded to [0,1]
        int coordsPerBox = 4;
        int numPotentialBoxes = outDim / coordsPerBox;
        int maxDet = Math.Min(numPotentialBoxes, _options.MaxDetections);
        int fieldsPerDet = 5;
        var rawDetections = new double[maxDet, fieldsPerDet];
        int validCount = 0;

        for (int b = 0; b < maxDet; b++)
        {
            int baseIdx = b * coordsPerBox;
            if (baseIdx + 3 >= outDim) break;

            // Read coordinate tokens and apply sigmoid normalization
            double rawX1 = NumOps.ToDouble(decoderOut[baseIdx]);
            double rawY1 = NumOps.ToDouble(decoderOut[baseIdx + 1]);
            double rawX2 = NumOps.ToDouble(decoderOut[baseIdx + 2]);
            double rawY2 = NumOps.ToDouble(decoderOut[baseIdx + 3]);

            double x1 = 1.0 / (1.0 + Math.Exp(-rawX1));
            double y1 = 1.0 / (1.0 + Math.Exp(-rawY1));
            double x2 = 1.0 / (1.0 + Math.Exp(-rawX2));
            double y2 = 1.0 / (1.0 + Math.Exp(-rawY2));

            // Ensure x2 > x1, y2 > y1
            if (x2 < x1) { double tmp = x1; x1 = x2; x2 = tmp; }
            if (y2 < y1) { double tmp = y1; y1 = y2; y2 = tmp; }

            // Confidence from coordinate token magnitudes
            double coordMag = Math.Abs(rawX1) + Math.Abs(rawY1) + Math.Abs(rawX2) + Math.Abs(rawY2);
            double conf = 1.0 / (1.0 + Math.Exp(-coordMag / 4.0 + 2.0));

            // Valid box check
            double boxArea = (x2 - x1) * (y2 - y1);
            if (conf >= confThreshold && boxArea > 0.001 && x2 > x1 && y2 > y1)
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
        // Shikra treats each category as a referential dialogue question
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Shikra-Native" : "Shikra-ONNX", Description = "Shikra: referential dialogue model with coordinate output for grounding.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Shikra";
        m.AdditionalInfo["CoordinateOutput"] = _options.EnableCoordinateOutput.ToString();
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
        writer.Write(_options.EnableCoordinateOutput);
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
        _options.EnableCoordinateOutput = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Shikra<T>(Architecture, mp, _options); return new Shikra<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Shikra<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
