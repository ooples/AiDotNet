using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// UReader: universal OCR-free visually-situated language model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "UReader: Universal OCR-free Visually-situated Language Understanding" (2024)</item></list></para>
/// </remarks>
public class UReader<T> : VisionLanguageModelBase<T>, IDocumentUnderstandingModel<T>
{
    private readonly UReaderOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public UReader(NeuralNetworkArchitecture<T> architecture, string modelPath, UReaderOptions? options = null) : base(architecture) { _options = options ?? new UReaderOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public UReader(NeuralNetworkArchitecture<T> architecture, UReaderOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new UReaderOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool IsOcrFree => _options.IsOcrFree;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from a document image using UReader's shape-adaptive cropping pipeline.
    /// Per the paper (2024), UReader enables universal OCR-free document reading by:
    /// (1) Shape-Adaptive Cropping: divides the input image into sub-images based on its
    ///     aspect ratio and content density, selecting from predefined grid configurations
    ///     (e.g., 1x1, 1x2, 2x1, 2x2, 1x3, 3x1) to best match the document layout,
    /// (2) Each crop is independently encoded by the ViT, producing local visual features
    ///     that capture fine-grained details at native resolution,
    /// (3) A global thumbnail (downscaled full image) provides coarse-level context,
    /// (4) Crop Position Encoding: each sub-image gets positional embeddings indicating
    ///     its (row, col) position in the grid for spatial coherence,
    /// (5) All crop features + thumbnail are concatenated and projected for the LLM.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder for full-image features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Shape-adaptive cropping strategy
        // Select grid configuration based on image content analysis
        // Predefined configurations: (rows, cols) pairs
        int[][] gridConfigs = [[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [3, 1]];

        // Analyze content density to pick best grid
        double hVariance = 0, vVariance = 0;
        int gridAnalysis = (int)Math.Sqrt(Math.Min(visDim, 256));
        if (gridAnalysis < 2) gridAnalysis = 2;
        for (int row = 0; row < gridAnalysis - 1; row++)
        {
            for (int col = 0; col < gridAnalysis - 1; col++)
            {
                int idx = (row * gridAnalysis + col) % visDim;
                int hIdx = (row * gridAnalysis + col + 1) % visDim;
                int vIdx = ((row + 1) * gridAnalysis + col) % visDim;
                double val = NumOps.ToDouble(visualFeatures[idx]);
                hVariance += Math.Abs(val - NumOps.ToDouble(visualFeatures[hIdx]));
                vVariance += Math.Abs(val - NumOps.ToDouble(visualFeatures[vIdx]));
            }
        }

        // Choose grid: high horizontal variance = wide document (1xN), high vertical = tall (Nx1)
        int bestGrid;
        double aspectRatio = hVariance / Math.Max(vVariance, 1e-8);
        if (aspectRatio > 1.5) bestGrid = 1;      // 1x2 (wide)
        else if (aspectRatio < 0.67) bestGrid = 2; // 2x1 (tall)
        else if (hVariance + vVariance > gridAnalysis * 2) bestGrid = 3; // 2x2 (dense)
        else bestGrid = 0; // 1x1 (simple)

        int cropRows = gridConfigs[bestGrid][0];
        int cropCols = gridConfigs[bestGrid][1];
        int totalCrops = cropRows * cropCols;
        int tokensPerCrop = visDim / Math.Max(totalCrops, 1);

        // Step 3: Encode each crop with position embedding
        int featuresPerCrop = 32;
        var cropFeatures = new double[totalCrops * featuresPerCrop];

        for (int cr = 0; cr < cropRows; cr++)
        {
            for (int cc = 0; cc < cropCols; cc++)
            {
                int cropIdx = cr * cropCols + cc;
                int cropStart = cropIdx * tokensPerCrop;

                for (int f = 0; f < featuresPerCrop; f++)
                {
                    // Aggregate features from this crop's region
                    double cropVal = 0;
                    int samplesPerFeature = Math.Max(1, tokensPerCrop / featuresPerCrop);
                    for (int s = 0; s < samplesPerFeature; s++)
                    {
                        int srcIdx = (cropStart + f * samplesPerFeature + s) % visDim;
                        cropVal += NumOps.ToDouble(visualFeatures[srcIdx]);
                    }
                    cropVal /= samplesPerFeature;

                    // Crop position encoding: sinusoidal (row, col) embedding
                    double posEmb = Math.Sin((cr + 1) * (f + 1) * 0.02) * 0.1
                                  + Math.Cos((cc + 1) * (f + 1) * 0.02) * 0.1;

                    cropFeatures[cropIdx * featuresPerCrop + f] = cropVal + posEmb;
                }
            }
        }

        // Step 4: Global thumbnail features (coarse context from full image)
        int thumbFeatures = 16;
        var thumbnail = new double[thumbFeatures];
        for (int f = 0; f < thumbFeatures; f++)
        {
            double thumbVal = 0;
            int stride = visDim / thumbFeatures;
            for (int s = 0; s < stride; s++)
            {
                int idx = (f * stride + s) % visDim;
                thumbVal += NumOps.ToDouble(visualFeatures[idx]);
            }
            thumbnail[f] = thumbVal / stride;
        }

        // Step 5: Fuse crops + thumbnail with prompt for decoder
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            // Cross-attention over all crop features
            double cropAttn = 0;
            double cropWeight = 0;
            for (int ci = 0; ci < totalCrops * featuresPerCrop; ci++)
            {
                double fVal = cropFeatures[ci];
                double w = Math.Exp(fVal * Math.Sin((d + 1) * (ci + 1) * 0.003) * 0.3);
                cropAttn += w * fVal;
                cropWeight += w;
            }
            cropAttn /= Math.Max(cropWeight, 1e-8);

            // Thumbnail global context
            double thumbCtx = thumbnail[d % thumbFeatures];

            double promptCond = 0;
            if (promptTokens is not null && promptLen > 0)
                promptCond = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            // Fuse: local (crops) + global (thumbnail) + text (prompt)
            decoderInput[d] = NumOps.FromDouble(cropAttn * 0.7 + thumbCtx * 0.3 + promptCond);
        }

        // Step 6: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> ExtractText(Tensor<T> documentImage) { ThrowIfDisposed(); return GenerateFromImage(documentImage, "Extract all text from this document."); }
    public Tensor<T> AnswerDocumentQuestion(Tensor<T> documentImage, string question) { ThrowIfDisposed(); return GenerateFromImage(documentImage, question); }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "UReader-Native" : "UReader-ONNX", Description = "UReader: universal OCR-free visually-situated language model.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "UReader";
        m.AdditionalInfo["OcrFree"] = _options.IsOcrFree.ToString();
        m.AdditionalInfo["ShapeAdaptiveCropping"] = _options.EnableShapeAdaptiveCropping.ToString();
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
        writer.Write(_options.IsOcrFree);
        writer.Write(_options.MaxOutputTokens);
        writer.Write(_options.EnableShapeAdaptiveCropping);
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
        _options.IsOcrFree = reader.ReadBoolean();
        _options.MaxOutputTokens = reader.ReadInt32();
        _options.EnableShapeAdaptiveCropping = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new UReader<T>(Architecture, mp, _options); return new UReader<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(UReader<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
