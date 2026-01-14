using AiDotNet.Document.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.Document.LayoutAware;

/// <summary>
/// DiT (Document Image Transformer) for document image understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DiT applies self-supervised pre-training on large-scale document images using
/// a Vision Transformer (ViT) backbone, enabling strong document layout analysis
/// without requiring OCR annotations.
/// </para>
/// <para>
/// <b>For Beginners:</b> DiT learns document understanding from images alone:
/// 1. Pre-trains on 42 million document images
/// 2. Uses masked image modeling (predicts missing patches)
/// 3. Learns document-specific visual patterns
///
/// Key features:
/// - Pure vision approach (no OCR needed for pre-training)
/// - ViT-base/large architectures
/// - State-of-the-art on document classification
/// - Strong layout analysis performance
///
/// Example usage:
/// <code>
/// var model = new DiT&lt;float&gt;(architecture);
/// var result = model.DetectLayout(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DiT: Self-supervised Pre-training for Document Image Transformer" (ACM MM 2022)
/// https://arxiv.org/abs/2203.02378
/// </para>
/// </remarks>
public class DiT<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentClassifier<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _numClasses;
    private readonly string _modelSize;

    // Native mode layers
    private readonly List<ILayer<T>> _patchEmbeddingLayers = [];
    private readonly List<ILayer<T>> _transformerLayers = [];
    private readonly List<ILayer<T>> _classificationHead = [];

    // Learnable embeddings
    private Tensor<T>? _positionEmbeddings;
    private Tensor<T>? _clsToken;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the patch size for the ViT backbone.
    /// </summary>
    public int PatchSize => _patchSize;

    /// <summary>
    /// Gets the model size variant (base/large).
    /// </summary>
    public string ModelSize => _modelSize;

    /// <inheritdoc/>
    public IReadOnlyList<LayoutElementType> SupportedElementTypes { get; } =
    [
        LayoutElementType.Text,
        LayoutElementType.Title,
        LayoutElementType.List,
        LayoutElementType.Table,
        LayoutElementType.Figure,
        LayoutElementType.Caption,
        LayoutElementType.Header,
        LayoutElementType.Footer
    ];

    /// <inheritdoc/>
    public IReadOnlyList<string> AvailableCategories { get; } =
    [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "scientific_publication", "specification",
        "file_folder", "news_article", "budget", "invoice",
        "presentation", "questionnaire", "resume", "memo"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DiT model using a pre-trained ONNX model for inference.
    /// </summary>
    public DiT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 16,
        int imageSize = 224,
        int patchSize = 16,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        string modelSize = "base",
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _modelSize = modelSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DiT model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (DiT-Base from ACM MM 2022):</b>
    /// - Architecture: ViT-Base
    /// - Hidden dimension: 768
    /// - Layers: 12, Heads: 12
    /// - Patch size: 16×16
    /// - Image size: 224×224
    /// - Pre-training: Masked image modeling on IIT-CDIP
    /// </para>
    /// </remarks>
    public DiT(
        NeuralNetworkArchitecture<T> architecture,
        int numClasses = 16,
        int imageSize = 224,
        int patchSize = 16,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        string modelSize = "base",
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _modelSize = modelSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

        InitializeLayers();
        InitializeEmbeddings();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultDiTLayers(
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            patchSize: _patchSize,
            imageSize: ImageSize,
            numClasses: _numClasses));
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        int numPatches = (ImageSize / _patchSize) * (ImageSize / _patchSize);

        _positionEmbeddings = Tensor<T>.CreateDefault([numPatches + 1, _hiddenDim], NumOps.Zero);
        _clsToken = Tensor<T>.CreateDefault([1, _hiddenDim], NumOps.Zero);

        InitializeWithSmallRandomValues(_positionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_clsToken, random, 0.02);
    }

    private void InitializeWithSmallRandomValues(Tensor<T> tensor, Random random, double stdDev)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            tensor.Data.Span[i] = NumOps.FromDouble(randStdNormal * stdDev);
        }
    }

    #endregion

    #region ILayoutDetector Implementation

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage)
    {
        return DetectLayout(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var regions = ParseLayoutOutput(output, confidenceThreshold);

        return new DocumentLayoutResult<T>
        {
            Regions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private List<LayoutRegion<T>> ParseLayoutOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<LayoutRegion<T>>();
        int numPatches = output.Shape[0];

        // DiT uses patch-level predictions for layout
        int patchesPerRow = ImageSize / _patchSize;
        for (int i = 0; i < numPatches; i++)
        {
            double maxConf = 0;
            int maxClass = 0;
            int classCount = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;

            for (int c = 0; c < classCount; c++)
            {
                double conf = NumOps.ToDouble(output[i, c]);
                if (conf > maxConf) { maxConf = conf; maxClass = c; }
            }

            if (maxConf >= threshold && maxClass > 0)
            {
                int patchX = i % patchesPerRow;
                int patchY = i / patchesPerRow;

                regions.Add(new LayoutRegion<T>
                {
                    ElementType = (LayoutElementType)Math.Min(maxClass, (int)LayoutElementType.Other),
                    Confidence = NumOps.FromDouble(maxConf),
                    ConfidenceValue = maxConf,
                    Index = i,
                    BoundingBox = new Vector<T>([
                        NumOps.FromDouble(patchX * _patchSize),
                        NumOps.FromDouble(patchY * _patchSize),
                        NumOps.FromDouble((patchX + 1) * _patchSize),
                        NumOps.FromDouble((patchY + 1) * _patchSize)
                    ])
                });
            }
        }

        return regions;
    }

    #endregion

    #region IDocumentClassifier Implementation

    /// <inheritdoc/>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage)
    {
        return ClassifyDocument(documentImage, 5);
    }

    /// <inheritdoc/>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage, int topK)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        // Get classification logits from CLS token
        var predictions = new List<(string Category, double Confidence)>();
        int numClasses = Math.Min(output.Shape.Length > 1 ? output.Shape[1] : _numClasses, AvailableCategories.Count);

        var scores = new List<(int Index, double Score)>();
        for (int c = 0; c < numClasses; c++)
        {
            double score = NumOps.ToDouble(output[0, c]);
            scores.Add((c, score));
        }

        // Softmax and top-k
        double maxScore = scores.Max(s => s.Score);
        double sumExp = scores.Sum(s => Math.Exp(s.Score - maxScore));

        foreach (var (idx, score) in scores.OrderByDescending(s => s.Score).Take(topK))
        {
            double prob = Math.Exp(score - maxScore) / sumExp;
            predictions.Add((AvailableCategories[idx], prob));
        }

        return new DocumentClassificationResult<T>
        {
            PredictedCategory = predictions.First().Category,
            Confidence = NumOps.FromDouble(predictions.First().Confidence),
            ConfidenceValue = predictions.First().Confidence,
            TopPredictions = predictions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public IEnumerable<DocumentClassificationResult<T>> ClassifyDocumentBatch(IEnumerable<Tensor<T>> documentImages)
    {
        foreach (var image in documentImages)
            yield return ClassifyDocument(image);
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("DiT Model Summary");
        sb.AppendLine("=================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: ViT-{_modelSize}");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Patch Size: {_patchSize}×{_patchSize}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Number of Patches: {(ImageSize / _patchSize) * (ImageSize / _patchSize)}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"OCR-Free: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies DiT's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// DiT (Document Image Transformer) uses ImageNet normalization with
    /// mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] (Microsoft paper).
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);
        double[] means = [0.485, 0.456, 0.406];
        double[] stds = [0.229, 0.224, 0.225];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double mean = c < means.Length ? means[c] : 0.5;
                double std = c < stds.Length ? stds[c] : 0.5;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        normalized.Data.Span[idx] = NumOps.FromDouble((NumOps.ToDouble(image.Data.Span[idx]) - mean) / std);
                    }
                }
            }
        }
        return normalized;
    }

    /// <summary>
    /// Applies DiT's industry-standard postprocessing: pass-through (classification outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DiT",
            ModelType = ModelType.NeuralNetwork,
            Description = "DiT for document image understanding (ACM MM 2022)",
            FeatureCount = _hiddenDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "patch_size", _patchSize },
                { "image_size", ImageSize },
                { "num_classes", _numClasses },
                { "model_size", _modelSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_patchSize);
        writer.Write(ImageSize);
        writer.Write(_numClasses);
        writer.Write(_modelSize);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        string modelSize = reader.ReadString();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DiT<T>(Architecture, _numClasses, ImageSize, _patchSize, _hiddenDim, _numLayers, _numHeads, _modelSize);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var gradient = Tensor<T>.FromVector(
            LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector()));

        for (int i = Layers.Count - 1; i >= 0; i--)
            gradient = Layers[i].Backward(gradient);

        UpdateParameters(CollectGradients());
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        var currentParams = GetParameters();
        T lr = NumOps.FromDouble(0.00005);
        for (int i = 0; i < currentParams.Length; i++)
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(lr, gradients[i]));
        SetParameters(currentParams);
    }

    private Vector<T> CollectGradients()
    {
        var grads = new List<T>();
        foreach (var layer in Layers)
            grads.AddRange(layer.GetParameterGradients());
        return new Vector<T>([.. grads]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
