using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DiT: Self-supervised Pre-training for Document Image Transformer", "https://doi.org/10.48550/arXiv.2203.02378", Year = 2022, Authors = "Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, Beta1 = 0.9, Beta2 = 0.999,
                WeightDecay = 0.05, ReferenceBatchSize = 2048, WarmupSteps = 10000,
                Source = "Li et al. 2022, Sec. 4.1: DiT is pre-trained for 500K steps at a batch size "
                        + "of 2,048, a learning rate of 1e-3, 10K warmup steps and a weight decay of "
                        + "0.05, with Adam betas 0.9 and 0.999. The 5e-4 elsewhere in the paper trains "
                        + "the tokenizer rather than the model.")]
public partial class DiT<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentClassifier<T>
{
    private readonly DiTOptions _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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

    // The CLS token and the learned position table used to be model fields here. They are now the
    // PrependCLSTokenLayer and LearnedPositionalEmbeddingLayer in the stack, where the forward pass
    // reads them -- see LayerHelper.CreateDefaultDiTLayers.

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
        DiTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new DiTOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        // Validated after the path checks so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        _numClasses = _options.NumClasses;
        _hiddenDim = _options.HiddenDim;
        _numLayers = _options.NumLayers;
        _numHeads = _options.NumHeads;
        _patchSize = _options.PatchSize;
        _modelSize = _options.ModelSize;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = _options.ImageSize;

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
        DiTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new DiTOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _numClasses = _options.NumClasses;
        _hiddenDim = _options.HiddenDim;
        _numLayers = _options.NumLayers;
        _numHeads = _options.NumHeads;
        _patchSize = _options.PatchSize;
        _modelSize = _options.ModelSize;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = _options.ImageSize;

        InitializeLayers();
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
        int numPatches = output.Shape[0];
        if (numPatches == 0)
            return new List<LayoutRegion<T>>();

        // DiT uses patch-level predictions for layout: a patch is a region when its best class is not the
        // background (class 0), is positive and reaches the threshold.
        int patchesPerRow = ImageSize / _patchSize;
        int classCount = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;
        var (maxClass, maxConf) = DocumentTensorOps.RowMax(Engine, output, classCount);
        var keep = Engine.TensorMultiply(
            Engine.TensorMultiply(Engine.TensorGreaterThan(maxConf, NumOps.Zero),
                DocumentTensorOps.AtLeast(Engine, maxConf, NumOps.FromDouble(threshold))),
            Engine.TensorGreaterThan(DocumentTensorOps.IntTensor<T>(maxClass), NumOps.Zero));
        int[] patches = DocumentTensorOps.Hits(Engine, keep);
        T[] confidence = DocumentTensorOps.Pick(Engine, maxConf, patches);
        T[] boxes = DocumentTensorOps.ToT<T>(patches.SelectMany(i => new[]
        {
            (i % patchesPerRow) * _patchSize, (i / patchesPerRow) * _patchSize,
            ((i % patchesPerRow) + 1) * _patchSize, ((i / patchesPerRow) + 1) * _patchSize
        }).ToArray());

        return patches.Select((i, k) => new LayoutRegion<T>
        {
            ElementType = (LayoutElementType)Math.Min(maxClass[i], (int)LayoutElementType.Other),
            Confidence = confidence[k],
            Index = i,
            BoundingBox = new Vector<T>(new[] { boxes[4 * k], boxes[(4 * k) + 1], boxes[(4 * k) + 2], boxes[(4 * k) + 3] })
        }).ToList();
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
        int numClasses = Math.Min(output.Shape.Length > 1 ? output.Shape[1] : _numClasses, AvailableCategories.Count);

        // Softmax and top-k
        var (order, probabilities) = DocumentTensorOps.RankedSoftmax(Engine, DocumentTensorOps.RowHead(Engine, output, 0, numClasses));
        var predictions = order.Take(topK).Select((idx, k) => (Category: AvailableCategories[idx], Score: probabilities[k])).ToList();

        return new DocumentClassificationResult<T>
        {
            PredictedCategory = predictions.First().Category,
            Confidence = predictions.First().Score,
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
    /// Applies DiT's preprocessing: BEiT's 0.5 / 0.5 normalization.
    /// </summary>
    /// <remarks>
    /// DiT (Li et al. 2022) is pretrained BEiT-style, and its released image processor normalizes every
    /// channel with mean 0.5 and std 0.5 (mapping [0, 1] to [-1, 1]), not with ImageNet statistics.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
        => DocumentTensorOps.NormalizeChannels(Engine, EnsureBatchDimension(rawImage),
            DocumentTensorOps.HalfStatistics, DocumentTensorOps.HalfStatistics);

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
            ModelDataProvider = () => SafeSerialize()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    /// <remarks>
    /// Replaces a hand-written throw that used to sit inside UpdateParameters. The base checks this
    /// on every mutating entry point rather than the one member the throw happened to guard, and
    /// reading -- ParameterCount and GetParameters -- stays available either way.
    /// </remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

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
