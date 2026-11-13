using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.LayoutAware;
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

namespace AiDotNet.Document.Analysis.PageSegmentation;

/// <summary>
/// DocBank model for document page segmentation and layout analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DocBank is a benchmark and model for document layout analysis that can segment document
/// pages into semantic regions including text, titles, figures, tables, and captions.
/// It combines visual features with optional text features for robust segmentation.
/// </para>
/// <para>
/// <b>For Beginners:</b> DocBank divides document pages into different regions:
/// - Paragraphs: Regular text content
/// - Titles: Document headings and titles
/// - Figures: Images and diagrams
/// - Tables: Tabular data regions
/// - Captions: Text describing figures/tables
/// - Lists: Bulleted or numbered lists
/// - Equations: Mathematical formulas
/// - And more...
///
/// Example usage:
/// <code>
/// var docbank = new DocBank&lt;float&gt;(architecture);
/// var result = docbank.SegmentPage(documentImage);
/// foreach (var region in result.Regions)
/// {
///     Console.WriteLine($"Found {region.RegionType} at {region.BoundingBox}");
/// }
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DocBank: A Benchmark Dataset for Document Layout Analysis" (COLING 2020)
/// https://arxiv.org/abs/2006.01038
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DocBank: A Benchmark for Document Layout Analysis", "https://doi.org/10.48550/arXiv.2006.01038", Year = 2020, Authors = "Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, Ming Zhou")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5,
                Source = "Li et al. 2020, Sec. 4: the model is optimized with AdamW at an initial "
                        + "learning rate of 5e-5.")]
public partial class DocBank<T> : DocumentNeuralNetworkBase<T>, IPageSegmenter<T>
{
    private readonly DocBankOptions _options;

    /// <summary>
    /// Builds the optimizer used only when the paper recipe cannot be resolved.
    /// </summary>
    /// <returns>An Adam optimizer at <see cref="DocBankOptions.LearningRate"/>, or Adam's 1e-3.</returns>
    /// <remarks>
    /// <para>
    /// DocBank declares its recipe (AdamW at 5e-5), so <see cref="PaperOptimizerFactory"/> builds the
    /// training optimizer and this fallback is not reached; it exists so a recipe that resolves to
    /// nothing still leaves the model trainable at the configured rate.
    /// </para>
    /// </remarks>
    private AdamOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate ?? 1e-3
            });

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    private readonly int _backboneChannels;
    private readonly int _numClasses;
    private readonly int _hiddenDim;
    private readonly bool _useTextFeatures;

    // Native mode layers
    private readonly List<ILayer<T>> _backboneLayers = [];
    private readonly List<ILayer<T>> _fpnLayers = [];
    private readonly List<ILayer<T>> _segmentationHead = [];

    // Class labels for DocBank
    private static readonly DocumentRegionType[] DocBankClasses =
    [
        DocumentRegionType.Other,       // 0: background
        DocumentRegionType.Paragraph,   // 1: paragraph
        DocumentRegionType.Title,       // 2: title
        DocumentRegionType.Equation,    // 3: equation
        DocumentRegionType.Table,       // 4: table
        DocumentRegionType.Figure,      // 5: figure
        DocumentRegionType.Caption,     // 6: caption
        DocumentRegionType.List,        // 7: list
        DocumentRegionType.Abstract,    // 8: abstract
        DocumentRegionType.Author,      // 9: author
        DocumentRegionType.Footer,      // 10: footer
        DocumentRegionType.Reference,   // 11: reference
        DocumentRegionType.Section      // 12: section
    ];

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => _useTextFeatures;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public IReadOnlyList<DocumentRegionType> SupportedRegionTypes => DocBankClasses;

    /// <inheritdoc/>
    public bool SupportsInstanceSegmentation => true;

    /// <summary>
    /// Gets the number of segmentation classes.
    /// </summary>
    public int NumClasses => _numClasses;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DocBank model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="imageSize">Expected input image size (default: 1024).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 256).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 13 for DocBank).</param>
    /// <param name="useTextFeatures">Whether to use text features (default: false for image-only).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <exception cref="ArgumentNullException">Thrown if onnxModelPath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX model file doesn't exist.</exception>
    public DocBank(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DocBankOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new DocBankOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));

        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model file not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _backboneChannels = _options.BackboneChannels;
        _numClasses = _options.NumClasses;
        // Previously the literal 256 here while the native constructor took it as a parameter,
        // so the two constructors agreed only by coincidence.
        _hiddenDim = _options.HiddenDim;
        _useTextFeatures = _options.UseTextFeatures;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(
        this, learningRateOverride: _options.LearningRate)
    ?? CreateDefaultOptimizer();

        ImageSize = _options.ImageSize;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DocBank model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="imageSize">Expected input image size (default: 1024).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 256).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 13 for DocBank).</param>
    /// <param name="hiddenDim">Hidden dimension for segmentation head (default: 256).</param>
    /// <param name="useTextFeatures">Whether to use text features (default: false for image-only).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (from COLING 2020 paper):</b>
    /// - Backbone: ResNet-101 with FPN
    /// - Image size: 1024×1024
    /// - Classes: 13 (paragraph, title, figure, table, etc.)
    /// - Can optionally incorporate text features from BERT
    /// </para>
    /// </remarks>
    public DocBank(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DocBankOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new DocBankOptions();
        Options = _options;

        _useNativeMode = true;
        _backboneChannels = _options.BackboneChannels;
        _numClasses = _options.NumClasses;
        _hiddenDim = _options.HiddenDim;
        _useTextFeatures = _options.UseTextFeatures;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(
        this, learningRateOverride: _options.LearningRate)
    ?? CreateDefaultOptimizer();

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

        // Check if user provided custom layers
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        // Use LayerHelper to create default DocBank layers
        Layers.AddRange(LayerHelper<T>.CreateDefaultDocBankLayers(
            imageSize: ImageSize,
            backboneChannels: _backboneChannels,
            numClasses: _numClasses,
            hiddenDim: _hiddenDim));
    }

    #endregion

    #region IPageSegmenter Implementation

    /// <inheritdoc/>
    public PageSegmentationResult<T> SegmentPage(Tensor<T> documentImage)
    {
        return SegmentPage(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public PageSegmentationResult<T> SegmentPage(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);

        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode
            ? Forward(preprocessed)
            : RunOnnxInference(preprocessed);

        var result = ParseSegmentationOutput(output, confidenceThreshold);

        return new PageSegmentationResult<T>
        {
            Regions = result.Regions,
            SegmentationMask = result.SegmentationMask,
            ClassProbabilities = result.ClassProbabilities,
            ReadingOrder = ComputeReadingOrder(result.Regions),
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public Tensor<T> GetSegmentationMask(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode
            ? Forward(preprocessed)
            : RunOnnxInference(preprocessed);

        return ExtractSegmentationMask(output);
    }

    internal PageSegmentationResult<T> ParseSegmentationOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<DocumentRegion<T>>();

        // The head emits per-class logits [batch, numClasses, height, width]; the class probabilities are their
        // softmax over the class axis, and a region's confidence is its mean winning-class probability.
        var (probabilities, height, width) = FirstImageProbabilities(output);
        var mask = ExtractSegmentationMask(output);
        int[] classes = mask.Cast<int>().ToArray();
        var winning = Engine.ReduceMax(probabilities, new[] { 0 }, keepDims: false);

        // 4-connected components of each foreground class.
        var visited = new bool[height * width];
        var components = new List<(int Class, int[] Pixels, int MinH, int MinW, int MaxH, int MaxW)>();
        var queue = new Queue<int>();
        for (int start = 0; start < classes.Length; start++)
        {
            int classIdx = classes[start];
            if (classIdx <= 0 || visited[start]) continue; // Skip background

            var pixels = new List<int>();
            int minH = start / width, maxH = minH, minW = start % width, maxW = minW;
            visited[start] = true;
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int p = queue.Dequeue();
                pixels.Add(p);
                int h = p / width, w = p % width;
                minH = Math.Min(minH, h);
                maxH = Math.Max(maxH, h);
                minW = Math.Min(minW, w);
                maxW = Math.Max(maxW, w);
                foreach (int n in new[] { h > 0 ? p - width : -1, h < height - 1 ? p + width : -1, w > 0 ? p - 1 : -1, w < width - 1 ? p + 1 : -1 })
                {
                    if (n < 0 || visited[n] || classes[n] != classIdx) continue;
                    visited[n] = true;
                    queue.Enqueue(n);
                }
            }

            components.Add((classIdx, pixels.ToArray(), minH, minW, maxH, maxW));
        }

        T[] confidences = components.Select(c => DocumentTensorOps.Mean(Engine, DocumentTensorOps.Pick(Engine, winning, c.Pixels))).ToArray();
        var passes = confidences.Length == 0
            ? Array.Empty<int>()
            : DocumentTensorOps.Hits(Engine, DocumentTensorOps.AtLeast(Engine, new Tensor<T>(confidences, new[] { confidences.Length }), NumOps.FromDouble(threshold)));
        foreach (int i in passes)
        {
            var c = components[i];
            if ((c.MaxH - c.MinH + 1) * (c.MaxW - c.MinW + 1) < 100) continue; // Minimum 100-pixel box

            regions.Add(new DocumentRegion<T>
            {
                RegionType = c.Class < DocBankClasses.Length ? DocBankClasses[c.Class] : DocumentRegionType.Other,
                BoundingBox = new Vector<T>(DocumentTensorOps.ToT<T>(new[] { c.MinW, c.MinH, c.MaxW, c.MaxH })),
                Confidence = confidences[i],
                Index = regions.Count
            });
        }

        return new PageSegmentationResult<T>
        {
            Regions = regions,
            SegmentationMask = mask,
            ClassProbabilities = Engine.Softmax(output, output.Shape.Length > 1 ? 1 : 0)
        };
    }

    /// <summary>The first image's class probabilities as <c>[numClasses, height * width]</c>, with the map size.</summary>
    private (Tensor<T> Probabilities, int Height, int Width) FirstImageProbabilities(Tensor<T> output)
    {
        int numClasses = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;
        int height = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int width = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;
        int plane = height * width;
        var first = Engine.TensorSlice(Engine.Reshape(output, new[] { output.Length / (numClasses * plane), numClasses, plane }),
            new[] { 0, 0, 0 }, new[] { 1, numClasses, plane });
        return (Engine.Softmax(Engine.Reshape(first, new[] { numClasses, plane }), 0), height, width);
    }

    private Tensor<T> ExtractSegmentationMask(Tensor<T> output)
    {
        // Output shape: [batch, numClasses, height, width]; the mask is the first image's per-pixel argmax.
        var (probabilities, height, width) = FirstImageProbabilities(output);
        return Engine.Reshape(Engine.TensorArgMax(probabilities, 0).Cast<T>(), new[] { height, width });
    }

    private IReadOnlyList<int> ComputeReadingOrder(IReadOnlyList<DocumentRegion<T>> regions)
    {
        // Simple reading order: top-to-bottom in ~50px rows, then left-to-right.
        if (regions.Count == 0) return Array.Empty<int>();
        var boxes = new Tensor<T>(regions.SelectMany(r => r.BoundingBox.ToArray().Take(4)).ToArray(), new[] { regions.Count, 4 });
        double[] Center(int lo) => Engine.TensorMultiplyScalar(Engine.TensorAdd(
            Engine.TensorSlice(boxes, new[] { 0, lo }, new[] { regions.Count, 1 }),
            Engine.TensorSlice(boxes, new[] { 0, lo + 2 }, new[] { regions.Count, 1 })), NumOps.FromDouble(0.5)).Cast<double>().ToArray();
        double[] centerX = Center(0), centerY = Center(1);

        return Enumerable.Range(0, regions.Count)
            .OrderBy(i => Math.Round(centerY[i] / 50) * 50)
            .ThenBy(i => centerX[i])
            .ToList();
    }

    #endregion

    #region Token Classification

    /// <summary>
    /// DocBank's twelve token labels (Li et al. 2020, Sec. 3), in the order of the token classifier's outputs.
    /// </summary>
    public static IReadOnlyList<DocumentRegionType> TokenLabels { get; } =
    [
        DocumentRegionType.Abstract,
        DocumentRegionType.Author,
        DocumentRegionType.Caption,
        DocumentRegionType.Equation,
        DocumentRegionType.Figure,
        DocumentRegionType.Footer,
        DocumentRegionType.List,
        DocumentRegionType.Paragraph,
        DocumentRegionType.Reference,
        DocumentRegionType.Section,
        DocumentRegionType.Table,
        DocumentRegionType.Title
    ];

    private LayoutLM<T>? _tokenClassifier;

    /// <summary>
    /// The paper's baseline (Li et al. 2020, Sec. 4): a LayoutLM token classifier over each token and its
    /// 2-D box, with one output per <see cref="TokenLabels"/> entry. Built on first use from
    /// <see cref="DocBankOptions.TokenClassifierOptions"/> (LayoutLM-base when unset) so a pixel-only caller
    /// never allocates it; train it with its own <c>Train</c> on packed <c>(tokenId, x0, y0, x1, y1)</c> rows.
    /// </summary>
    public LayoutLM<T> TokenClassifier => _tokenClassifier ??= CreateTokenClassifier();

    private LayoutLM<T> CreateTokenClassifier()
    {
        var options = _options.TokenClassifierOptions ?? new LayoutLMOptions();
        options.NumClasses = TokenLabels.Count;
        return new LayoutLM<T>(new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: options.MaxSequenceLength,
            inputWidth: 5,
            outputSize: TokenLabels.Count), options);
    }

    /// <summary>
    /// Labels each token of a page as DocBank's baselines do: <paramref name="packedTokens"/> is
    /// <c>[seq, 5]</c> rows of <c>(tokenId, x0, y0, x1, y1)</c> with boxes on LayoutLM's 0-1000 grid; each
    /// token gets the softmax argmax of the <see cref="TokenClassifier"/> and that probability.
    /// </summary>
    public IReadOnlyList<(DocumentRegionType Label, T Confidence)> ClassifyTokens(Tensor<T> packedTokens)
    {
        if (packedTokens is null)
            throw new ArgumentNullException(nameof(packedTokens));
        if (packedTokens.Rank != 2 || packedTokens.Shape[1] != 5)
            throw new ArgumentException("DocBank token classification expects packed [seq, 5] (tokenId, x0, y0, x1, y1) rows.", nameof(packedTokens));
        return TokenPredictions(TokenClassifier.Predict(packedTokens));
    }

    /// <summary>Per-token label (softmax argmax over the <see cref="TokenLabels"/> logits) and its probability.</summary>
    internal IReadOnlyList<(DocumentRegionType Label, T Confidence)> TokenPredictions(Tensor<T> logits)
    {
        int classes = TokenLabels.Count;
        int rows = logits.Length / classes;
        if (rows == 0)
            return Array.Empty<(DocumentRegionType, T)>();
        var (labels, scores) = DocumentTensorOps.RowMax(Engine, Engine.Softmax(Engine.Reshape(logits, new[] { rows, classes }), 1), classes);
        T[] confidences = scores.ToArray();
        return labels.Select((label, i) => (TokenLabels[label], confidences[i])).ToList();
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
        sb.AppendLine("DocBank Model Summary");
        sb.AppendLine("=====================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: ResNet + FPN + Segmentation Head");
        sb.AppendLine();
        sb.AppendLine($"Backbone Channels: {_backboneChannels}");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Use Text Features: {_useTextFeatures}");
        sb.AppendLine();
        sb.AppendLine("Supported Region Types:");
        foreach (var regionType in DocBankClasses.Skip(1)) // Skip background
        {
            sb.AppendLine($"  - {regionType}");
        }
        sb.AppendLine();
        sb.AppendLine($"Instance Segmentation: {SupportsInstanceSegmentation}");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies DocBank's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// DocBank uses ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
        => DocumentTensorOps.NormalizeChannels(Engine, EnsureBatchDimension(rawImage),
            DocumentTensorOps.ImageNetMean, DocumentTensorOps.ImageNetStd);

    /// <summary>
    /// Applies DocBank's industry-standard postprocessing: softmax over class dimension.
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput)
    {
        // Apply softmax over class dimension (industry standard for segmentation)
        return ApplySoftmax(modelOutput);
    }

    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        int classAxis = input.Shape.Length > 1 ? 1 : 0;
        return Engine.Softmax(input, classAxis);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DocBank",
            Description = "Document layout analysis and page segmentation (COLING 2020)",
            FeatureCount = _backboneChannels,
            Complexity = Layers.Count,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "backbone_channels", _backboneChannels },
                { "num_classes", _numClasses },
                { "hidden_dim", _hiddenDim },
                { "image_size", ImageSize },
                { "use_text_features", _useTextFeatures },
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
        if (!_useNativeMode) return RunOnnxInference(preprocessed);
        return RunBackbone(preprocessed);
    }

    /// <summary>
    /// Runs the convolutional backbone directly over the (rank-3 [C,H,W]) page image. The base
    /// NeuralNetworkBase.Forward normalizes/reshapes the input for its accelerated path, which mangles
    /// the conv feature-map layout into a flattened [H*W, C] rank-2 tensor for this ResNet backbone
    /// (residual blocks then re-throw on the rank). Driving the layers directly keeps the [C,H,W]
    /// layout intact — the same approach the frame-interpolation / audio models use for their custom
    /// dataflow. Every layer's Forward is tape-aware, so this serves both inference and training.
    /// </summary>
    private Tensor<T> RunBackbone(Tensor<T> x)
    {
        var c = x;
        foreach (var layer in Layers) c = layer.Forward(c);
        return c;
    }

    /// <summary>Training forward — the same direct backbone pass as inference (tape-aware).</summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => RunBackbone(PreprocessDocument(input));

    /// <summary>
    /// The optimizer the tape-training path (TrainWithTape) uses: the one the constructor resolved.
    /// </summary>
    /// <remarks>
    /// That is the caller's optimizer when one was supplied, otherwise the paper recipe (AdamW at
    /// 5e-5, or at <see cref="DocBankOptions.LearningRate"/> when set). This previously built a
    /// separate decaying Adam at a hardcoded 1e-3, so training ignored both the recipe and the
    /// option, and every configured learning rate left the weights bit-identical.
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer;

    /// <inheritdoc />
    /// <remarks>The weights belong to the loaded graph in this mode. The base refuses
    /// the write on every parameter surface, so the guard is stated once, here.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxSession?.Dispose();
            _tokenClassifier?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
