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
public class DocBank<T> : DocumentNeuralNetworkBase<T>, IPageSegmenter<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
        int imageSize = 1024,
        int backboneChannels = 256,
        int numClasses = 13,
        bool useTextFeatures = false,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));

        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model file not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _backboneChannels = backboneChannels;
        _numClasses = numClasses;
        _hiddenDim = 256;
        _useTextFeatures = useTextFeatures;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

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
        int imageSize = 1024,
        int backboneChannels = 256,
        int numClasses = 13,
        int hiddenDim = 256,
        bool useTextFeatures = false,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _backboneChannels = backboneChannels;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _useTextFeatures = useTextFeatures;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

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

    private PageSegmentationResult<T> ParseSegmentationOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<DocumentRegion<T>>();

        // Output shape: [batch, numClasses, height, width]
        int height = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int width = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;

        // Extract segmentation mask (argmax over classes)
        var mask = ExtractSegmentationMask(output);

        // Find connected components for each class
        var visited = new bool[height, width];

        for (int startH = 0; startH < height; startH++)
        {
            for (int startW = 0; startW < width; startW++)
            {
                int classIdx = (int)NumOps.ToDouble(mask[startH, startW]);

                if (classIdx > 0 && !visited[startH, startW]) // Skip background
                {
                    // BFS to find connected component
                    var queue = new Queue<(int h, int w)>();
                    queue.Enqueue((startH, startW));
                    visited[startH, startW] = true;

                    int minH = startH, maxH = startH;
                    int minW = startW, maxW = startW;
                    double sumConfidence = 0;
                    int pixelCount = 0;

                    while (queue.Count > 0)
                    {
                        var (h, w) = queue.Dequeue();
                        pixelCount++;

                        // Get confidence from class probabilities
                        if (output.Shape[1] > classIdx)
                        {
                            int probIdx = classIdx * height * width + h * width + w;
                            if (probIdx < output.Data.Length)
                            {
                                sumConfidence += NumOps.ToDouble(output.Data[probIdx]);
                            }
                        }

                        minH = Math.Min(minH, h);
                        maxH = Math.Max(maxH, h);
                        minW = Math.Min(minW, w);
                        maxW = Math.Max(maxW, w);

                        // Check 4-connected neighbors
                        int[] dh = [-1, 1, 0, 0];
                        int[] dw = [0, 0, -1, 1];
                        for (int d = 0; d < 4; d++)
                        {
                            int nh = h + dh[d];
                            int nw = w + dw[d];
                            if (nh >= 0 && nh < height && nw >= 0 && nw < width && !visited[nh, nw])
                            {
                                int neighborClass = (int)NumOps.ToDouble(mask[nh, nw]);
                                if (neighborClass == classIdx)
                                {
                                    visited[nh, nw] = true;
                                    queue.Enqueue((nh, nw));
                                }
                            }
                        }
                    }

                    // Filter by minimum area
                    int area = (maxH - minH + 1) * (maxW - minW + 1);
                    double avgConfidence = pixelCount > 0 ? sumConfidence / pixelCount : 0;

                    if (avgConfidence >= threshold && area >= 100) // Minimum 100 pixels
                    {
                        var regionType = classIdx < DocBankClasses.Length
                            ? DocBankClasses[classIdx]
                            : DocumentRegionType.Other;

                        regions.Add(new DocumentRegion<T>
                        {
                            RegionType = regionType,
                            BoundingBox = new Vector<T>([
                                NumOps.FromDouble(minW),
                                NumOps.FromDouble(minH),
                                NumOps.FromDouble(maxW),
                                NumOps.FromDouble(maxH)
                            ]),
                            Confidence = NumOps.FromDouble(avgConfidence),
                            ConfidenceValue = avgConfidence,
                            Index = regions.Count
                        });
                    }
                }
            }
        }

        return new PageSegmentationResult<T>
        {
            Regions = regions,
            SegmentationMask = mask,
            ClassProbabilities = output
        };
    }

    private Tensor<T> ExtractSegmentationMask(Tensor<T> output)
    {
        // Output shape: [batch, numClasses, height, width]
        int numClasses = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;
        int height = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int width = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;

        var mask = new Tensor<T>([height, width]);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double maxProb = double.MinValue;
                int maxClass = 0;

                for (int c = 0; c < numClasses; c++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < output.Data.Length)
                    {
                        double prob = NumOps.ToDouble(output.Data[idx]);
                        if (prob > maxProb)
                        {
                            maxProb = prob;
                            maxClass = c;
                        }
                    }
                }

                mask[h, w] = NumOps.FromDouble(maxClass);
            }
        }

        return mask;
    }

    private IReadOnlyList<int> ComputeReadingOrder(IReadOnlyList<DocumentRegion<T>> regions)
    {
        // Simple reading order: top-to-bottom, left-to-right
        // More sophisticated implementations would use XY-cut or learned models

        var indexed = regions.Select((r, i) => new
        {
            Index = i,
            CenterY = (NumOps.ToDouble(r.BoundingBox[1]) + NumOps.ToDouble(r.BoundingBox[3])) / 2,
            CenterX = (NumOps.ToDouble(r.BoundingBox[0]) + NumOps.ToDouble(r.BoundingBox[2])) / 2
        }).ToList();

        // Sort by Y first (top to bottom), then X (left to right)
        var sorted = indexed
            .OrderBy(r => Math.Round(r.CenterY / 50) * 50) // Group by ~50px rows
            .ThenBy(r => r.CenterX)
            .Select(r => r.Index)
            .ToList();

        // Assign reading order positions
        for (int i = 0; i < sorted.Count; i++)
        {
            // Update region's reading order position (would need mutable access)
        }

        return sorted;
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
    {
        var image = EnsureBatchDimension(rawImage);

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);

        // ImageNet normalization (industry standard for DocBank)
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
                        double value = NumOps.ToDouble(image.Data[idx]);
                        normalized.Data[idx] = NumOps.FromDouble((value - mean) / std);
                    }
                }
            }
        }

        return normalized;
    }

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
        int numClasses = input.Shape.Length > 1 ? input.Shape[1] : _numClasses;
        int height = input.Shape.Length > 2 ? input.Shape[2] : ImageSize;
        int width = input.Shape.Length > 3 ? input.Shape[3] : ImageSize;

        var output = new Tensor<T>(input.Shape);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double maxVal = double.MinValue;
                for (int c = 0; c < numClasses; c++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < input.Data.Length)
                    {
                        double val = NumOps.ToDouble(input.Data[idx]);
                        if (val > maxVal) maxVal = val;
                    }
                }

                double sumExp = 0;
                for (int c = 0; c < numClasses; c++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < input.Data.Length)
                    {
                        double val = NumOps.ToDouble(input.Data[idx]);
                        sumExp += Math.Exp(val - maxVal);
                    }
                }

                for (int c = 0; c < numClasses; c++)
                {
                    int idx = c * height * width + h * width + w;
                    if (idx < input.Data.Length)
                    {
                        double val = NumOps.ToDouble(input.Data[idx]);
                        output.Data[idx] = NumOps.FromDouble(Math.Exp(val - maxVal) / sumExp);
                    }
                }
            }
        }

        return output;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DocBank",
            ModelType = ModelType.NeuralNetwork,
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
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_backboneChannels);
        writer.Write(_numClasses);
        writer.Write(_hiddenDim);
        writer.Write(ImageSize);
        writer.Write(_useTextFeatures);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int backboneChannels = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        int hiddenDim = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        bool useTextFeatures = reader.ReadBoolean();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DocBank<T>(
            Architecture,
            ImageSize,
            _backboneChannels,
            _numClasses,
            _hiddenDim,
            _useTextFeatures);
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
        {
            throw new NotSupportedException("Training is not supported in ONNX inference mode.");
        }

        SetTrainingMode(true);

        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var lossGradient = LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        var paramGradients = CollectParameterGradients();
        UpdateParameters(paramGradients);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Parameter updates are not supported in ONNX inference mode.");
        }

        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.0001);

        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(currentParams);
    }

    private Vector<T> CollectParameterGradients()
    {
        var gradients = new List<T>();

        foreach (var layer in Layers)
        {
            var layerGradients = layer.GetParameterGradients();
            gradients.AddRange(layerGradients);
        }

        return new Vector<T>([.. gradients]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
