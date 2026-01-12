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

namespace AiDotNet.Document.OCR.TextDetection;

/// <summary>
/// PSENet (Progressive Scale Expansion Network) for text detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PSENet uses a novel progressive scale expansion algorithm to accurately detect
/// text instances of various shapes and sizes, especially useful for closely spaced text.
/// </para>
/// <para>
/// <b>For Beginners:</b> PSENet handles difficult text detection scenarios:
/// 1. Detects text at multiple scales (kernels)
/// 2. Progressively expands from smallest to largest
/// 3. Separates closely spaced text instances
/// 4. Handles arbitrary-shaped text
///
/// Key features:
/// - Multi-scale kernel prediction
/// - Progressive scale expansion algorithm
/// - Handles closely adjacent text
/// - Accurate boundary detection
///
/// Example usage:
/// <code>
/// var model = new PSENet&lt;float&gt;(architecture);
/// var result = model.DetectText(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Shape Robust Text Detection with Progressive Scale Expansion Network" (CVPR 2019)
/// https://arxiv.org/abs/1903.12473
/// </para>
/// </remarks>
public class PSENet<T> : DocumentNeuralNetworkBase<T>, ITextDetector<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _backboneChannels;
    private readonly int _featureChannels;
    private readonly int _numKernels;

    // Native mode layers
    private readonly List<ILayer<T>> _backboneLayers = [];
    private readonly List<ILayer<T>> _fpnLayers = [];
    private readonly List<ILayer<T>> _segmentationLayers = [];

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public bool SupportsRotatedText => true;

    /// <inheritdoc/>
    public int MinTextHeight => 8;

    /// <inheritdoc/>
    public bool SupportsPolygonOutput => true;

    /// <summary>
    /// Gets the number of scale kernels.
    /// </summary>
    public int NumKernels => _numKernels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a PSENet model using a pre-trained ONNX model for inference.
    /// </summary>
    public PSENet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageSize = 640,
        int backboneChannels = 256,
        int featureChannels = 256,
        int numKernels = 7,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _backboneChannels = backboneChannels;
        _featureChannels = featureChannels;
        _numKernels = numKernels;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a PSENet model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (PSENet from CVPR 2019):</b>
    /// - Backbone: ResNet-50/152
    /// - FPN: Feature Pyramid Network
    /// - Output: Multi-scale kernels (default 7)
    /// - Post-processing: Progressive scale expansion
    /// </para>
    /// </remarks>
    public PSENet(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 640,
        int backboneChannels = 256,
        int featureChannels = 256,
        int numKernels = 7,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _backboneChannels = backboneChannels;
        _featureChannels = featureChannels;
        _numKernels = numKernels;
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

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultPSENetLayers(
            imageSize: ImageSize,
            backboneChannels: _backboneChannels,
            featureChannels: _featureChannels,
            numKernels: _numKernels));
    }

    #endregion

    #region ITextDetector Implementation

    /// <inheritdoc/>
    public TextDetectionResult<T> DetectText(Tensor<T> documentImage)
    {
        return DetectText(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public TextDetectionResult<T> DetectText(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var regions = ParsePSENetOutput(output, confidenceThreshold);

        return new TextDetectionResult<T>
        {
            TextRegions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public IEnumerable<TextDetectionResult<T>> DetectTextBatch(IEnumerable<Tensor<T>> documentImages)
    {
        foreach (var image in documentImages)
            yield return DetectText(image);
    }

    /// <inheritdoc/>
    public Tensor<T> GetHeatmap()
    {
        return Tensor<T>.CreateDefault([ImageSize, ImageSize], NumOps.Zero);
    }

    /// <inheritdoc/>
    public Tensor<T> GetProbabilityMap(Tensor<T> image)
    {
        ValidateImageShape(image);
        var preprocessed = PreprocessDocument(image);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        // Return the largest kernel (full text region)
        int outH = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int outW = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;
        var probMap = Tensor<T>.CreateDefault([outH, outW], NumOps.Zero);

        for (int h = 0; h < outH; h++)
        {
            for (int w = 0; w < outW; w++)
            {
                // Use the last kernel (largest scale)
                probMap[h, w] = output[0, _numKernels - 1, h, w];
            }
        }

        return probMap;
    }

    private List<TextRegion<T>> ParsePSENetOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<TextRegion<T>>();

        // PSENet output: [batch, numKernels, H, W]
        // Kernels are ordered from smallest (index 0) to largest (index numKernels-1)
        // Progressive Scale Expansion: start from smallest kernel and expand

        int outH = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int outW = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;

        // Step 1: Find connected components on smallest kernel (text centers)
        var labels = new int[outH, outW];
        var componentPixels = new Dictionary<int, List<(int y, int x)>>();
        var componentScores = new Dictionary<int, double>();
        int nextLabel = 1;

        // Binarize smallest kernel
        for (int y = 0; y < outH; y++)
        {
            for (int x = 0; x < outW; x++)
            {
                double score = NumOps.ToDouble(output[0, 0, y, x]);
                if (score >= threshold)
                {
                    // Check neighbors for existing labels
                    int label = 0;
                    if (y > 0 && labels[y - 1, x] > 0) label = labels[y - 1, x];
                    else if (x > 0 && labels[y, x - 1] > 0) label = labels[y, x - 1];

                    if (label == 0)
                    {
                        label = nextLabel++;
                        componentPixels[label] = [];
                        componentScores[label] = 0;
                    }

                    labels[y, x] = label;
                    componentPixels[label].Add((y, x));
                    componentScores[label] = Math.Max(componentScores[label], score);
                }
            }
        }

        // Step 2: Progressive scale expansion through remaining kernels
        for (int k = 1; k < _numKernels; k++)
        {
            var expanded = true;
            while (expanded)
            {
                expanded = false;
                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        if (labels[y, x] == 0)
                        {
                            double score = NumOps.ToDouble(output[0, k, y, x]);
                            if (score >= threshold * 0.5) // Lower threshold for expansion
                            {
                                // Check 4-connected neighbors for existing labels
                                int neighborLabel = 0;
                                if (y > 0 && labels[y - 1, x] > 0) neighborLabel = labels[y - 1, x];
                                else if (y < outH - 1 && labels[y + 1, x] > 0) neighborLabel = labels[y + 1, x];
                                else if (x > 0 && labels[y, x - 1] > 0) neighborLabel = labels[y, x - 1];
                                else if (x < outW - 1 && labels[y, x + 1] > 0) neighborLabel = labels[y, x + 1];

                                if (neighborLabel > 0)
                                {
                                    labels[y, x] = neighborLabel;
                                    componentPixels[neighborLabel].Add((y, x));
                                    expanded = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Extract bounding boxes and polygons from expanded components
        int regionId = 0;
        foreach (var (label, pixels) in componentPixels)
        {
            if (pixels.Count < 10) continue; // Filter tiny components

            // Get bounding box
            int minY = pixels.Min(p => p.y);
            int maxY = pixels.Max(p => p.y);
            int minX = pixels.Min(p => p.x);
            int maxX = pixels.Max(p => p.x);

            // Extract convex hull for polygon (simplified: use bounding corners)
            var polygon = ExtractConvexHull(pixels);

            double avgScore = componentScores[label];

            regions.Add(new TextRegion<T>
            {
                Confidence = NumOps.FromDouble(avgScore),
                ConfidenceValue = avgScore,
                BoundingBox = new Vector<T>([
                    NumOps.FromDouble(minX),
                    NumOps.FromDouble(minY),
                    NumOps.FromDouble(maxX),
                    NumOps.FromDouble(maxY)
                ]),
                PolygonPoints = polygon.Select(p => new Vector<T>([
                    NumOps.FromDouble(p.x),
                    NumOps.FromDouble(p.y)
                ])).ToList(),
                Index = regionId++
            });
        }

        return regions;
    }

    /// <summary>
    /// Extracts convex hull from pixel coordinates (simplified algorithm).
    /// </summary>
    private static List<(double x, double y)> ExtractConvexHull(List<(int y, int x)> pixels)
    {
        if (pixels.Count < 3) return pixels.Select(p => ((double)p.x, (double)p.y)).ToList();

        // Find extreme points
        var topLeft = pixels.OrderBy(p => p.y).ThenBy(p => p.x).First();
        var topRight = pixels.OrderBy(p => p.y).ThenByDescending(p => p.x).First();
        var bottomRight = pixels.OrderByDescending(p => p.y).ThenByDescending(p => p.x).First();
        var bottomLeft = pixels.OrderByDescending(p => p.y).ThenBy(p => p.x).First();

        // Return quadrilateral approximation of hull
        return
        [
            (topLeft.x, topLeft.y),
            (topRight.x, topRight.y),
            (bottomRight.x, bottomRight.y),
            (bottomLeft.x, bottomLeft.y)
        ];
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
        sb.AppendLine("PSENet Model Summary");
        sb.AppendLine("====================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: ResNet + FPN + Multi-scale Kernels");
        sb.AppendLine($"Backbone Channels: {_backboneChannels}");
        sb.AppendLine($"Feature Channels: {_featureChannels}");
        sb.AppendLine($"Number of Kernels: {_numKernels}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Progressive Expansion: Yes");
        sb.AppendLine($"Arbitrary Shapes: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies PSENet's industry-standard preprocessing: ImageNet normalization with scale.
    /// </summary>
    /// <remarks>
    /// PSENet (Progressive Scale Expansion Network) uses ImageNet normalization with
    /// mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225], with /255 scaling.
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
                        normalized.Data[idx] = NumOps.FromDouble((NumOps.ToDouble(image.Data[idx]) / 255.0 - mean) / std);
                    }
                }
            }
        }
        return normalized;
    }

    /// <summary>
    /// Applies PSENet's industry-standard postprocessing: pass-through (kernel maps are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "PSENet",
            ModelType = ModelType.NeuralNetwork,
            Description = "PSENet for progressive scale expansion text detection (CVPR 2019)",
            FeatureCount = _featureChannels,
            Complexity = _numKernels,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "backbone_channels", _backboneChannels },
                { "feature_channels", _featureChannels },
                { "num_kernels", _numKernels },
                { "image_size", ImageSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_backboneChannels);
        writer.Write(_featureChannels);
        writer.Write(_numKernels);
        writer.Write(ImageSize);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int backboneChannels = reader.ReadInt32();
        int featureChannels = reader.ReadInt32();
        int numKernels = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new PSENet<T>(Architecture, ImageSize, _backboneChannels, _featureChannels, _numKernels);
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
        T lr = NumOps.FromDouble(0.0001);
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
