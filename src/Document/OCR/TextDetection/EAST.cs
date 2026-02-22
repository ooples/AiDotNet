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

namespace AiDotNet.Document.OCR.TextDetection;

/// <summary>
/// EAST (Efficient and Accurate Scene Text Detector) for text detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// EAST is a fast and accurate scene text detector that directly predicts text regions
/// without requiring complex post-processing like NMS across multiple stages.
/// </para>
/// <para>
/// <b>For Beginners:</b> EAST is designed for speed and accuracy:
/// 1. Single-shot detection (no multi-stage pipeline)
/// 2. Outputs rotated boxes or quadrilaterals
/// 3. Very fast inference
/// 4. Works on arbitrary text orientations
///
/// Key features:
/// - Fully convolutional architecture
/// - Multi-scale feature fusion
/// - Direct geometry prediction
/// - Efficient NMS
///
/// Example usage:
/// <code>
/// var model = new EAST&lt;float&gt;(architecture);
/// var result = model.DetectText(sceneImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "EAST: An Efficient and Accurate Scene Text Detector" (CVPR 2017)
/// https://arxiv.org/abs/1704.03155
/// </para>
/// </remarks>
public class EAST<T> : DocumentNeuralNetworkBase<T>, ITextDetector<T>
{
    private readonly EASTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _backboneChannels;
    private readonly int _featureChannels;
    private readonly string _geometryType;

    // Native mode layers
    private readonly List<ILayer<T>> _backboneLayers = [];
    private readonly List<ILayer<T>> _mergeLayers = [];
    private readonly List<ILayer<T>> _outputLayers = [];

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
    /// Gets the geometry output type (RBOX or QUAD).
    /// </summary>
    public string GeometryType => _geometryType;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an EAST model using a pre-trained ONNX model for inference.
    /// </summary>
    public EAST(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageSize = 512,
        int backboneChannels = 512,
        int featureChannels = 128,
        string geometryType = "RBOX",
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        EASTOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _options = options ?? new EASTOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _backboneChannels = backboneChannels;
        _featureChannels = featureChannels;
        _geometryType = geometryType;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates an EAST model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (EAST from CVPR 2017):</b>
    /// - Backbone: PVANet or VGG16
    /// - Feature merge: U-Net style
    /// - Output: Score map + Geometry (RBOX or QUAD)
    /// - NMS threshold: 0.2
    /// </para>
    /// </remarks>
    public EAST(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 512,
        int backboneChannels = 512,
        int featureChannels = 128,
        string geometryType = "RBOX",
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        EASTOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _options = options ?? new EASTOptions();
        Options = _options;

        _useNativeMode = true;
        _backboneChannels = backboneChannels;
        _featureChannels = featureChannels;
        _geometryType = geometryType;
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

        Layers.AddRange(LayerHelper<T>.CreateDefaultEASTLayers(
            imageSize: ImageSize,
            backboneChannels: _backboneChannels,
            featureChannels: _featureChannels,
            geometryType: _geometryType));
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

        var regions = ParseEASTOutput(output, confidenceThreshold);

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
        return Tensor<T>.CreateDefault([ImageSize / 4, ImageSize / 4], NumOps.Zero);
    }

    /// <inheritdoc/>
    public Tensor<T> GetProbabilityMap(Tensor<T> image)
    {
        ValidateImageShape(image);
        var preprocessed = PreprocessDocument(image);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        int outH = ImageSize / 4;
        int outW = ImageSize / 4;
        var probMap = Tensor<T>.CreateDefault([outH, outW], NumOps.Zero);

        for (int h = 0; h < outH; h++)
        {
            for (int w = 0; w < outW; w++)
            {
                probMap[h, w] = output[0, 0, h, w];
            }
        }

        return probMap;
    }

    private List<TextRegion<T>> ParseEASTOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<TextRegion<T>>();

        // EAST output: [batch, channels, H/4, W/4]
        // Channel 0: score map
        // RBOX: Channels 1-4 (d_top, d_right, d_bottom, d_left), Channel 5 (angle)
        // QUAD: Channels 1-8 (x,y offsets for 4 corners)

        int outH = output.Shape.Length > 2 ? output.Shape[2] : ImageSize / 4;
        int outW = output.Shape.Length > 3 ? output.Shape[3] : ImageSize / 4;
        int stride = 4; // EAST output stride

        int regionId = 0;
        for (int y = 0; y < outH; y++)
        {
            for (int x = 0; x < outW; x++)
            {
                double score = NumOps.ToDouble(output[0, 0, y, x]);
                if (score >= threshold)
                {
                    // Center point in original image coordinates
                    double cx = (x + 0.5) * stride;
                    double cy = (y + 0.5) * stride;

                    Vector<T> bbox;
                    List<(double x, double y)> polygonPoints;

                    if (_geometryType == "RBOX")
                    {
                        // RBOX geometry: distances from center to edges + angle
                        double dTop = NumOps.ToDouble(output[0, 1, y, x]);
                        double dRight = NumOps.ToDouble(output[0, 2, y, x]);
                        double dBottom = NumOps.ToDouble(output[0, 3, y, x]);
                        double dLeft = NumOps.ToDouble(output[0, 4, y, x]);
                        double angle = output.Shape[1] > 5 ? NumOps.ToDouble(output[0, 5, y, x]) : 0;

                        // Calculate axis-aligned bounding box
                        double x1 = cx - dLeft;
                        double y1 = cy - dTop;
                        double x2 = cx + dRight;
                        double y2 = cy + dBottom;

                        bbox = new Vector<T>([
                            NumOps.FromDouble(Math.Max(0, x1)),
                            NumOps.FromDouble(Math.Max(0, y1)),
                            NumOps.FromDouble(Math.Min(ImageSize, x2)),
                            NumOps.FromDouble(Math.Min(ImageSize, y2))
                        ]);

                        // Calculate rotated polygon points if angle is significant
                        polygonPoints = CalculateRotatedBox(cx, cy, dLeft + dRight, dTop + dBottom, angle);
                    }
                    else // QUAD geometry
                    {
                        // QUAD: 8 values representing x,y offsets for 4 corners
                        double[] offsets = new double[8];
                        for (int i = 0; i < 8 && i + 1 < output.Shape[1]; i++)
                        {
                            offsets[i] = NumOps.ToDouble(output[0, 1 + i, y, x]);
                        }

                        // Calculate corner points
                        polygonPoints =
                        [
                            (cx + offsets[0], cy + offsets[1]), // Top-left
                            (cx + offsets[2], cy + offsets[3]), // Top-right
                            (cx + offsets[4], cy + offsets[5]), // Bottom-right
                            (cx + offsets[6], cy + offsets[7])  // Bottom-left
                        ];

                        // Calculate bounding box from polygon
                        double minX = polygonPoints.Min(p => p.x);
                        double minY = polygonPoints.Min(p => p.y);
                        double maxX = polygonPoints.Max(p => p.x);
                        double maxY = polygonPoints.Max(p => p.y);

                        bbox = new Vector<T>([
                            NumOps.FromDouble(Math.Max(0, minX)),
                            NumOps.FromDouble(Math.Max(0, minY)),
                            NumOps.FromDouble(Math.Min(ImageSize, maxX)),
                            NumOps.FromDouble(Math.Min(ImageSize, maxY))
                        ]);
                    }

                    regions.Add(new TextRegion<T>
                    {
                        Confidence = NumOps.FromDouble(score),
                        ConfidenceValue = score,
                        BoundingBox = bbox,
                        PolygonPoints = polygonPoints.Select(p => new Vector<T>([
                            NumOps.FromDouble(p.x),
                            NumOps.FromDouble(p.y)
                        ])).ToList(),
                        Index = regionId++
                    });
                }
            }
        }

        // Apply non-maximum suppression
        return ApplyNMS(regions, 0.4);
    }

    /// <summary>
    /// Calculates rotated bounding box corners.
    /// </summary>
    private static List<(double x, double y)> CalculateRotatedBox(double cx, double cy, double width, double height, double angle)
    {
        double cos = Math.Cos(angle);
        double sin = Math.Sin(angle);
        double hw = width / 2;
        double hh = height / 2;

        // Calculate four corners of rotated rectangle
        return
        [
            (cx + (-hw * cos - (-hh) * sin), cy + (-hw * sin + (-hh) * cos)), // Top-left
            (cx + (hw * cos - (-hh) * sin), cy + (hw * sin + (-hh) * cos)),   // Top-right
            (cx + (hw * cos - hh * sin), cy + (hw * sin + hh * cos)),         // Bottom-right
            (cx + (-hw * cos - hh * sin), cy + (-hw * sin + hh * cos))        // Bottom-left
        ];
    }

    /// <summary>
    /// Applies non-maximum suppression to remove overlapping detections.
    /// </summary>
    private static List<TextRegion<T>> ApplyNMS(List<TextRegion<T>> regions, double iouThreshold)
    {
        if (regions.Count <= 1) return regions;

        // Sort by confidence descending
        var sorted = regions.OrderByDescending(r => r.ConfidenceValue).ToList();
        var kept = new List<TextRegion<T>>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            kept.Add(best);
            sorted.RemoveAt(0);

            sorted = sorted.Where(r => CalculateIoU(best, r) < iouThreshold).ToList();
        }

        return kept;
    }

    /// <summary>
    /// Calculates intersection over union between two regions.
    /// </summary>
    private static double CalculateIoU(TextRegion<T> a, TextRegion<T> b)
    {
        if (a.BoundingBox.Length < 4 || b.BoundingBox.Length < 4) return 0;

        var numOps = MathHelper.GetNumericOperations<T>();

        double ax1 = numOps.ToDouble(a.BoundingBox[0]);
        double ay1 = numOps.ToDouble(a.BoundingBox[1]);
        double ax2 = numOps.ToDouble(a.BoundingBox[2]);
        double ay2 = numOps.ToDouble(a.BoundingBox[3]);

        double bx1 = numOps.ToDouble(b.BoundingBox[0]);
        double by1 = numOps.ToDouble(b.BoundingBox[1]);
        double bx2 = numOps.ToDouble(b.BoundingBox[2]);
        double by2 = numOps.ToDouble(b.BoundingBox[3]);

        double ix1 = Math.Max(ax1, bx1);
        double iy1 = Math.Max(ay1, by1);
        double ix2 = Math.Min(ax2, bx2);
        double iy2 = Math.Min(ay2, by2);

        if (ix1 >= ix2 || iy1 >= iy2) return 0;

        double intersection = (ix2 - ix1) * (iy2 - iy1);
        double areaA = (ax2 - ax1) * (ay2 - ay1);
        double areaB = (bx2 - bx1) * (by2 - by1);
        double union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0;
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
        sb.AppendLine("EAST Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: FCN with Feature Pyramid");
        sb.AppendLine($"Backbone Channels: {_backboneChannels}");
        sb.AppendLine($"Feature Channels: {_featureChannels}");
        sb.AppendLine($"Geometry Type: {_geometryType}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Output Size: {ImageSize / 4}x{ImageSize / 4}");
        sb.AppendLine($"Rotated Text: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies EAST's industry-standard preprocessing: VGG mean subtraction.
    /// </summary>
    /// <remarks>
    /// EAST (Efficient and Accurate Scene Text detector) uses VGG-style mean subtraction
    /// with mean=[123.68, 116.78, 103.94] (CVPR 2017 paper).
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);
        double[] means = [123.68, 116.78, 103.94];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double mean = c < means.Length ? means[c] : 128;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        normalized.Data.Span[idx] = NumOps.FromDouble(NumOps.ToDouble(image.Data.Span[idx]) - mean);
                    }
                }
            }
        }
        return normalized;
    }

    /// <summary>
    /// Applies EAST's industry-standard postprocessing: pass-through (geometry maps are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "EAST",
            ModelType = ModelType.NeuralNetwork,
            Description = "EAST for efficient scene text detection (CVPR 2017)",
            FeatureCount = _featureChannels,
            Complexity = Layers.Count,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "backbone_channels", _backboneChannels },
                { "feature_channels", _featureChannels },
                { "geometry_type", _geometryType },
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
        writer.Write(_geometryType);
        writer.Write(ImageSize);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int backboneChannels = reader.ReadInt32();
        int featureChannels = reader.ReadInt32();
        string geometryType = reader.ReadString();
        int imageSize = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new EAST<T>(Architecture, ImageSize, _backboneChannels, _featureChannels, _geometryType);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Overrides Forward to handle EAST's parallel output heads (score map + geometry).
    /// The last two layers are parallel heads that both receive the feature map,
    /// not sequential layers.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (Layers.Count < 3)
            return base.Forward(input);

        // Run all layers except the last two (which are parallel output heads)
        Tensor<T> featureMap = input;
        for (int i = 0; i < Layers.Count - 2; i++)
        {
            featureMap = Layers[i].Forward(featureMap);
        }

        // Run score map head and geometry head in parallel on the same feature map
        var scoreMap = Layers[^2].Forward(featureMap);
        var geometry = Layers[^1].Forward(featureMap);

        // Concatenate along channel dimension: [batch, 1+geometryChannels, H, W]
        return ConcatenateTensors(scoreMap, geometry);
    }

    private static Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        // Both tensors have shape [batch, channels, H, W]
        // Concatenate along dimension 1 (channels)
        int batch = a.Shape[0];
        int cA = a.Shape[1];
        int cB = b.Shape[1];
        int h = a.Shape[2];
        int w = a.Shape[3];
        int totalChannels = cA + cB;

        var result = new Tensor<T>([batch, totalChannels, h, w]);
        int planeSize = h * w;

        for (int n = 0; n < batch; n++)
        {
            int batchOffset = n * totalChannels * planeSize;
            int srcBatchOffsetA = n * cA * planeSize;
            int srcBatchOffsetB = n * cB * planeSize;

            // Copy channels from tensor a
            for (int c = 0; c < cA; c++)
            {
                a.Data.Span.Slice(srcBatchOffsetA + c * planeSize, planeSize)
                    .CopyTo(result.Data.Span.Slice(batchOffset + c * planeSize, planeSize));
            }

            // Copy channels from tensor b
            for (int c = 0; c < cB; c++)
            {
                b.Data.Span.Slice(srcBatchOffsetB + c * planeSize, planeSize)
                    .CopyTo(result.Data.Span.Slice(batchOffset + (cA + c) * planeSize, planeSize));
            }
        }

        return result;
    }

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
