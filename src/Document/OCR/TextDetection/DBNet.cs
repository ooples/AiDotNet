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
/// DBNet (Differentiable Binarization Network) for real-time text detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DBNet is a fast and accurate text detection model that uses differentiable binarization
/// to produce sharp text boundary predictions. It outputs probability, threshold, and binary maps.
/// </para>
/// <para>
/// <b>For Beginners:</b> DBNet finds where text is located in an image. It works by:
/// 1. Creating a "probability map" showing how likely each pixel is to be text
/// 2. Creating a "threshold map" that adapts to different text styles
/// 3. Combining them into a "binary map" showing exact text regions
///
/// The key innovation is that the threshold is learned, not fixed, which helps with
/// various fonts, sizes, and backgrounds.
///
/// Example usage:
/// <code>
/// var dbnet = new DBNet&lt;float&gt;(architecture);
/// var result = dbnet.DetectText(documentImage);
/// foreach (var region in result.TextRegions)
/// {
///     Console.WriteLine($"Found text at: {region.BoundingBox}");
/// }
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Real-time Scene Text Detection with Differentiable Binarization" (AAAI 2020)
/// https://arxiv.org/abs/1911.08947
/// </para>
/// </remarks>
public class DBNet<T> : DocumentNeuralNetworkBase<T>, ITextDetector<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _backboneChannels;
    private readonly int _innerChannels;
    private readonly double _expandRatio;
    private readonly double _thresholdK;
    private readonly int _minTextArea;

    // Native mode layers
    private readonly List<ILayer<T>> _backboneLayers = [];
    private readonly List<ILayer<T>> _fpnLayers = [];
    private readonly List<ILayer<T>> _probabilityHead = [];
    private readonly List<ILayer<T>> _thresholdHead = [];

    // Cached outputs for inspection
    private Tensor<T>? _lastProbabilityMap;
    private Tensor<T>? _lastThresholdMap;
    private Tensor<T>? _lastBinaryMap;

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
    public bool SupportsPolygonOutput => true;

    /// <inheritdoc/>
    public int MinTextHeight => 8;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DBNet model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="imageSize">Expected input image size (default: 640).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 256 for ResNet-18).</param>
    /// <param name="innerChannels">FPN inner channels (default: 256).</param>
    /// <param name="expandRatio">Ratio for expanding detected regions (default: 1.5).</param>
    /// <param name="thresholdK">K value for DB formula (default: 50).</param>
    /// <param name="minTextArea">Minimum text area in pixels (default: 16).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <exception cref="ArgumentNullException">Thrown if onnxModelPath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX model file doesn't exist.</exception>
    public DBNet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int imageSize = 640,
        int backboneChannels = 256,
        int innerChannels = 256,
        double expandRatio = 1.5,
        double thresholdK = 50,
        int minTextArea = 16,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));

        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model file not found: {onnxModelPath}", onnxModelPath);

        _useNativeMode = false;
        _backboneChannels = backboneChannels;
        _innerChannels = innerChannels;
        _expandRatio = expandRatio;
        _thresholdK = thresholdK;
        _minTextArea = minTextArea;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DBNet model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="imageSize">Expected input image size (default: 640).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 256 for ResNet-18).</param>
    /// <param name="innerChannels">FPN inner channels (default: 256).</param>
    /// <param name="expandRatio">Ratio for expanding detected regions (default: 1.5).</param>
    /// <param name="thresholdK">K value for DB formula (default: 50).</param>
    /// <param name="minTextArea">Minimum text area in pixels (default: 16).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (from AAAI 2020 paper):</b>
    /// - Backbone: ResNet-18 or ResNet-50
    /// - FPN channels: 256
    /// - Threshold K: 50
    /// - Input size: 640×640
    /// </para>
    /// </remarks>
    public DBNet(
        NeuralNetworkArchitecture<T> architecture,
        int imageSize = 640,
        int backboneChannels = 256,
        int innerChannels = 256,
        double expandRatio = 1.5,
        double thresholdK = 50,
        int minTextArea = 16,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _backboneChannels = backboneChannels;
        _innerChannels = innerChannels;
        _expandRatio = expandRatio;
        _thresholdK = thresholdK;
        _minTextArea = minTextArea;
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

        // Use LayerHelper to create default DBNet layers
        Layers.AddRange(LayerHelper<T>.CreateDefaultDBNetLayers(
            imageSize: ImageSize,
            backboneChannels: _backboneChannels,
            innerChannels: _innerChannels));
    }

    #endregion

    #region ITextDetector Implementation

    /// <inheritdoc/>
    public TextDetectionResult<T> DetectText(Tensor<T> image)
    {
        return DetectText(image, 0.3);
    }

    /// <inheritdoc/>
    public TextDetectionResult<T> DetectText(Tensor<T> image, double confidenceThreshold)
    {
        ValidateImageShape(image);

        var startTime = DateTime.UtcNow;

        var result = _useNativeMode
            ? DetectTextNative(image, confidenceThreshold)
            : DetectTextOnnx(image, confidenceThreshold);

        return new TextDetectionResult<T>
        {
            TextRegions = result.TextRegions,
            ProbabilityMap = _lastProbabilityMap,
            ThresholdMap = _lastThresholdMap,
            BinaryMap = _lastBinaryMap,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public Tensor<T> GetProbabilityMap(Tensor<T> image)
    {
        ValidateImageShape(image);

        var preprocessed = PreprocessDocument(image);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        // Extract probability map from output (first channel)
        return ExtractProbabilityMap(output);
    }

    private TextDetectionResult<T> DetectTextNative(Tensor<T> image, double threshold)
    {
        var preprocessed = PreprocessDocument(image);
        var output = Forward(preprocessed);

        return ParseDetectionOutput(output, threshold);
    }

    private TextDetectionResult<T> DetectTextOnnx(Tensor<T> image, double threshold)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var preprocessed = PreprocessDocument(image);
        var output = RunOnnxInference(preprocessed);

        return ParseDetectionOutput(output, threshold);
    }

    private TextDetectionResult<T> ParseDetectionOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<TextRegion<T>>();

        // DBNet outputs: probability map, threshold map, binary map
        // Output shape: [batch, 3, height, width] or [batch, 1, height, width] for just probability
        int height = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int width = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;

        // Extract probability map (first channel)
        _lastProbabilityMap = ExtractProbabilityMap(output);

        // Extract threshold map (second channel if available)
        _lastThresholdMap = ExtractThresholdMap(output);

        // Compute binary map using differentiable binarization
        _lastBinaryMap = ComputeBinaryMap(_lastProbabilityMap, _lastThresholdMap);

        // Find connected components in binary map
        regions = FindTextRegions(_lastBinaryMap, threshold);

        return new TextDetectionResult<T>
        {
            TextRegions = regions
        };
    }

    private Tensor<T> ExtractProbabilityMap(Tensor<T> output)
    {
        int height = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int width = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;

        var probMap = new Tensor<T>([height, width]);

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int srcIdx = h * width + w;
                if (srcIdx < output.Data.Length)
                {
                    probMap[h, w] = output.Data[srcIdx];
                }
            }
        }

        return probMap;
    }

    private Tensor<T> ExtractThresholdMap(Tensor<T> output)
    {
        int height = output.Shape.Length > 2 ? output.Shape[2] : ImageSize;
        int width = output.Shape.Length > 3 ? output.Shape[3] : ImageSize;
        int channels = output.Shape.Length > 1 ? output.Shape[1] : 1;

        var threshMap = new Tensor<T>([height, width]);

        if (channels >= 2)
        {
            int channelOffset = height * width;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int srcIdx = channelOffset + h * width + w;
                    if (srcIdx < output.Data.Length)
                    {
                        threshMap[h, w] = output.Data[srcIdx];
                    }
                }
            }
        }
        else
        {
            // Default threshold if not available
            T defaultThresh = NumOps.FromDouble(0.3);
            for (int i = 0; i < threshMap.Data.Length; i++)
            {
                threshMap.Data[i] = defaultThresh;
            }
        }

        return threshMap;
    }

    private Tensor<T> ComputeBinaryMap(Tensor<T> probMap, Tensor<T> threshMap)
    {
        // Differentiable binarization: B = 1 / (1 + exp(-k * (P - T)))
        var binaryMap = new Tensor<T>(probMap.Shape);

        for (int i = 0; i < probMap.Data.Length; i++)
        {
            double p = NumOps.ToDouble(probMap.Data[i]);
            double t = NumOps.ToDouble(threshMap.Data[i]);
            double b = 1.0 / (1.0 + Math.Exp(-_thresholdK * (p - t)));
            binaryMap.Data[i] = NumOps.FromDouble(b);
        }

        return binaryMap;
    }

    private List<TextRegion<T>> FindTextRegions(Tensor<T> binaryMap, double threshold)
    {
        var regions = new List<TextRegion<T>>();
        int height = binaryMap.Shape[0];
        int width = binaryMap.Shape[1];

        // Simple connected component analysis
        var visited = new bool[height, width];
        int regionIndex = 0;

        for (int startH = 0; startH < height; startH++)
        {
            for (int startW = 0; startW < width; startW++)
            {
                double value = NumOps.ToDouble(binaryMap[startH, startW]);
                if (value > threshold && !visited[startH, startW])
                {
                    // BFS to find connected component
                    var componentPixels = new List<(int h, int w)>();
                    var queue = new Queue<(int h, int w)>();
                    queue.Enqueue((startH, startW));
                    visited[startH, startW] = true;

                    int minH = startH, maxH = startH;
                    int minW = startW, maxW = startW;
                    double sumConfidence = 0;

                    while (queue.Count > 0)
                    {
                        var (h, w) = queue.Dequeue();
                        componentPixels.Add((h, w));
                        sumConfidence += NumOps.ToDouble(binaryMap[h, w]);

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
                            if (nh >= 0 && nh < height && nw >= 0 && nw < width &&
                                !visited[nh, nw] && NumOps.ToDouble(binaryMap[nh, nw]) > threshold)
                            {
                                visited[nh, nw] = true;
                                queue.Enqueue((nh, nw));
                            }
                        }
                    }

                    // Filter by minimum area
                    int area = (maxH - minH + 1) * (maxW - minW + 1);
                    if (area >= _minTextArea)
                    {
                        double avgConfidence = sumConfidence / componentPixels.Count;

                        // Expand bounding box
                        double expandH = (maxH - minH) * (_expandRatio - 1) / 2;
                        double expandW = (maxW - minW) * (_expandRatio - 1) / 2;

                        regions.Add(new TextRegion<T>
                        {
                            BoundingBox = new Vector<T>([
                                NumOps.FromDouble(Math.Max(0, minW - expandW)),
                                NumOps.FromDouble(Math.Max(0, minH - expandH)),
                                NumOps.FromDouble(Math.Min(width, maxW + expandW)),
                                NumOps.FromDouble(Math.Min(height, maxH + expandH))
                            ]),
                            Confidence = NumOps.FromDouble(avgConfidence),
                            ConfidenceValue = avgConfidence,
                            Index = regionIndex++
                        });
                    }
                }
            }
        }

        return regions;
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
        sb.AppendLine("DBNet Model Summary");
        sb.AppendLine("===================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Backbone Channels: {_backboneChannels}");
        sb.AppendLine($"FPN Inner Channels: {_innerChannels}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Threshold K: {_thresholdK}");
        sb.AppendLine($"Expand Ratio: {_expandRatio}");
        sb.AppendLine($"Min Text Area: {_minTextArea}");
        sb.AppendLine($"Supports Rotated Text: {SupportsRotatedText}");
        sb.AppendLine($"Supports Polygon Output: {SupportsPolygonOutput}");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies DBNet's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// DBNet (Differentiable Binarization) uses ImageNet normalization with
    /// mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] (MegVii paper).
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);

        // ImageNet normalization
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
    /// Applies DBNet's industry-standard postprocessing: pass-through (probability map is already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput)
    {
        return modelOutput;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DBNet",
            ModelType = ModelType.NeuralNetwork,
            Description = "Differentiable Binarization Network for real-time text detection (AAAI 2020)",
            FeatureCount = _innerChannels,
            Complexity = Layers.Count,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "backbone_channels", _backboneChannels },
                { "inner_channels", _innerChannels },
                { "image_size", ImageSize },
                { "threshold_k", _thresholdK },
                { "expand_ratio", _expandRatio },
                { "min_text_area", _minTextArea },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_backboneChannels);
        writer.Write(_innerChannels);
        writer.Write(ImageSize);
        writer.Write(_expandRatio);
        writer.Write(_thresholdK);
        writer.Write(_minTextArea);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int backboneChannels = reader.ReadInt32();
        int innerChannels = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        double expandRatio = reader.ReadDouble();
        double thresholdK = reader.ReadDouble();
        int minTextArea = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DBNet<T>(
            Architecture,
            ImageSize,
            _backboneChannels,
            _innerChannels,
            _expandRatio,
            _thresholdK,
            _minTextArea);
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
        T learningRate = NumOps.FromDouble(0.001);

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
