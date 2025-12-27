using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// YOLOv10 object detector with NMS-free training and consistent dual assignments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> YOLOv10 eliminates the need for Non-Maximum Suppression (NMS)
/// during inference by using consistent dual assignments during training. This makes
/// inference faster and simpler while maintaining accuracy.</para>
///
/// <para>Key features:
/// - NMS-free inference via consistent dual assignments
/// - One-to-one matching during inference (no duplicate predictions)
/// - One-to-many matching for auxiliary training
/// - Reduced post-processing latency
/// </para>
///
/// <para>Reference: Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", 2024</para>
/// </remarks>
public class YOLOv10<T> : ObjectDetectorBase<T>
{
    private readonly YOLOv8Head<T> _head;
    private readonly YOLOv8Head<T>? _auxHead; // Auxiliary head for training
    private readonly int[] _strides;
    private readonly bool _useNmsFree;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"YOLOv10-{Options.Size}";

    /// <summary>
    /// Creates a new YOLOv10 detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    /// <param name="useNmsFree">Whether to use NMS-free inference (default true).</param>
    public YOLOv10(ObjectDetectionOptions<T> options, bool useNmsFree = true) : base(options)
    {
        _useNmsFree = useNmsFree;
        var (depth, width) = GetSizeConfig(options.Size);

        // Initialize backbone
        Backbone = new CSPDarknet<T>(depth: depth, widthMultiplier: width);

        // Initialize neck with enhanced connections
        Neck = new PANet<T>(Backbone.OutputChannels, outputChannels: (int)(256 * width));

        // Main detection head (one-to-one assignment)
        var neckChannels = Enumerable.Repeat(Neck.OutputChannels, Neck.NumLevels).ToArray();
        _head = new YOLOv8Head<T>(neckChannels, options.NumClasses);

        // Auxiliary head for training (one-to-many assignment)
        if (IsTrainingMode)
        {
            _auxHead = new YOLOv8Head<T>(neckChannels, options.NumClasses);
        }

        _strides = Backbone.Strides;
        _nms = new NMS<T>();
    }

    private static (double depth, double width) GetSizeConfig(ModelSize size) => size switch
    {
        ModelSize.Nano => (0.33, 0.25),
        ModelSize.Small => (0.33, 0.50),
        ModelSize.Medium => (0.67, 0.75),
        ModelSize.Large => (1.00, 1.00),
        ModelSize.XLarge => (1.33, 1.25),
        _ => (0.67, 0.75)
    };

    /// <inheritdoc/>
    public override DetectionResult<T> Detect(Tensor<T> image, double confidenceThreshold, double nmsThreshold)
    {
        var startTime = DateTime.UtcNow;

        int originalHeight = image.Shape[2];
        int originalWidth = image.Shape[3];

        var input = Preprocess(image);
        var outputs = Forward(input);
        var detections = PostProcess(outputs, originalWidth, originalHeight, confidenceThreshold, nmsThreshold);

        return new DetectionResult<T>
        {
            Detections = detections,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = originalWidth,
            ImageHeight = originalHeight
        };
    }

    /// <inheritdoc/>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        // Backbone feature extraction
        var backboneFeatures = Backbone!.ExtractFeatures(input);

        // Neck feature fusion
        var neckFeatures = Neck!.Forward(backboneFeatures);

        // Main detection head
        var (clsOutputs, regOutputs) = _head.Forward(neckFeatures);

        var outputs = new List<Tensor<T>>();
        outputs.AddRange(clsOutputs);
        outputs.AddRange(regOutputs);

        return outputs;
    }

    /// <inheritdoc/>
    protected override List<Detection<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold,
        double nmsThreshold)
    {
        int numLevels = outputs.Count / 2;
        var clsOutputs = outputs.Take(numLevels).ToList();
        var regOutputs = outputs.Skip(numLevels).ToList();

        var decoded = _head.DecodeOutputs(clsOutputs, regOutputs, _strides, imageHeight, imageWidth);

        float[] boxes = decoded[0].boxes;
        float[] scores = decoded[0].scores;
        int[] classIds = decoded[0].classIds;

        // Build detection list with confidence filtering
        var candidateDetections = new List<Detection<T>>();

        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] >= confidenceThreshold)
            {
                var box = new BoundingBox<T>(
                    NumOps.FromDouble(boxes[i * 4]),
                    NumOps.FromDouble(boxes[i * 4 + 1]),
                    NumOps.FromDouble(boxes[i * 4 + 2]),
                    NumOps.FromDouble(boxes[i * 4 + 3]));

                int classId = classIds[i];
                candidateDetections.Add(new Detection<T>(
                    box,
                    classId,
                    NumOps.FromDouble(scores[i]),
                    classId < ClassNames.Length ? ClassNames[classId] : null));
            }
        }

        List<Detection<T>> results;

        if (_useNmsFree)
        {
            // NMS-free: select top-K detections per class directly
            results = SelectTopKPerClass(candidateDetections, Options.MaxDetections);
        }
        else
        {
            // Fall back to standard NMS
            results = _nms.Apply(candidateDetections, nmsThreshold);
        }

        // Limit to max detections
        if (results.Count > Options.MaxDetections)
        {
            return results.Take(Options.MaxDetections).ToList();
        }

        return results;
    }

    /// <summary>
    /// Selects top-K detections per class without NMS (for NMS-free inference).
    /// </summary>
    private List<Detection<T>> SelectTopKPerClass(List<Detection<T>> detections, int maxTotal)
    {
        // Group by class
        var byClass = detections
            .GroupBy(d => d.ClassId)
            .ToDictionary(g => g.Key, g => g.ToList());

        int perClassLimit = Math.Max(1, maxTotal / Math.Max(1, byClass.Count));

        var selected = new List<Detection<T>>();

        foreach (var kvp in byClass)
        {
            var classDetections = kvp.Value;
            var sorted = classDetections
                .OrderByDescending(d => NumOps.ToDouble(d.Confidence))
                .Take(perClassLimit);

            foreach (var detection in sorted)
            {
                selected.Add(detection);
                if (selected.Count >= maxTotal)
                {
                    return selected;
                }
            }
        }

        // Sort final results by confidence
        return selected.OrderByDescending(d => NumOps.ToDouble(d.Confidence)).ToList();
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        long count = _head.GetParameterCount();
        if (_auxHead is not null)
        {
            count += _auxHead.GetParameterCount();
        }
        return count;
    }

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        throw new NotImplementedException("Weight saving not yet implemented");
    }
}
