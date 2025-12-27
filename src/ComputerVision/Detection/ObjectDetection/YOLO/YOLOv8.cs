using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// YOLOv8 object detector - anchor-free, decoupled head architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> YOLOv8 is a state-of-the-art real-time object detector
/// developed by Ultralytics. It's faster and more accurate than previous YOLO versions,
/// using an anchor-free design that simplifies training and improves generalization.</para>
///
/// <para>Key improvements over YOLOv5:
/// - Anchor-free design eliminates anchor tuning
/// - Decoupled head separates classification and localization
/// - Distribution Focal Loss (DFL) for better box regression
/// - C2f modules for improved feature extraction
/// </para>
///
/// <para>Reference: Jocher et al., "YOLOv8" Ultralytics, 2023</para>
/// </remarks>
public class YOLOv8<T> : ObjectDetectorBase<T>
{
    private readonly YOLOv8Head<T> _head;
    private readonly int[] _strides;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"YOLOv8-{Options.Size}";

    /// <summary>
    /// Creates a new YOLOv8 detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public YOLOv8(ObjectDetectionOptions<T> options) : base(options)
    {
        // Get configuration based on model size
        var (depth, width) = GetSizeConfig(options.Size);

        // Initialize backbone
        Backbone = new CSPDarknet<T>(depth: depth, widthMultiplier: width);

        // Initialize neck
        Neck = new PANet<T>(Backbone.OutputChannels, outputChannels: (int)(256 * width));

        // Initialize detection head
        var neckChannels = Enumerable.Repeat(Neck.OutputChannels, Neck.NumLevels).ToArray();
        _head = new YOLOv8Head<T>(neckChannels, options.NumClasses);

        _strides = Backbone.Strides;
        _nms = new NMS<T>();
    }

    /// <summary>
    /// Gets depth and width multipliers for each model size.
    /// </summary>
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

        // Preprocess
        var input = Preprocess(image);

        // Forward pass
        var outputs = Forward(input);

        // Post-process
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
        // Extract features through backbone
        var backboneFeatures = Backbone!.ExtractFeatures(input);

        // Fuse features through neck
        var neckFeatures = Neck!.Forward(backboneFeatures);

        // Get detection outputs from head
        var (clsOutputs, regOutputs) = _head.Forward(neckFeatures);

        // Combine outputs for post-processing
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
        // Split outputs back into classification and regression
        int numLevels = outputs.Count / 2;
        var clsOutputs = outputs.Take(numLevels).ToList();
        var regOutputs = outputs.Skip(numLevels).ToList();

        // Decode outputs
        var decoded = _head.DecodeOutputs(clsOutputs, regOutputs, _strides, imageHeight, imageWidth);

        // Get boxes, scores, class IDs
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

        // Apply NMS
        var nmsResults = _nms.Apply(candidateDetections, nmsThreshold);

        // Limit to max detections
        if (nmsResults.Count > Options.MaxDetections)
        {
            return nmsResults.Take(Options.MaxDetections).ToList();
        }

        return nmsResults;
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        return _head.GetParameterCount();
    }

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        // Weight loading implementation would go here
        // For now, return completed task as weights are randomly initialized
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        // Weight saving implementation would go here
        throw new NotImplementedException("Weight saving not yet implemented");
    }
}
