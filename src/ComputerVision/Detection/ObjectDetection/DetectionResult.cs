using AiDotNet.Augmentation.Image;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection;

/// <summary>
/// Contains the results of object detection on an image.
/// </summary>
/// <typeparam name="T">The numeric type used for coordinates and scores.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class holds all the objects detected in an image.
/// Each detection includes:
/// - A bounding box showing where the object is
/// - The class/category of the object (e.g., "person", "car")
/// - A confidence score indicating how sure the model is
/// </para>
/// </remarks>
public class DetectionResult<T>
{
    /// <summary>
    /// List of all detected objects in the image.
    /// </summary>
    public List<Detection<T>> Detections { get; set; } = new();

    /// <summary>
    /// Time taken to run inference.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Original image width in pixels.
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Original image height in pixels.
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Model name that produced these detections.
    /// </summary>
    public string? ModelName { get; set; }

    /// <summary>
    /// Gets the number of detections.
    /// </summary>
    public int Count => Detections.Count;

    /// <summary>
    /// Filters detections by confidence threshold.
    /// </summary>
    /// <param name="minConfidence">Minimum confidence score to keep.</param>
    /// <returns>A new DetectionResult with filtered detections.</returns>
    public DetectionResult<T> FilterByConfidence(double minConfidence)
    {
        var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var threshold = numOps.FromDouble(minConfidence);

        return new DetectionResult<T>
        {
            Detections = Detections.Where(d => numOps.GreaterThanOrEquals(d.Confidence, threshold)).ToList(),
            InferenceTime = InferenceTime,
            ImageWidth = ImageWidth,
            ImageHeight = ImageHeight,
            ModelName = ModelName
        };
    }

    /// <summary>
    /// Filters detections by class ID.
    /// </summary>
    /// <param name="classIds">Class IDs to keep.</param>
    /// <returns>A new DetectionResult with filtered detections.</returns>
    public DetectionResult<T> FilterByClass(params int[] classIds)
    {
        var classSet = new HashSet<int>(classIds);
        return new DetectionResult<T>
        {
            Detections = Detections.Where(d => classSet.Contains(d.ClassId)).ToList(),
            InferenceTime = InferenceTime,
            ImageWidth = ImageWidth,
            ImageHeight = ImageHeight,
            ModelName = ModelName
        };
    }

    /// <summary>
    /// Gets detections sorted by confidence (highest first).
    /// </summary>
    /// <returns>A new DetectionResult with sorted detections.</returns>
    public DetectionResult<T> SortByConfidence()
    {
        var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new DetectionResult<T>
        {
            Detections = Detections.OrderByDescending(d => numOps.ToDouble(d.Confidence)).ToList(),
            InferenceTime = InferenceTime,
            ImageWidth = ImageWidth,
            ImageHeight = ImageHeight,
            ModelName = ModelName
        };
    }

    /// <summary>
    /// Gets the top N detections by confidence.
    /// </summary>
    /// <param name="n">Maximum number of detections to return.</param>
    /// <returns>A new DetectionResult with top N detections.</returns>
    public DetectionResult<T> TopN(int n)
    {
        return new DetectionResult<T>
        {
            Detections = SortByConfidence().Detections.Take(n).ToList(),
            InferenceTime = InferenceTime,
            ImageWidth = ImageWidth,
            ImageHeight = ImageHeight,
            ModelName = ModelName
        };
    }
}

/// <summary>
/// Represents a single detected object.
/// </summary>
/// <typeparam name="T">The numeric type used for coordinates and scores.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This represents one object found in the image.
/// It tells you:
/// - Where the object is (bounding box)
/// - What type of object it is (class ID and name)
/// - How confident the model is (0.0 to 1.0)
/// - Optionally, a pixel mask for instance segmentation
/// </para>
/// </remarks>
public class Detection<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new detection with default numeric operations.
    /// </summary>
    public Detection()
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        Box = new BoundingBox<T>();
        Confidence = _numOps.Zero;
    }

    /// <summary>
    /// Creates a new detection with specified values.
    /// </summary>
    /// <param name="box">The bounding box.</param>
    /// <param name="classId">The class ID.</param>
    /// <param name="confidence">The confidence score.</param>
    /// <param name="className">Optional class name.</param>
    public Detection(BoundingBox<T> box, int classId, T confidence, string? className = null)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        Box = box;
        ClassId = classId;
        Confidence = confidence;
        ClassName = className;
    }

    /// <summary>
    /// The bounding box defining the object's location.
    /// </summary>
    public BoundingBox<T> Box { get; set; }

    /// <summary>
    /// The class ID (index) of the detected object.
    /// </summary>
    public int ClassId { get; set; }

    /// <summary>
    /// The human-readable name of the class (e.g., "person", "car").
    /// </summary>
    public string? ClassName { get; set; }

    /// <summary>
    /// Confidence score from 0 to 1 indicating detection certainty.
    /// </summary>
    public T Confidence { get; set; }

    /// <summary>
    /// Optional instance segmentation mask for this object.
    /// </summary>
    /// <remarks>
    /// <para>Only populated when using instance segmentation models like Mask R-CNN or YOLOv8-Seg.</para>
    /// </remarks>
    public SegmentationMask<T>? Mask { get; set; }

    /// <summary>
    /// Optional keypoints for pose estimation.
    /// </summary>
    public List<Keypoint<T>>? Keypoints { get; set; }

    /// <summary>
    /// Optional track ID for object tracking across frames.
    /// </summary>
    public int? TrackId { get; set; }

    /// <summary>
    /// Gets the area of the bounding box.
    /// </summary>
    public double Area => Box.Area();

    /// <summary>
    /// Gets the center X coordinate of the bounding box.
    /// </summary>
    public double CenterX
    {
        get
        {
            var (cx, _, _, _) = Box.ToCXCYWH();
            return cx;
        }
    }

    /// <summary>
    /// Gets the center Y coordinate of the bounding box.
    /// </summary>
    public double CenterY
    {
        get
        {
            var (_, cy, _, _) = Box.ToCXCYWH();
            return cy;
        }
    }

    /// <summary>
    /// Gets a string representation of this detection.
    /// </summary>
    public override string ToString()
    {
        var confPercent = _numOps.ToDouble(Confidence) * 100;
        var name = ClassName ?? $"class_{ClassId}";
        return $"{name}: {confPercent:F1}% at ({Box.X1}, {Box.Y1}, {Box.X2}, {Box.Y2})";
    }
}

/// <summary>
/// Represents a batch of detection results for multiple images.
/// </summary>
/// <typeparam name="T">The numeric type used for coordinates and scores.</typeparam>
public class BatchDetectionResult<T>
{
    /// <summary>
    /// Detection results for each image in the batch.
    /// </summary>
    public List<DetectionResult<T>> Results { get; set; } = new();

    /// <summary>
    /// Total time to process the entire batch.
    /// </summary>
    public TimeSpan TotalInferenceTime { get; set; }

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize => Results.Count;

    /// <summary>
    /// Gets the total number of detections across all images.
    /// </summary>
    public int TotalDetections => Results.Sum(r => r.Count);

    /// <summary>
    /// Gets the average inference time per image.
    /// </summary>
    public TimeSpan AverageInferenceTime => BatchSize > 0
        ? TimeSpan.FromTicks(TotalInferenceTime.Ticks / BatchSize)
        : TimeSpan.Zero;

    /// <summary>
    /// Gets the result for a specific image index.
    /// </summary>
    /// <param name="index">The image index.</param>
    /// <returns>The detection result for that image.</returns>
    public DetectionResult<T> this[int index] => Results[index];
}

/// <summary>
/// Statistics about detection results.
/// </summary>
/// <typeparam name="T">The numeric type used for coordinates and scores.</typeparam>
public class DetectionStatistics<T>
{
    /// <summary>
    /// Number of detections per class.
    /// </summary>
    public Dictionary<int, int> CountByClass { get; set; } = new();

    /// <summary>
    /// Average confidence per class.
    /// </summary>
    public Dictionary<int, double> AverageConfidenceByClass { get; set; } = new();

    /// <summary>
    /// Total number of detections.
    /// </summary>
    public int TotalDetections { get; set; }

    /// <summary>
    /// Overall average confidence.
    /// </summary>
    public double AverageConfidence { get; set; }

    /// <summary>
    /// Computes statistics from a detection result.
    /// </summary>
    /// <param name="result">The detection result to analyze.</param>
    /// <returns>Statistics about the detections.</returns>
    public static DetectionStatistics<T> FromResult(DetectionResult<T> result)
    {
        var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var stats = new DetectionStatistics<T>
        {
            TotalDetections = result.Count
        };

        if (result.Count == 0) return stats;

        // Group by class
        var grouped = result.Detections.GroupBy(d => d.ClassId);
        foreach (var group in grouped)
        {
            stats.CountByClass[group.Key] = group.Count();
            stats.AverageConfidenceByClass[group.Key] = group.Average(d => numOps.ToDouble(d.Confidence));
        }

        stats.AverageConfidence = result.Detections.Average(d => numOps.ToDouble(d.Confidence));

        return stats;
    }
}
