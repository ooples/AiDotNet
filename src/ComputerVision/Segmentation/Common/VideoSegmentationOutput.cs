namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Output for video segmentation containing per-frame masks with temporal tracking.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Video segmentation tracks objects across frames. This output
/// contains masks for each frame along with tracking IDs that let you know which
/// object in frame 1 corresponds to which object in frame 50.
/// </para>
/// </remarks>
public class VideoSegmentationOutput<T>
{
    /// <summary>
    /// Per-frame segmentation results.
    /// </summary>
    public List<VideoFrameSegmentation<T>> Frames { get; set; } = [];

    /// <summary>
    /// Total number of tracked objects across all frames.
    /// </summary>
    public int TotalTrackedObjects { get; set; }

    /// <summary>
    /// Total number of frames processed.
    /// </summary>
    public int TotalFrames => Frames.Count;

    /// <summary>
    /// Total inference time for all frames.
    /// </summary>
    public TimeSpan TotalInferenceTime { get; set; }

    /// <summary>
    /// Average frames per second achieved.
    /// </summary>
    public double AverageFps => TotalFrames > 0 && TotalInferenceTime.TotalSeconds > 0
        ? TotalFrames / TotalInferenceTime.TotalSeconds
        : 0;

    /// <summary>
    /// Per-object tracking summaries across the video.
    /// </summary>
    public List<ObjectTrackSummary> TrackSummaries { get; set; } = [];
}

/// <summary>
/// Segmentation result for a single video frame.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VideoFrameSegmentation<T>
{
    /// <summary>
    /// Frame index in the video sequence.
    /// </summary>
    public int FrameIndex { get; set; }

    /// <summary>
    /// Per-object masks for this frame.
    /// </summary>
    public List<SegmentationMask<T>> Masks { get; set; } = [];

    /// <summary>
    /// Combined semantic map for this frame [H, W] (if applicable).
    /// </summary>
    public Tensor<T>? SemanticMap { get; set; }

    /// <summary>
    /// Inference time for this frame.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Number of visible objects in this frame.
    /// </summary>
    public int VisibleObjectCount => Masks.Count;

    /// <summary>
    /// Timestamp of this frame in the video (seconds).
    /// </summary>
    public double Timestamp { get; set; }
}

/// <summary>
/// Summary of an object's track across a video sequence.
/// </summary>
public class ObjectTrackSummary
{
    /// <summary>
    /// Unique tracking ID for this object.
    /// </summary>
    public int TrackingId { get; set; }

    /// <summary>
    /// Class ID of the tracked object.
    /// </summary>
    public int ClassId { get; set; }

    /// <summary>
    /// Class name (if available).
    /// </summary>
    public string? ClassName { get; set; }

    /// <summary>
    /// First frame where this object appears.
    /// </summary>
    public int FirstFrame { get; set; }

    /// <summary>
    /// Last frame where this object appears.
    /// </summary>
    public int LastFrame { get; set; }

    /// <summary>
    /// Total number of frames where this object is visible.
    /// </summary>
    public int VisibleFrameCount { get; set; }

    /// <summary>
    /// Average confidence score across all visible frames.
    /// </summary>
    public double AverageScore { get; set; }

    /// <summary>
    /// Average mask area across all visible frames (in pixels).
    /// </summary>
    public double AverageArea { get; set; }

    /// <summary>
    /// Whether this object is currently being tracked.
    /// </summary>
    public bool IsActive { get; set; }
}
