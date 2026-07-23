namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for video segmentation models that track and segment objects across video frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Video segmentation extends image segmentation to temporal sequences by tracking objects
/// across frames. Models maintain memory of previously seen objects to ensure consistent
/// segmentation throughout the video.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video segmentation is like image segmentation but for videos.
/// The key challenge is tracking: once you segment a car in frame 1, you need to keep
/// tracking that same car through all subsequent frames, even when it moves, gets
/// partially hidden, or changes appearance.
///
/// Types of video segmentation:
/// - Semi-supervised VOS: You provide masks on the first frame, model tracks through video
/// - Unsupervised VOS: Model automatically finds and tracks salient objects
/// - Video Instance Segmentation (VIS): Detect + segment + track all instances per frame
/// - Video Panoptic Segmentation (VPS): Panoptic segmentation with tracking
///
/// Models implementing this interface:
/// - SAM 2 (Meta, streaming memory architecture)
/// - Cutie (CVPR 2024, object-level memory)
/// - XMem (ECCV 2022, three-level memory)
/// - DEVA (ICCV 2023, decoupled propagation)
/// - EfficientTAM (lightweight, mobile-ready)
/// - UniVS (CVPR 2024, universal video segmentation)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("VideoSegmentation")]
public interface IVideoSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets the maximum number of objects that can be tracked simultaneously.
    /// </summary>
    int MaxTrackedObjects { get; }

    /// <summary>
    /// Gets whether the model supports streaming (frame-by-frame) processing.
    /// </summary>
    bool SupportsStreaming { get; }

    /// <summary>
    /// Initializes tracking with masks on the first frame.
    /// </summary>
    /// <param name="frame">First video frame tensor [C, H, W].</param>
    /// <param name="masks">Initial object masks [numObjects, H, W].</param>
    /// <param name="objectIds">Optional unique IDs for each object.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells the model "here are the objects I want you to track"
    /// by showing it the first frame with masks drawn on each object.
    /// </para>
    /// </remarks>
    void InitializeTracking(Tensor<T> frame, Tensor<T> masks, int[]? objectIds = null);

    /// <summary>
    /// Propagates segmentation masks to the next frame.
    /// </summary>
    /// <param name="frame">Next video frame tensor [C, H, W].</param>
    /// <returns>Predicted masks and tracking state for this frame.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After initializing with the first frame, call this for each
    /// subsequent frame. The model uses its memory of previous frames to predict where
    /// each object has moved.
    /// </para>
    /// </remarks>
    VideoSegmentationResult<T> PropagateToFrame(Tensor<T> frame);

    /// <summary>
    /// Adds a correction mask for an object at the current frame.
    /// </summary>
    /// <param name="objectId">ID of the object to correct.</param>
    /// <param name="correctionMask">Corrected mask for this object [H, W].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If the model makes a mistake tracking an object, you can
    /// provide a corrected mask. The model incorporates this correction into its memory
    /// so future frames benefit from the fix.
    /// </para>
    /// </remarks>
    void AddCorrection(int objectId, Tensor<T> correctionMask);

    /// <summary>
    /// Resets the tracking state and memory.
    /// </summary>
    void ResetTracking();
}

/// <summary>
/// Result of video segmentation for a single frame.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VideoSegmentationResult<T>
{
    /// <summary>
    /// Per-object masks [numObjects, H, W].
    /// </summary>
    public Tensor<T> Masks { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Object IDs corresponding to each mask.
    /// </summary>
    public int[] ObjectIds { get; set; } = [];

    /// <summary>
    /// Confidence scores for each tracked object.
    /// </summary>
    public double[] Confidences { get; set; } = [];

    /// <summary>
    /// Frame index in the video sequence.
    /// </summary>
    public int FrameIndex { get; set; }

    /// <summary>
    /// Whether each object is considered visible (not fully occluded) in this frame.
    /// </summary>
    public bool[] IsVisible { get; set; } = [];
}
