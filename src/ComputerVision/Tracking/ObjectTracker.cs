using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Helpers;
using AiDotNet.Solvers.Assignment;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Tracking;

/// <summary>
/// Represents a tracked object across frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class Track<T>
{
    /// <summary>
    /// Unique track identifier.
    /// </summary>
    public int TrackId { get; set; }

    /// <summary>
    /// Current bounding box.
    /// </summary>
    public BoundingBox<T> Box { get; set; }

    /// <summary>
    /// Class ID of the tracked object.
    /// </summary>
    public int ClassId { get; set; }

    /// <summary>
    /// Current confidence score.
    /// </summary>
    public T Confidence { get; set; }

    /// <summary>
    /// Velocity in x direction (pixels/frame).
    /// </summary>
    public T VelocityX { get; set; }

    /// <summary>
    /// Velocity in y direction (pixels/frame).
    /// </summary>
    public T VelocityY { get; set; }

    /// <summary>
    /// Number of consecutive frames without detection.
    /// </summary>
    public int TimeSinceUpdate { get; set; }

    /// <summary>
    /// Total number of frames this track has been active.
    /// </summary>
    public int Age { get; set; }

    /// <summary>
    /// Number of frames with detection hits.
    /// </summary>
    public int Hits { get; set; }

    /// <summary>
    /// Track state (tentative, confirmed, deleted).
    /// </summary>
    public TrackState State { get; set; }

    /// <summary>
    /// Optional appearance feature embedding.
    /// </summary>
    public Tensor<T>? AppearanceFeature { get; set; }

    /// <summary>
    /// Creates a new track.
    /// </summary>
    public Track(int trackId, BoundingBox<T> box, int classId, T confidence)
    {
        TrackId = trackId;
        Box = box;
        ClassId = classId;
        Confidence = confidence;
        var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        VelocityX = numOps.Zero;
        VelocityY = numOps.Zero;
        State = TrackState.Tentative;
        Hits = 1;
        Age = 1;
    }
}

/// <summary>
/// Track lifecycle states.
/// </summary>
public enum TrackState
{
    /// <summary>Newly created, not yet confirmed.</summary>
    Tentative,
    /// <summary>Confirmed active track.</summary>
    Confirmed,
    /// <summary>Temporarily lost, can be reactivated.</summary>
    Lost,
    /// <summary>Marked for deletion.</summary>
    Deleted
}

/// <summary>
/// Result of tracking on a frame.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TrackingResult<T>
{
    /// <summary>
    /// Active tracks in this frame.
    /// </summary>
    public List<Track<T>> Tracks { get; set; } = new();

    /// <summary>
    /// Frame number.
    /// </summary>
    public int FrameNumber { get; set; }

    /// <summary>
    /// Time taken for tracking.
    /// </summary>
    public TimeSpan TrackingTime { get; set; }
}

/// <summary>
/// Options for object tracking.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TrackingOptions<T>
{
    private static readonly INumericOperations<T> NumOps =
        Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Maximum age (frames without detection) before track deletion.
    /// </summary>
    public int MaxAge { get; set; } = 30;

    /// <summary>
    /// Minimum hits before track is confirmed.
    /// </summary>
    public int MinHits { get; set; } = 3;

    /// <summary>
    /// IoU threshold for detection-to-track association.
    /// </summary>
    public T IouThreshold { get; set; } = NumOps.FromDouble(0.3);

    /// <summary>
    /// Confidence threshold for detections.
    /// </summary>
    public T ConfidenceThreshold { get; set; } = NumOps.FromDouble(0.5);

    /// <summary>
    /// Whether to use appearance features for matching.
    /// </summary>
    public bool UseAppearance { get; set; } = false;

    /// <summary>
    /// Weight of the appearance (cosine) distance in DeepSORT's association cost; the motion
    /// (Mahalanobis) distance gets the rest. This is <c>1 - lambda</c> in Wojke et al. (2017), Eq. 5.
    /// </summary>
    /// <remarks>The paper's experiments and the official code use lambda = 0, an appearance weight of 1.</remarks>
    public T AppearanceWeight { get; set; } = NumOps.One;

    /// <summary>
    /// Maximum cosine distance for appearance matching (the official deep_sort default is 0.2).
    /// </summary>
    public T MaxCosineDistance { get; set; } = NumOps.FromDouble(0.2);
}

/// <summary>
/// Base class for object trackers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ObjectTracker")]
public abstract class ObjectTrackerBase<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly TrackingOptions<T> Options;
    protected readonly List<Track<T>> Tracks;
    protected int NextTrackId;
    protected int FrameCount;

    /// <summary>
    /// Name of this tracker.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Creates a new object tracker.
    /// </summary>
    protected ObjectTrackerBase(TrackingOptions<T> options)
    {
        NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        Options = options;
        Tracks = new List<Track<T>>();
        NextTrackId = 1;
    }

    /// <summary>
    /// Updates tracks with new detections.
    /// </summary>
    /// <param name="detections">Detections from the current frame.</param>
    /// <returns>Tracking result with updated tracks.</returns>
    public abstract TrackingResult<T> Update(List<Detection<T>> detections);

    /// <summary>
    /// Updates tracks with new detections and image for appearance features.
    /// </summary>
    /// <param name="detections">Detections from the current frame.</param>
    /// <param name="image">Current frame image.</param>
    /// <returns>Tracking result with updated tracks.</returns>
    public virtual TrackingResult<T> Update(List<Detection<T>> detections, Tensor<T> image)
    {
        return Update(detections);
    }

    /// <summary>
    /// Resets the tracker state.
    /// </summary>
    public virtual void Reset()
    {
        Tracks.Clear();
        NextTrackId = 1;
        FrameCount = 0;
    }

    /// <summary>
    /// Gets confirmed tracks only.
    /// </summary>
    public List<Track<T>> GetConfirmedTracks()
    {
        return Tracks.Where(t => t.State == TrackState.Confirmed).ToList();
    }

    /// <summary>
    /// Computes IoU between two bounding boxes given as (x1, y1, x2, y2).
    /// </summary>
    protected T ComputeIoU(BoundingBox<T> a, BoundingBox<T> b)
    {
        var iou = AiDotNetEngine.Current.BoxIou(
            TrackingAssociation<T>.BoxesToTensor(new[] { a }),
            TrackingAssociation<T>.BoxesToTensor(new[] { b }));
        return iou[0, 0];
    }

    /// <summary>
    /// Computes cosine similarity between two feature vectors (0 when their lengths differ).
    /// </summary>
    protected T ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length)
            return NumOps.Zero;

        return AiDotNetEngine.Current.CosineSimilarity(a.ToVector(), b.ToVector());
    }

    /// <summary>
    /// Solves the linear assignment problem exactly (Hungarian algorithm): the returned array gives
    /// the column assigned to each row of <paramref name="costMatrix"/>, or -1 when the row is left
    /// unmatched (only possible when there are more rows than columns).
    /// </summary>
    protected int[] HungarianAssignment(Matrix<T> costMatrix)
    {
        return TrackingAssociation<T>.SolveAssignment(Tensor<T>.FromMatrix(costMatrix));
    }

    /// <summary>
    /// Creates a new track from detection.
    /// </summary>
    protected Track<T> CreateTrack(Detection<T> detection)
    {
        var track = new Track<T>(NextTrackId++, detection.Box, detection.ClassId, detection.Confidence);
        return track;
    }

    /// <summary>
    /// Updates a track with a matched detection.
    /// </summary>
    protected void UpdateTrack(Track<T> track, Detection<T> detection)
    {
        // Velocity: an exponential average (0.7 / 0.3) of the centre displacement.
        T half = NumOps.FromDouble(0.5);
        T keep = NumOps.FromDouble(0.7);
        T blend = NumOps.FromDouble(0.3);
        T oldCx = NumOps.Multiply(NumOps.Add(track.Box.X1, track.Box.X2), half);
        T oldCy = NumOps.Multiply(NumOps.Add(track.Box.Y1, track.Box.Y2), half);
        T newCx = NumOps.Multiply(NumOps.Add(detection.Box.X1, detection.Box.X2), half);
        T newCy = NumOps.Multiply(NumOps.Add(detection.Box.Y1, detection.Box.Y2), half);

        track.VelocityX = NumOps.Add(NumOps.Multiply(keep, track.VelocityX),
            NumOps.Multiply(blend, NumOps.Subtract(newCx, oldCx)));
        track.VelocityY = NumOps.Add(NumOps.Multiply(keep, track.VelocityY),
            NumOps.Multiply(blend, NumOps.Subtract(newCy, oldCy)));

        // Update box and confidence
        track.Box = detection.Box;
        track.Confidence = detection.Confidence;
        track.TimeSinceUpdate = 0;
        track.Hits++;
        track.Age++;

        // Confirm track after enough hits
        if (track.State == TrackState.Tentative && track.Hits >= Options.MinHits)
        {
            track.State = TrackState.Confirmed;
        }
    }

    /// <summary>
    /// Predicts track position for next frame using velocity.
    /// </summary>
    protected BoundingBox<T> PredictNextPosition(Track<T> track)
    {
        return new BoundingBox<T>(
            NumOps.Add(track.Box.X1, track.VelocityX), NumOps.Add(track.Box.Y1, track.VelocityY),
            NumOps.Add(track.Box.X2, track.VelocityX), NumOps.Add(track.Box.Y2, track.VelocityY),
            BoundingBoxFormat.XYXY);
    }
}
