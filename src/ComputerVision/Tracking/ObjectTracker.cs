using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
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
    public double VelocityX { get; set; }

    /// <summary>
    /// Velocity in y direction (pixels/frame).
    /// </summary>
    public double VelocityY { get; set; }

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
    /// Weight for appearance similarity in cost matrix.
    /// </summary>
    public double AppearanceWeight { get; set; } = 0.5;

    /// <summary>
    /// Maximum cosine distance for appearance matching.
    /// </summary>
    public double MaxCosineDistance { get; set; } = 0.3;
}

/// <summary>
/// Base class for object trackers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
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
    /// Computes IoU between two bounding boxes.
    /// </summary>
    protected double ComputeIoU(BoundingBox<T> a, BoundingBox<T> b)
    {
        double x1 = Math.Max(NumOps.ToDouble(a.X1), NumOps.ToDouble(b.X1));
        double y1 = Math.Max(NumOps.ToDouble(a.Y1), NumOps.ToDouble(b.Y1));
        double x2 = Math.Min(NumOps.ToDouble(a.X2), NumOps.ToDouble(b.X2));
        double y2 = Math.Min(NumOps.ToDouble(a.Y2), NumOps.ToDouble(b.Y2));

        double intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);

        double areaA = (NumOps.ToDouble(a.X2) - NumOps.ToDouble(a.X1)) *
                       (NumOps.ToDouble(a.Y2) - NumOps.ToDouble(a.Y1));
        double areaB = (NumOps.ToDouble(b.X2) - NumOps.ToDouble(b.X1)) *
                       (NumOps.ToDouble(b.Y2) - NumOps.ToDouble(b.Y1));

        double union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0;
    }

    /// <summary>
    /// Computes cosine similarity between two feature vectors.
    /// </summary>
    protected double ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length)
            return 0;

        double dot = 0, normA = 0, normB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            double va = NumOps.ToDouble(a[i]);
            double vb = NumOps.ToDouble(b[i]);
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        if (normA <= 0 || normB <= 0)
            return 0;

        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    /// <summary>
    /// Solves linear assignment problem (Hungarian algorithm).
    /// </summary>
    protected int[] HungarianAssignment(double[,] costMatrix)
    {
        int n = costMatrix.GetLength(0);
        int m = costMatrix.GetLength(1);

        if (n == 0 || m == 0)
            return Array.Empty<int>();

        // Simple greedy assignment (approximate Hungarian)
        var assignment = new int[n];
        for (int i = 0; i < n; i++)
            assignment[i] = -1;

        var usedCols = new bool[m];

        for (int i = 0; i < n; i++)
        {
            int bestCol = -1;
            double bestCost = double.MaxValue;

            for (int j = 0; j < m; j++)
            {
                if (!usedCols[j] && costMatrix[i, j] < bestCost)
                {
                    bestCost = costMatrix[i, j];
                    bestCol = j;
                }
            }

            if (bestCol >= 0 && bestCost < 1.0) // Cost threshold
            {
                assignment[i] = bestCol;
                usedCols[bestCol] = true;
            }
        }

        return assignment;
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
        // Update velocity
        double oldCx = (NumOps.ToDouble(track.Box.X1) + NumOps.ToDouble(track.Box.X2)) / 2;
        double oldCy = (NumOps.ToDouble(track.Box.Y1) + NumOps.ToDouble(track.Box.Y2)) / 2;
        double newCx = (NumOps.ToDouble(detection.Box.X1) + NumOps.ToDouble(detection.Box.X2)) / 2;
        double newCy = (NumOps.ToDouble(detection.Box.Y1) + NumOps.ToDouble(detection.Box.Y2)) / 2;

        track.VelocityX = 0.7 * track.VelocityX + 0.3 * (newCx - oldCx);
        track.VelocityY = 0.7 * track.VelocityY + 0.3 * (newCy - oldCy);

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
        double x1 = NumOps.ToDouble(track.Box.X1) + track.VelocityX;
        double y1 = NumOps.ToDouble(track.Box.Y1) + track.VelocityY;
        double x2 = NumOps.ToDouble(track.Box.X2) + track.VelocityX;
        double y2 = NumOps.ToDouble(track.Box.Y2) + track.VelocityY;

        return new BoundingBox<T>(
            NumOps.FromDouble(x1), NumOps.FromDouble(y1),
            NumOps.FromDouble(x2), NumOps.FromDouble(y2),
            BoundingBoxFormat.XYXY);
    }
}
