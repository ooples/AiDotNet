using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Tracking;

/// <summary>
/// SORT (Simple Online and Realtime Tracking) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SORT is a simple yet effective online tracking algorithm
/// that uses Kalman filtering for state estimation and the Hungarian algorithm for
/// detection-to-track association based on IoU overlap.</para>
///
/// <para>Key features:
/// - Kalman filter for motion prediction
/// - IoU-based association using Hungarian algorithm
/// - No appearance features (pure motion-based)
/// - Real-time performance (~260 FPS)
/// </para>
///
/// <para>Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016</para>
/// </remarks>
public class SORT<T> : ObjectTrackerBase<T>
{
    private readonly List<KalmanTrack<T>> _kalmanTracks;

    /// <inheritdoc/>
    public override string Name => "SORT";

    /// <summary>
    /// Creates a new SORT tracker.
    /// </summary>
    public SORT(TrackingOptions<T> options) : base(options)
    {
        _kalmanTracks = new List<KalmanTrack<T>>();
    }

    /// <inheritdoc/>
    public override TrackingResult<T> Update(List<Detection<T>> detections)
    {
        var startTime = DateTime.UtcNow;
        FrameCount++;

        // Filter detections by confidence
        var filteredDetections = detections
            .Where(d => NumOps.ToDouble(d.Confidence) >= NumOps.ToDouble(Options.ConfidenceThreshold))
            .ToList();

        // Predict new locations of existing tracks
        foreach (var track in _kalmanTracks)
        {
            track.Predict();
        }

        // Build cost matrix based on IoU
        var costMatrix = BuildIoUCostMatrix(filteredDetections);

        // Solve assignment problem
        var assignment = HungarianAssignment(costMatrix);

        // Update matched tracks
        var matchedDetections = new HashSet<int>();
        var matchedTracks = new HashSet<int>();

        for (int trackIdx = 0; trackIdx < assignment.Length; trackIdx++)
        {
            int detIdx = assignment[trackIdx];
            if (detIdx >= 0)
            {
                double iou = 1.0 - costMatrix[trackIdx, detIdx];
                if (iou >= NumOps.ToDouble(Options.IouThreshold))
                {
                    _kalmanTracks[trackIdx].Update(filteredDetections[detIdx]);
                    matchedDetections.Add(detIdx);
                    matchedTracks.Add(trackIdx);
                }
            }
        }

        // Handle unmatched tracks (mark as missing)
        for (int i = 0; i < _kalmanTracks.Count; i++)
        {
            if (!matchedTracks.Contains(i))
            {
                _kalmanTracks[i].MarkMissed();
            }
        }

        // Create new tracks for unmatched detections
        for (int i = 0; i < filteredDetections.Count; i++)
        {
            if (!matchedDetections.Contains(i))
            {
                var newTrack = new KalmanTrack<T>(
                    NextTrackId++,
                    filteredDetections[i],
                    NumOps);
                _kalmanTracks.Add(newTrack);
            }
        }

        // Remove dead tracks
        _kalmanTracks.RemoveAll(t => t.TimeSinceUpdate > Options.MaxAge);

        // Update base class tracks list
        Tracks.Clear();
        foreach (var kt in _kalmanTracks)
        {
            if (kt.Hits >= Options.MinHits || FrameCount <= Options.MinHits)
            {
                Tracks.Add(kt.ToTrack());
            }
        }

        return new TrackingResult<T>
        {
            Tracks = GetConfirmedTracks(),
            FrameNumber = FrameCount,
            TrackingTime = DateTime.UtcNow - startTime
        };
    }

    private double[,] BuildIoUCostMatrix(List<Detection<T>> detections)
    {
        int numTracks = _kalmanTracks.Count;
        int numDets = detections.Count;

        var costMatrix = new double[numTracks, numDets];

        for (int t = 0; t < numTracks; t++)
        {
            var predictedBox = _kalmanTracks[t].GetPredictedBox();

            for (int d = 0; d < numDets; d++)
            {
                double iou = ComputeIoU(predictedBox, detections[d].Box);
                costMatrix[t, d] = 1.0 - iou; // Convert to cost
            }
        }

        return costMatrix;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _kalmanTracks.Clear();
    }
}

/// <summary>
/// Kalman filter-based track for SORT.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class KalmanTrack<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _trackId;
    private int _classId;
    private T _confidence;

    // Kalman state: [x, y, s, r, vx, vy, vs]
    // x, y = center position
    // s = area (w*h)
    // r = aspect ratio (w/h)
    // vx, vy, vs = velocities
    private double[] _state;
    private double[,] _covariance;

    // Process and measurement noise
    private readonly double[,] _processNoise;
    private readonly double[,] _measurementNoise;

    public int TimeSinceUpdate { get; private set; }
    public int Hits { get; private set; }
    public int Age { get; private set; }
    public TrackState State { get; private set; }

    public KalmanTrack(int trackId, Detection<T> detection, INumericOperations<T> numOps)
    {
        _trackId = trackId;
        _numOps = numOps;
        _classId = detection.ClassId;
        _confidence = detection.Confidence;

        // Initialize state from detection
        var (cx, cy, s, r) = BoxToState(detection.Box);
        _state = new double[] { cx, cy, s, r, 0, 0, 0 };

        // Initialize covariance (high uncertainty for velocities)
        _covariance = new double[7, 7];
        for (int i = 0; i < 7; i++)
        {
            _covariance[i, i] = i < 4 ? 10 : 1000;
        }

        // Process noise
        _processNoise = new double[7, 7];
        for (int i = 0; i < 7; i++)
        {
            _processNoise[i, i] = i < 4 ? 1 : 0.01;
        }

        // Measurement noise
        _measurementNoise = new double[4, 4];
        for (int i = 0; i < 4; i++)
        {
            _measurementNoise[i, i] = 1;
        }

        TimeSinceUpdate = 0;
        Hits = 1;
        Age = 1;
        State = TrackState.Tentative;
    }

    public void Predict()
    {
        // State transition: x' = Fx
        // F is identity with velocity integration
        _state[0] += _state[4]; // x += vx
        _state[1] += _state[5]; // y += vy
        _state[2] += _state[6]; // s += vs

        // Ensure positive area
        if (_state[2] <= 0)
            _state[2] = 1;

        // Update covariance: P' = FPF' + Q
        // Simplified: just add process noise
        for (int i = 0; i < 7; i++)
        {
            _covariance[i, i] += _processNoise[i, i];
        }

        Age++;
        TimeSinceUpdate++;
    }

    public void Update(Detection<T> detection)
    {
        var (cx, cy, s, r) = BoxToState(detection.Box);
        double[] measurement = { cx, cy, s, r };

        // Kalman gain (simplified)
        double[] innovation = new double[4];
        for (int i = 0; i < 4; i++)
        {
            innovation[i] = measurement[i] - _state[i];
        }

        // Update state with measurement
        double alpha = 0.7; // Smoothing factor
        for (int i = 0; i < 4; i++)
        {
            _state[i] += alpha * innovation[i];
        }

        // Update velocities based on position change
        _state[4] = 0.5 * _state[4] + 0.5 * innovation[0];
        _state[5] = 0.5 * _state[5] + 0.5 * innovation[1];
        _state[6] = 0.5 * _state[6] + 0.5 * innovation[2];

        // Reduce covariance
        for (int i = 0; i < 7; i++)
        {
            _covariance[i, i] *= (1 - alpha);
        }

        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        TimeSinceUpdate = 0;
        Hits++;

        if (State == TrackState.Tentative && Hits >= 3)
        {
            State = TrackState.Confirmed;
        }
    }

    public void MarkMissed()
    {
        if (State == TrackState.Tentative)
        {
            State = TrackState.Deleted;
        }
    }

    public BoundingBox<T> GetPredictedBox()
    {
        return StateToBox(_state[0], _state[1], _state[2], _state[3]);
    }

    public Track<T> ToTrack()
    {
        var box = GetPredictedBox();
        return new Track<T>(_trackId, box, _classId, _confidence)
        {
            VelocityX = _state[4],
            VelocityY = _state[5],
            TimeSinceUpdate = TimeSinceUpdate,
            Age = Age,
            Hits = Hits,
            State = State
        };
    }

    private (double cx, double cy, double s, double r) BoxToState(BoundingBox<T> box)
    {
        double x1 = _numOps.ToDouble(box.X1);
        double y1 = _numOps.ToDouble(box.Y1);
        double x2 = _numOps.ToDouble(box.X2);
        double y2 = _numOps.ToDouble(box.Y2);

        double w = x2 - x1;
        double h = y2 - y1;
        double cx = x1 + w / 2;
        double cy = y1 + h / 2;
        double s = w * h;
        double r = h > 0 ? w / h : 1;

        return (cx, cy, s, r);
    }

    private BoundingBox<T> StateToBox(double cx, double cy, double s, double r)
    {
        double w = Math.Sqrt(Math.Max(1, s * r));
        double h = r > 0 ? w / r : w;

        double x1 = cx - w / 2;
        double y1 = cy - h / 2;
        double x2 = cx + w / 2;
        double y2 = cy + h / 2;

        return new BoundingBox<T>(
            _numOps.FromDouble(x1), _numOps.FromDouble(y1),
            _numOps.FromDouble(x2), _numOps.FromDouble(y2),
            BoundingBoxFormat.XYXY);
    }
}
