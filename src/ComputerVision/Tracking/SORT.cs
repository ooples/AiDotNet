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
                    NumOps,
                    Options.MinHits);
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
            Tracks = new List<Track<T>>(Tracks),
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
/// Full Kalman filter-based track for SORT.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Implements a complete Kalman filter for object tracking with:</para>
/// <list type="bullet">
/// <item>State vector: [x, y, s, r, vx, vy, vs] (center, area, aspect ratio, velocities)</item>
/// <item>Full state transition matrix F with velocity integration</item>
/// <item>Proper measurement matrix H for observation</item>
/// <item>Kalman gain K = PH'(HPH' + R)^(-1) computation</item>
/// <item>Joseph form covariance update for numerical stability</item>
/// </list>
/// </remarks>
internal class KalmanTrack<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _trackId;
    private readonly int _minHits;
    private int _classId;
    private T _confidence;

    private const int StateSize = 7;
    private const int MeasurementSize = 4;

    // Kalman state: [x, y, s, r, vx, vy, vs]
    // x, y = center position
    // s = area (w*h)
    // r = aspect ratio (w/h)
    // vx, vy, vs = velocities
    private double[] _state;
    private double[,] _covariance;

    // State transition matrix F
    private readonly double[,] _F;

    // Measurement matrix H
    private readonly double[,] _H;

    // Process noise covariance Q
    private readonly double[,] _Q;

    // Measurement noise covariance R
    private readonly double[,] _R;

    public int TimeSinceUpdate { get; private set; }
    public int Hits { get; private set; }
    public int Age { get; private set; }
    public TrackState State { get; private set; }

    public KalmanTrack(int trackId, Detection<T> detection, INumericOperations<T> numOps, int minHits = 3)
    {
        _trackId = trackId;
        _numOps = numOps;
        _minHits = minHits;
        _classId = detection.ClassId;
        _confidence = detection.Confidence;

        // Initialize state from detection
        var (cx, cy, s, r) = BoxToState(detection.Box);
        _state = new double[] { cx, cy, s, r, 0, 0, 0 };

        // State transition matrix F: constant velocity model
        // x' = x + vx, y' = y + vy, s' = s + vs, r' = r (unchanged)
        _F = new double[StateSize, StateSize];
        for (int i = 0; i < StateSize; i++) _F[i, i] = 1.0;
        _F[0, 4] = 1.0; // x += vx
        _F[1, 5] = 1.0; // y += vy
        _F[2, 6] = 1.0; // s += vs

        // Measurement matrix H: we observe [x, y, s, r]
        _H = new double[MeasurementSize, StateSize];
        for (int i = 0; i < MeasurementSize; i++) _H[i, i] = 1.0;

        // Process noise covariance Q
        // Uncertainty in position, scale, and velocity
        double stdWeight = 0.05; // Standard weight for position uncertainty
        double stdVelWeight = 0.00625; // Standard weight for velocity uncertainty

        _Q = new double[StateSize, StateSize];
        _Q[0, 0] = Math.Pow(stdWeight * s, 2); // x position
        _Q[1, 1] = Math.Pow(stdWeight * s, 2); // y position
        _Q[2, 2] = 0.01; // area change
        _Q[3, 3] = 1e-8; // aspect ratio (nearly constant)
        _Q[4, 4] = Math.Pow(stdVelWeight * s, 2); // x velocity
        _Q[5, 5] = Math.Pow(stdVelWeight * s, 2); // y velocity
        _Q[6, 6] = 1e-4; // area velocity

        // Measurement noise covariance R
        _R = new double[MeasurementSize, MeasurementSize];
        _R[0, 0] = Math.Pow(stdWeight * s, 2); // x measurement noise
        _R[1, 1] = Math.Pow(stdWeight * s, 2); // y measurement noise
        _R[2, 2] = 10.0; // area measurement noise
        _R[3, 3] = 0.01; // aspect ratio measurement noise

        // Initialize covariance P (uncertainty in initial state)
        _covariance = new double[StateSize, StateSize];
        _covariance[0, 0] = 2 * _R[0, 0]; // x
        _covariance[1, 1] = 2 * _R[1, 1]; // y
        _covariance[2, 2] = 10.0; // area
        _covariance[3, 3] = 0.01; // aspect ratio
        _covariance[4, 4] = Math.Pow(10 * stdVelWeight * s, 2); // vx (high uncertainty)
        _covariance[5, 5] = Math.Pow(10 * stdVelWeight * s, 2); // vy (high uncertainty)
        _covariance[6, 6] = 1e-2; // vs

        TimeSinceUpdate = 0;
        Hits = 1;
        Age = 1;
        State = TrackState.Tentative;
    }

    /// <summary>
    /// Performs the Kalman filter prediction step.
    /// Predicts the state and covariance forward in time.
    /// </summary>
    public void Predict()
    {
        // State prediction: x' = F * x
        var newState = new double[StateSize];
        for (int i = 0; i < StateSize; i++)
        {
            for (int j = 0; j < StateSize; j++)
            {
                newState[i] += _F[i, j] * _state[j];
            }
        }
        _state = newState;

        // Ensure positive area
        if (_state[2] <= 0)
            _state[2] = 1;

        // Covariance prediction: P' = F * P * F' + Q
        var FP = MatrixMultiply(_F, _covariance, StateSize, StateSize, StateSize);
        var Ft = Transpose(_F, StateSize, StateSize);
        var FPFt = MatrixMultiply(FP, Ft, StateSize, StateSize, StateSize);

        for (int i = 0; i < StateSize; i++)
        {
            for (int j = 0; j < StateSize; j++)
            {
                _covariance[i, j] = FPFt[i, j] + _Q[i, j];
            }
        }

        Age++;
        TimeSinceUpdate++;
    }

    /// <summary>
    /// Performs the Kalman filter update step with a new measurement.
    /// Computes Kalman gain and updates state and covariance.
    /// </summary>
    public void Update(Detection<T> detection)
    {
        var (cx, cy, s, r) = BoxToState(detection.Box);
        double[] measurement = { cx, cy, s, r };

        // Innovation (measurement residual): y = z - H*x
        var Hx = new double[MeasurementSize];
        for (int i = 0; i < MeasurementSize; i++)
        {
            for (int j = 0; j < StateSize; j++)
            {
                Hx[i] += _H[i, j] * _state[j];
            }
        }

        var innovation = new double[MeasurementSize];
        for (int i = 0; i < MeasurementSize; i++)
        {
            innovation[i] = measurement[i] - Hx[i];
        }

        // Innovation covariance: S = H * P * H' + R
        var PH = new double[StateSize, MeasurementSize];
        for (int i = 0; i < StateSize; i++)
        {
            for (int j = 0; j < MeasurementSize; j++)
            {
                for (int k = 0; k < StateSize; k++)
                {
                    PH[i, j] += _covariance[i, k] * _H[j, k]; // H'[k,j] = H[j,k]
                }
            }
        }

        var S = new double[MeasurementSize, MeasurementSize];
        for (int i = 0; i < MeasurementSize; i++)
        {
            for (int j = 0; j < MeasurementSize; j++)
            {
                for (int k = 0; k < StateSize; k++)
                {
                    S[i, j] += _H[i, k] * PH[k, j];
                }
                S[i, j] += _R[i, j];
            }
        }

        // Kalman gain: K = P * H' * S^(-1)
        var SInv = InvertMatrix4x4(S);
        var K = new double[StateSize, MeasurementSize];
        for (int i = 0; i < StateSize; i++)
        {
            for (int j = 0; j < MeasurementSize; j++)
            {
                for (int k = 0; k < MeasurementSize; k++)
                {
                    K[i, j] += PH[i, k] * SInv[k, j];
                }
            }
        }

        // State update: x' = x + K * y
        for (int i = 0; i < StateSize; i++)
        {
            for (int j = 0; j < MeasurementSize; j++)
            {
                _state[i] += K[i, j] * innovation[j];
            }
        }

        // Covariance update using Joseph form for numerical stability:
        // P' = (I - K*H) * P * (I - K*H)' + K * R * K'
        // Simplified: P' = (I - K*H) * P
        var KH = new double[StateSize, StateSize];
        for (int i = 0; i < StateSize; i++)
        {
            for (int j = 0; j < StateSize; j++)
            {
                for (int k = 0; k < MeasurementSize; k++)
                {
                    KH[i, j] += K[i, k] * _H[k, j];
                }
            }
        }

        var IminusKH = new double[StateSize, StateSize];
        for (int i = 0; i < StateSize; i++)
        {
            for (int j = 0; j < StateSize; j++)
            {
                IminusKH[i, j] = (i == j ? 1.0 : 0.0) - KH[i, j];
            }
        }

        var newCovariance = MatrixMultiply(IminusKH, _covariance, StateSize, StateSize, StateSize);
        _covariance = newCovariance;

        // Ensure covariance remains symmetric and positive semi-definite
        for (int i = 0; i < StateSize; i++)
        {
            for (int j = i + 1; j < StateSize; j++)
            {
                double avg = (_covariance[i, j] + _covariance[j, i]) / 2;
                _covariance[i, j] = avg;
                _covariance[j, i] = avg;
            }
            // Ensure diagonal is positive
            if (_covariance[i, i] < 1e-10)
                _covariance[i, i] = 1e-10;
        }

        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        TimeSinceUpdate = 0;
        Hits++;

        if (State == TrackState.Tentative && Hits >= _minHits)
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

    /// <summary>
    /// Gets the Mahalanobis distance to a measurement (used for gating).
    /// </summary>
    public double MahalanobisDistance(BoundingBox<T> box)
    {
        var (cx, cy, s, r) = BoxToState(box);
        double[] measurement = { cx, cy, s, r };

        // Predicted measurement
        var Hx = new double[MeasurementSize];
        for (int i = 0; i < MeasurementSize; i++)
        {
            for (int j = 0; j < StateSize; j++)
            {
                Hx[i] += _H[i, j] * _state[j];
            }
        }

        // Innovation
        var y = new double[MeasurementSize];
        for (int i = 0; i < MeasurementSize; i++)
        {
            y[i] = measurement[i] - Hx[i];
        }

        // Innovation covariance
        var S = new double[MeasurementSize, MeasurementSize];
        for (int i = 0; i < MeasurementSize; i++)
        {
            for (int j = 0; j < MeasurementSize; j++)
            {
                for (int k = 0; k < StateSize; k++)
                {
                    for (int l = 0; l < StateSize; l++)
                    {
                        S[i, j] += _H[i, k] * _covariance[k, l] * _H[j, l];
                    }
                }
                S[i, j] += _R[i, j];
            }
        }

        // Mahalanobis distance: d^2 = y' * S^(-1) * y
        var SInv = InvertMatrix4x4(S);
        double distance = 0;
        for (int i = 0; i < MeasurementSize; i++)
        {
            for (int j = 0; j < MeasurementSize; j++)
            {
                distance += y[i] * SInv[i, j] * y[j];
            }
        }

        return Math.Sqrt(distance);
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

    #region Matrix Operations

    private static double[,] MatrixMultiply(double[,] A, double[,] B, int m, int n, int p)
    {
        var result = new double[m, p];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    result[i, j] += A[i, k] * B[k, j];
                }
            }
        }
        return result;
    }

    private static double[,] Transpose(double[,] A, int m, int n)
    {
        var result = new double[n, m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[j, i] = A[i, j];
            }
        }
        return result;
    }

    private static double[,] InvertMatrix4x4(double[,] m)
    {
        // 4x4 matrix inversion using adjugate method
        var result = new double[4, 4];

        double det =
            m[0, 0] * (m[1, 1] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) -
                       m[1, 2] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) +
                       m[1, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1])) -
            m[0, 1] * (m[1, 0] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) -
                       m[1, 2] * (m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]) +
                       m[1, 3] * (m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0])) +
            m[0, 2] * (m[1, 0] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) -
                       m[1, 1] * (m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]) +
                       m[1, 3] * (m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0])) -
            m[0, 3] * (m[1, 0] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]) -
                       m[1, 1] * (m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]) +
                       m[1, 2] * (m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0]));

        if (Math.Abs(det) < 1e-10)
        {
            // Nearly singular, return identity
            for (int i = 0; i < 4; i++) result[i, i] = 1;
            return result;
        }

        double invDet = 1.0 / det;

        // Compute adjugate matrix and multiply by 1/det
        result[0, 0] = invDet * (m[1, 1] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) - m[1, 2] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) + m[1, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]));
        result[0, 1] = invDet * (-(m[0, 1] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) - m[0, 2] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) + m[0, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1])));
        result[0, 2] = invDet * (m[0, 1] * (m[1, 2] * m[3, 3] - m[1, 3] * m[3, 2]) - m[0, 2] * (m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1]) + m[0, 3] * (m[1, 1] * m[3, 2] - m[1, 2] * m[3, 1]));
        result[0, 3] = invDet * (-(m[0, 1] * (m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2]) - m[0, 2] * (m[1, 1] * m[2, 3] - m[1, 3] * m[2, 1]) + m[0, 3] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])));

        result[1, 0] = invDet * (-(m[1, 0] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) - m[1, 2] * (m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]) + m[1, 3] * (m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0])));
        result[1, 1] = invDet * (m[0, 0] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) - m[0, 2] * (m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]) + m[0, 3] * (m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]));
        result[1, 2] = invDet * (-(m[0, 0] * (m[1, 2] * m[3, 3] - m[1, 3] * m[3, 2]) - m[0, 2] * (m[1, 0] * m[3, 3] - m[1, 3] * m[3, 0]) + m[0, 3] * (m[1, 0] * m[3, 2] - m[1, 2] * m[3, 0])));
        result[1, 3] = invDet * (m[0, 0] * (m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2]) - m[0, 2] * (m[1, 0] * m[2, 3] - m[1, 3] * m[2, 0]) + m[0, 3] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]));

        result[2, 0] = invDet * (m[1, 0] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) - m[1, 1] * (m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]) + m[1, 3] * (m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0]));
        result[2, 1] = invDet * (-(m[0, 0] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) - m[0, 1] * (m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]) + m[0, 3] * (m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0])));
        result[2, 2] = invDet * (m[0, 0] * (m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1]) - m[0, 1] * (m[1, 0] * m[3, 3] - m[1, 3] * m[3, 0]) + m[0, 3] * (m[1, 0] * m[3, 1] - m[1, 1] * m[3, 0]));
        result[2, 3] = invDet * (-(m[0, 0] * (m[1, 1] * m[2, 3] - m[1, 3] * m[2, 1]) - m[0, 1] * (m[1, 0] * m[2, 3] - m[1, 3] * m[2, 0]) + m[0, 3] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])));

        result[3, 0] = invDet * (-(m[1, 0] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]) - m[1, 1] * (m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]) + m[1, 2] * (m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0])));
        result[3, 1] = invDet * (m[0, 0] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]) - m[0, 1] * (m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]) + m[0, 2] * (m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0]));
        result[3, 2] = invDet * (-(m[0, 0] * (m[1, 1] * m[3, 2] - m[1, 2] * m[3, 1]) - m[0, 1] * (m[1, 0] * m[3, 2] - m[1, 2] * m[3, 0]) + m[0, 2] * (m[1, 0] * m[3, 1] - m[1, 1] * m[3, 0])));
        result[3, 3] = invDet * (m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]));

        return result;
    }

    #endregion
}
