using AiDotNet.Attributes;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

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
/// <para>This follows the reference implementation (abewley/sort): a 7-state constant-velocity
/// Kalman filter over (u, v, s, r) with filterpy's noise settings, an optimal assignment on the IoU
/// matrix, matches below <see cref="TrackingOptions{T}.IouThreshold"/> rejected, and a track reported
/// only in frames where it was matched and its hit streak has reached
/// <see cref="TrackingOptions{T}.MinHits"/> (or during the first MinHits frames).</para>
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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Simple Online and Realtime Tracking",
    "https://arxiv.org/abs/1602.00763",
    Year = 2016,
    Authors = "Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft")]
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
        var engine = AiDotNetEngine.Current;

        // Filter detections by confidence
        var scores = TrackingAssociation<T>.Scores(detections);
        var keep = TrackingAssociation<T>.Where(
            engine.TensorGreaterEqual(scores, engine.TensorFullLike(scores, Options.ConfidenceThreshold)));
        var filteredDetections = keep.Select(i => detections[i]).ToList();

        // Predict new locations of existing tracks; drop trackers whose prediction is not a number.
        foreach (var track in _kalmanTracks)
        {
            track.Predict();
        }

        _kalmanTracks.RemoveAll(t => t.HasInvalidPrediction());

        var (matches, unmatchedDetections) = Associate(filteredDetections);

        foreach (var (detIdx, trackIdx) in matches)
        {
            _kalmanTracks[trackIdx].Update(filteredDetections[detIdx]);
        }

        foreach (int detIdx in unmatchedDetections)
        {
            _kalmanTracks.Add(new KalmanTrack<T>(NextTrackId++, filteredDetections[detIdx]));
        }

        // Report tracks matched this frame whose hit streak reached MinHits (or every matched or new
        // track during the first MinHits frames), then drop tracks unmatched for more than MaxAge frames.
        Tracks.Clear();
        foreach (var kt in _kalmanTracks)
        {
            if (kt.TimeSinceUpdate < 1 && (kt.HitStreak >= Options.MinHits || FrameCount <= Options.MinHits))
            {
                Tracks.Add(kt.ToTrack(Options.MinHits));
            }
        }

        _kalmanTracks.RemoveAll(t => t.TimeSinceUpdate > Options.MaxAge);

        return new TrackingResult<T>
        {
            Tracks = new List<Track<T>>(Tracks),
            FrameNumber = FrameCount,
            TrackingTime = DateTime.UtcNow - startTime
        };
    }

    /// <summary>
    /// <c>associate_detections_to_trackers</c>: IoU matrix (detections x trackers); when every row and
    /// column has at most one IoU above the threshold those pairs are taken directly, otherwise the
    /// assignment that maximises total IoU is solved; pairs below the threshold are rejected.
    /// </summary>
    private (List<(int Det, int Track)> Matches, List<int> UnmatchedDetections) Associate(List<Detection<T>> detections)
    {
        var matches = new List<(int Det, int Track)>();
        if (_kalmanTracks.Count == 0 || detections.Count == 0)
        {
            return (matches, Enumerable.Range(0, detections.Count).ToList());
        }

        var engine = AiDotNetEngine.Current;
        var detBoxes = TrackingAssociation<T>.BoxesToTensor(detections.Select(d => d.Box).ToList());
        var trackBoxes = TrackingAssociation<T>.BoxesToTensor(_kalmanTracks.Select(t => t.GetPredictedBox()).ToList());
        var iou = TrackingAssociation<T>.IouMatrix(detBoxes, trackBoxes);
        int numTracks = _kalmanTracks.Count;

        var above = engine.TensorGreaterThan(iou, Options.IouThreshold);
        T maxPerRow = engine.TensorMaxValue(engine.ReduceSum(above, new[] { 1 }, keepDims: false));
        T maxPerCol = engine.TensorMaxValue(engine.ReduceSum(above, new[] { 0 }, keepDims: false));

        var candidates = new List<(int Det, int Track)>();
        if (NumOps.Equals(maxPerRow, NumOps.One) && NumOps.Equals(maxPerCol, NumOps.One))
        {
            int[] pairs = TrackingAssociation<T>.Where(above);
            for (int k = 0; k + 1 < pairs.Length; k += 2)
            {
                candidates.Add((pairs[k], pairs[k + 1]));
            }
        }
        else
        {
            int[] assignment = TrackingAssociation<T>.SolveAssignment(engine.TensorNegate(iou));
            for (int d = 0; d < assignment.Length; d++)
            {
                if (assignment[d] >= 0)
                {
                    candidates.Add((d, assignment[d]));
                }
            }
        }

        int[] rejected = candidates.Count == 0
            ? Array.Empty<int>()
            : engine.TensorLessThan(
                engine.TensorTake(iou, new Tensor<int>(
                    candidates.Select(c => c.Det * numTracks + c.Track).ToArray(), new[] { candidates.Count })),
                Options.IouThreshold).Cast<int>().ToArray();

        var detMatched = new bool[detections.Count];
        for (int k = 0; k < candidates.Count; k++)
        {
            if (rejected[k] == 0)
            {
                matches.Add(candidates[k]);
                detMatched[candidates[k].Det] = true;
            }
        }

        return (matches, Enumerable.Range(0, detections.Count).Where(d => !detMatched[d]).ToList());
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _kalmanTracks.Clear();
    }
}

/// <summary>
/// SORT's <c>KalmanBoxTracker</c>: a constant-velocity Kalman filter over the state
/// <c>[u, v, s, r, u', v', s']</c> (box centre, area, aspect ratio and the first three velocities).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Noise follows the reference implementation on top of filterpy's identity defaults:
/// R = diag(1, 1, 10, 10), P0 = diag(10, 10, 10, 10, 1e4, 1e4, 1e4) and
/// Q = diag(1, 1, 1, 1, 0.01, 0.01, 1e-4). The update is filterpy's, with the Joseph-form covariance
/// <c>(I - KH) P (I - KH)^T + K R K^T</c>.</para>
/// <para>The filter state and covariance are held in double whatever <typeparamref name="T"/> is; boxes are
/// converted on the way in and rounded to <typeparamref name="T"/> once on the way out. The state carries the
/// box AREA s = w * h, which passes Half's 65504 for any box larger than about 256 x 256 (and the covariance
/// starts at 1e4 and grows with s), so a Half state turned an ordinary 300 x 300 box into an infinite area and
/// a NaN box. filterpy's reference filter runs in float64.</para>
/// </remarks>
internal sealed class KalmanTrack<T>
{
    private static readonly INumericOperations<T> NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly int _trackId;
    private readonly Tensor<double> _f;
    private readonly Tensor<double> _fT;
    private readonly Tensor<double> _h;
    private readonly Tensor<double> _hT;
    private readonly Tensor<double> _q;
    private readonly Tensor<double> _r;
    private readonly Tensor<double> _identity;
    private int _classId;
    private T _confidence;
    private Tensor<double> _state;
    private Tensor<double> _covariance;

    public int TimeSinceUpdate { get; private set; }
    public int Hits { get; private set; }
    public int HitStreak { get; private set; }
    public int Age { get; private set; }

    public KalmanTrack(int trackId, Detection<T> detection)
    {
        _trackId = trackId;
        _classId = detection.ClassId;
        _confidence = detection.Confidence;

        _identity = _engine.TensorEye<double>(7);
        var shift = _engine.TensorSetSlice(new Tensor<double>(new[] { 7, 7 }), _engine.TensorEye<double>(3), new[] { 0, 4 });
        _f = _engine.TensorAdd(_identity, shift);
        _fT = _engine.TensorTranspose(_f);
        _h = _engine.TensorSlice(_identity, new[] { 0, 0 }, new[] { 4, 7 });
        _hT = _engine.TensorTranspose(_h);
        _r = _engine.TensorDiag(new Tensor<double>(new[] { 1.0, 1.0, 10.0, 10.0 }, new[] { 4 }));
        _q = _engine.TensorDiag(new Tensor<double>(new[] { 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4 }, new[] { 7 }));
        _covariance = _engine.TensorDiag(new Tensor<double>(new[] { 10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4 }, new[] { 7 }));
        _state = _engine.TensorConcatenate(new[] { BoxToMeasurement(detection.Box), new Tensor<double>(new[] { 3, 1 }) }, 0);
    }

    /// <summary>Advances the state one frame (<c>KalmanBoxTracker.predict</c>).</summary>
    public void Predict()
    {
        // Keep the predicted area non-negative: drop the area velocity when it would make s <= 0.
        if (_state[6, 0] + _state[2, 0] <= 0.0)
        {
            _state[6, 0] = 0.0;
        }

        _state = _engine.TensorMatMul(_f, _state);
        _covariance = _engine.TensorAdd(_engine.TensorMatMul(_engine.TensorMatMul(_f, _covariance), _fT), _q);

        Age++;
        if (TimeSinceUpdate > 0)
        {
            HitStreak = 0;
        }

        TimeSinceUpdate++;
    }

    /// <summary>True when the predicted box contains NaN (the reference tracker then drops the track).</summary>
    public bool HasInvalidPrediction()
    {
        var box = GetPredictedBox();
        return NumOps.IsNaN(NumOps.Add(NumOps.Add(box.X1, box.Y1), NumOps.Add(box.X2, box.Y2)));
    }

    /// <summary>Corrects the state with a matched detection (<c>KalmanBoxTracker.update</c>).</summary>
    public void Update(Detection<T> detection)
    {
        TimeSinceUpdate = 0;
        Hits++;
        HitStreak++;
        _classId = detection.ClassId;
        _confidence = detection.Confidence;

        var measurement = BoxToMeasurement(detection.Box);
        var innovation = _engine.TensorSubtract(measurement, _engine.TensorMatMul(_h, _state));
        var pht = _engine.TensorMatMul(_covariance, _hT);
        var s = _engine.TensorAdd(_engine.TensorMatMul(_h, pht), _r);
        var gain = _engine.TensorMatMul(pht, LinalgDispatch<double>.Inv(s));

        _state = _engine.TensorAdd(_state, _engine.TensorMatMul(gain, innovation));
        var iKh = _engine.TensorSubtract(_identity, _engine.TensorMatMul(gain, _h));
        _covariance = _engine.TensorAdd(
            _engine.TensorMatMul(_engine.TensorMatMul(iKh, _covariance), _engine.TensorTranspose(iKh)),
            _engine.TensorMatMul(_engine.TensorMatMul(gain, _r), _engine.TensorTranspose(gain)));
    }

    /// <summary>The current state as an (x1, y1, x2, y2) box (<c>convert_x_to_bbox</c>), rounded to T once.</summary>
    public BoundingBox<T> GetPredictedBox()
    {
        double w = Math.Sqrt(_state[2, 0] * _state[3, 0]);
        double h = _state[2, 0] / w;
        double cx = _state[0, 0];
        double cy = _state[1, 0];
        return new BoundingBox<T>(
            NumOps.FromDouble(cx - (w * 0.5)), NumOps.FromDouble(cy - (h * 0.5)),
            NumOps.FromDouble(cx + (w * 0.5)), NumOps.FromDouble(cy + (h * 0.5)),
            BoundingBoxFormat.XYXY);
    }

    public Track<T> ToTrack(int minHits)
    {
        return new Track<T>(_trackId, GetPredictedBox(), _classId, _confidence)
        {
            VelocityX = NumOps.FromDouble(_state[4, 0]),
            VelocityY = NumOps.FromDouble(_state[5, 0]),
            TimeSinceUpdate = TimeSinceUpdate,
            Age = Age,
            Hits = Hits,
            State = HitStreak >= minHits ? TrackState.Confirmed : TrackState.Tentative
        };
    }

    /// <summary>(x1, y1, x2, y2) to the <c>[4, 1]</c> measurement (u, v, s, r) in double
    /// (<c>convert_bbox_to_z</c>); a zero-height box gets aspect ratio 1 instead of a division by zero.</summary>
    private static Tensor<double> BoxToMeasurement(BoundingBox<T> box)
    {
        double x1 = NumOps.ToDouble(box.X1);
        double y1 = NumOps.ToDouble(box.Y1);
        double w = NumOps.ToDouble(box.X2) - x1;
        double h = NumOps.ToDouble(box.Y2) - y1;
        double r = h > 0.0 ? w / h : 1.0;
        return new Tensor<double>(new[] { x1 + (w * 0.5), y1 + (h * 0.5), w * h, r }, new[] { 4, 1 });
    }
}
