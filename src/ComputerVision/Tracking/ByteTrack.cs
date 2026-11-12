using AiDotNet.Attributes;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Enums;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Tracking;

/// <summary>
/// ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> ByteTrack improves upon SORT by using almost all detection
/// boxes (including low-confidence ones) for tracking. It uses a two-stage association:
/// first matching high-confidence detections, then low-confidence ones.</para>
///
/// <para>Key features:
/// - Associates all detection boxes (high and low confidence)
/// - Two-stage matching strategy
/// - Recovers occluded objects using low-score detections
/// - State-of-the-art performance on MOT benchmarks
/// </para>
///
/// <para>This follows the official BYTETracker: an xyah Kalman filter per track, IoU distance with
/// detection scores fused in, <c>lapjv</c>-style assignment that accepts a pair only when its cost is at
/// most the threshold (0.8, 0.5 and 0.7 for the high-score, low-score and unconfirmed rounds), and new
/// tracks only from detections scoring at least <c>track_thresh + 0.1</c>.</para>
///
/// <para>Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
/// Every Detection Box", ECCV 2022</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
    "https://arxiv.org/abs/2110.06864",
    Year = 2022,
    Authors = "Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang")]
public class ByteTrack<T> : ObjectTrackerBase<T>
{
    private readonly XyahKalmanFilter<T> _kalman;
    private readonly HashSet<int> _removedTrackIds;
    private readonly T _trackThresh;
    private readonly T _lowThresh;
    private readonly T _detThresh;
    private readonly T _matchThresh;
    private readonly T _secondMatchThresh;
    private readonly T _unconfirmedMatchThresh;
    private readonly T _duplicateThresh;
    private readonly bool _fuseScore;
    private readonly int _maxTimeLost;
    private List<STrack<T>> _trackedTracks;
    private List<STrack<T>> _lostTracks;

    /// <inheritdoc/>
    public override string Name => "ByteTrack";

    /// <summary>
    /// Creates a new ByteTrack tracker.
    /// </summary>
    /// <param name="options">
    /// Tracking options. <see cref="TrackingOptions{T}.ConfidenceThreshold"/> is <c>track_thresh</c> and
    /// <see cref="TrackingOptions{T}.MaxAge"/> is the lost-track buffer. Pass a
    /// <see cref="ByteTrackOptions{T}"/> to change the association thresholds; otherwise the official
    /// defaults are used.
    /// </param>
    public ByteTrack(TrackingOptions<T> options) : base(options)
    {
        var byteOptions = options as ByteTrackOptions<T> ?? new ByteTrackOptions<T>();
        _kalman = new XyahKalmanFilter<T>();
        _removedTrackIds = new HashSet<int>();
        _trackedTracks = new List<STrack<T>>();
        _lostTracks = new List<STrack<T>>();

        _trackThresh = options.ConfidenceThreshold;
        _lowThresh = byteOptions.LowScoreThreshold;
        _detThresh = NumOps.Add(_trackThresh, byteOptions.NewTrackScoreMargin);
        _matchThresh = byteOptions.MatchThreshold;
        _secondMatchThresh = byteOptions.SecondMatchThreshold;
        _unconfirmedMatchThresh = byteOptions.UnconfirmedMatchThreshold;
        _duplicateThresh = byteOptions.DuplicateIouDistance;
        _fuseScore = byteOptions.FuseScore;
        _maxTimeLost = options.MaxAge;
    }

    /// <inheritdoc/>
    public override TrackingResult<T> Update(List<Detection<T>> detections)
    {
        var startTime = DateTime.UtcNow;
        FrameCount++;
        var engine = AiDotNetEngine.Current;

        var activated = new List<STrack<T>>();
        var refind = new List<STrack<T>>();
        var lost = new List<STrack<T>>();
        var removed = new List<STrack<T>>();

        // High: score > track_thresh. Second: low_thresh < score < track_thresh (both strict, as in
        // the official BYTETracker; a score equal to track_thresh joins neither set).
        var scores = TrackingAssociation<T>.Scores(detections);
        int[] highIdx = TrackingAssociation<T>.Where(engine.TensorGreaterThan(scores, _trackThresh));
        int[] secondIdx = TrackingAssociation<T>.Where(engine.TensorMultiply(
            engine.TensorGreaterThan(scores, _lowThresh), engine.TensorLessThan(scores, _trackThresh)));
        var highDets = highIdx.Select(i => new STrack<T>(detections[i], _kalman)).ToList();
        var secondDets = secondIdx.Select(i => new STrack<T>(detections[i], _kalman)).ToList();

        var unconfirmed = _trackedTracks.Where(t => !t.IsActivated).ToList();
        var tracked = _trackedTracks.Where(t => t.IsActivated).ToList();

        // Step 2: first association, tracked + lost tracks against the high-score detections.
        var pool = JoinTracks(tracked, _lostTracks);
        foreach (var track in pool)
        {
            track.Predict();
        }

        var dists = IouDistance(pool, highDets);
        if (_fuseScore)
        {
            dists = FuseScore(dists, highDets);
        }

        var (matches, uTrack, uDetection) = TrackingAssociation<T>.LinearAssignment(dists, _matchThresh);
        ApplyMatches(matches, pool, highDets, activated, refind);

        // Step 3: second association, remaining tracked (not lost) tracks against the low-score detections.
        var remainingTracked = uTrack.Select(i => pool[i]).Where(t => t.State == TrackState.Confirmed).ToList();
        dists = IouDistance(remainingTracked, secondDets);
        var (matches2, uTrack2, _) = TrackingAssociation<T>.LinearAssignment(dists, _secondMatchThresh);
        ApplyMatches(matches2, remainingTracked, secondDets, activated, refind);

        foreach (int it in uTrack2)
        {
            var track = remainingTracked[it];
            if (track.State != TrackState.Lost)
            {
                track.MarkLost();
                lost.Add(track);
            }
        }

        // Unconfirmed tracks (one frame old) against the high-score detections left over.
        var leftoverDets = uDetection.Select(i => highDets[i]).ToList();
        dists = IouDistance(unconfirmed, leftoverDets);
        if (_fuseScore)
        {
            dists = FuseScore(dists, leftoverDets);
        }

        var (matches3, uUnconfirmed, uDetection3) = TrackingAssociation<T>.LinearAssignment(dists, _unconfirmedMatchThresh);
        foreach (var (row, col) in matches3)
        {
            unconfirmed[row].Update(leftoverDets[col], FrameCount);
            activated.Add(unconfirmed[row]);
        }

        foreach (int it in uUnconfirmed)
        {
            unconfirmed[it].MarkRemoved();
            removed.Add(unconfirmed[it]);
        }

        // Step 4: start new tracks from unmatched high-score detections scoring at least det_thresh.
        var newCandidates = uDetection3.Select(i => leftoverDets[i]).ToList();
        int[] belowDetThresh = newCandidates.Count == 0
            ? Array.Empty<int>()
            : engine.TensorLessThan(
                TrackingAssociation<T>.Scores(newCandidates.Select(t => t.Detection).ToList()), _detThresh)
                .Cast<int>().ToArray();
        for (int k = 0; k < newCandidates.Count; k++)
        {
            if (belowDetThresh[k] != 0)
            {
                continue;
            }

            newCandidates[k].Activate(NextTrackId++, FrameCount);
            activated.Add(newCandidates[k]);
        }

        // Step 5: lost tracks past the buffer are removed.
        foreach (var track in _lostTracks)
        {
            if (FrameCount - track.EndFrame > _maxTimeLost)
            {
                track.MarkRemoved();
                removed.Add(track);
            }
        }

        _trackedTracks = _trackedTracks.Where(t => t.State == TrackState.Confirmed).ToList();
        _trackedTracks = JoinTracks(_trackedTracks, activated);
        _trackedTracks = JoinTracks(_trackedTracks, refind);
        _lostTracks = SubTracks(_lostTracks, _trackedTracks.Select(t => t.TrackId));
        _lostTracks.AddRange(lost);
        _lostTracks = SubTracks(_lostTracks, _removedTrackIds);
        foreach (var track in removed)
        {
            _removedTrackIds.Add(track.TrackId);
        }

        RemoveDuplicateTracks();

        Tracks.Clear();
        foreach (var track in _trackedTracks)
        {
            if (track.IsActivated)
            {
                Tracks.Add(track.ToTrack());
            }
        }

        return new TrackingResult<T>
        {
            Tracks = Tracks.ToList(),
            FrameNumber = FrameCount,
            TrackingTime = DateTime.UtcNow - startTime
        };
    }

    private void ApplyMatches(List<(int Row, int Col)> matches, List<STrack<T>> tracks, List<STrack<T>> dets,
        List<STrack<T>> activated, List<STrack<T>> refind)
    {
        foreach (var (row, col) in matches)
        {
            var track = tracks[row];
            if (track.State == TrackState.Confirmed)
            {
                track.Update(dets[col], FrameCount);
                activated.Add(track);
            }
            else
            {
                track.ReActivate(dets[col], FrameCount);
                refind.Add(track);
            }
        }
    }

    private static Tensor<T> IouDistance(List<STrack<T>> tracks, List<STrack<T>> dets)
    {
        return TrackingAssociation<T>.IouDistance(
            TrackingAssociation<T>.BoxesToTensor(tracks.Select(t => t.Tlbr).ToList()),
            TrackingAssociation<T>.BoxesToTensor(dets.Select(t => t.Tlbr).ToList()));
    }

    /// <summary>
    /// <c>fuse_score</c>: the IoU similarity is multiplied by each detection's score, so
    /// cost = 1 - IoU * score.
    /// </summary>
    private Tensor<T> FuseScore(Tensor<T> cost, List<STrack<T>> dets)
    {
        if (cost.Length == 0)
        {
            return cost;
        }

        var engine = AiDotNetEngine.Current;
        var scores = engine.Reshape(TrackingAssociation<T>.Scores(dets.Select(d => d.Detection).ToList()), new[] { 1, dets.Count });
        var similarity = engine.TensorMultiply(
            engine.ScalarMinusTensor(NumOps.One, cost), engine.TensorBroadcastTo(scores, cost.Shape.ToArray()));
        return engine.ScalarMinusTensor(NumOps.One, similarity);
    }

    private static List<STrack<T>> JoinTracks(List<STrack<T>> tracksA, List<STrack<T>> tracksB)
    {
        var result = new List<STrack<T>>(tracksA);
        var existingIds = new HashSet<int>(tracksA.Select(t => t.TrackId));
        foreach (var track in tracksB)
        {
            if (existingIds.Add(track.TrackId))
            {
                result.Add(track);
            }
        }

        return result;
    }

    private static List<STrack<T>> SubTracks(List<STrack<T>> tracksA, IEnumerable<int> idsToRemove)
    {
        var ids = new HashSet<int>(idsToRemove);
        return tracksA.Where(t => !ids.Contains(t.TrackId)).ToList();
    }

    /// <summary>
    /// <c>remove_duplicate_stracks</c>: a tracked/lost pair with IoU distance below 0.15 keeps only the
    /// track that has been followed longer (ties drop the tracked one).
    /// </summary>
    private void RemoveDuplicateTracks()
    {
        if (_trackedTracks.Count == 0 || _lostTracks.Count == 0)
        {
            return;
        }

        var close = AiDotNetEngine.Current.TensorLessThan(IouDistance(_trackedTracks, _lostTracks), _duplicateThresh);
        int[] pairs = TrackingAssociation<T>.Where(close);
        var dupTracked = new HashSet<int>();
        var dupLost = new HashSet<int>();
        for (int k = 0; k + 1 < pairs.Length; k += 2)
        {
            int p = pairs[k];
            int q = pairs[k + 1];
            int timeP = _trackedTracks[p].FrameId - _trackedTracks[p].StartFrame;
            int timeQ = _lostTracks[q].FrameId - _lostTracks[q].StartFrame;
            if (timeP > timeQ)
            {
                dupLost.Add(q);
            }
            else
            {
                dupTracked.Add(p);
            }
        }

        _trackedTracks = _trackedTracks.Where((_, i) => !dupTracked.Contains(i)).ToList();
        _lostTracks = _lostTracks.Where((_, i) => !dupLost.Contains(i)).ToList();
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _trackedTracks.Clear();
        _lostTracks.Clear();
        _removedTrackIds.Clear();
    }
}

/// <summary>
/// Association thresholds of <see cref="ByteTrack{T}"/>; the defaults are those of the official
/// BYTETracker (Zhang et al. 2022).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ByteTrackOptions<T> : TrackingOptions<T>
{
    private static readonly INumericOperations<T> Ops = Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Largest IoU distance (1 - IoU, after score fusion) accepted in the first association
    /// (<c>match_thresh</c>, default 0.8, i.e. IoU of at least 0.2).
    /// </summary>
    public T MatchThreshold { get; set; } = Ops.FromDouble(0.8);

    /// <summary>Largest IoU distance accepted when matching low-score detections (default 0.5).</summary>
    public T SecondMatchThreshold { get; set; } = Ops.FromDouble(0.5);

    /// <summary>Largest IoU distance accepted when matching unconfirmed tracks (default 0.7).</summary>
    public T UnconfirmedMatchThreshold { get; set; } = Ops.FromDouble(0.7);

    /// <summary>Detections scoring at or below this are discarded (default 0.1).</summary>
    public T LowScoreThreshold { get; set; } = Ops.FromDouble(0.1);

    /// <summary>
    /// A new track needs a score of at least <c>ConfidenceThreshold + NewTrackScoreMargin</c>
    /// (<c>det_thresh = track_thresh + 0.1</c>).
    /// </summary>
    public T NewTrackScoreMargin { get; set; } = Ops.FromDouble(0.1);

    /// <summary>
    /// Whether detection scores are fused into the IoU cost of the high-score and unconfirmed
    /// associations (<c>fuse_score</c>; the official code skips it only for MOT20). Default true.
    /// </summary>
    public bool FuseScore { get; set; } = true;

    /// <summary>
    /// IoU distance below which a tracked and a lost track are considered duplicates (default 0.15).
    /// </summary>
    public T DuplicateIouDistance { get; set; } = Ops.FromDouble(0.15);
}

/// <summary>
/// Single track for ByteTrack (STrack), filtered with the xyah Kalman filter.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class STrack<T>
{
    private static readonly INumericOperations<T> NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
    private readonly XyahKalmanFilter<T> _kalman;
    private Tensor<T>? _mean;
    private Tensor<T>? _covariance;

    public STrack(Detection<T> detection, XyahKalmanFilter<T> kalman)
    {
        _kalman = kalman;
        Detection = detection;
        ClassId = detection.ClassId;
        Score = detection.Confidence;
        State = TrackState.Tentative;
    }

    /// <summary>The detection this track was created from (or last updated with).</summary>
    public Detection<T> Detection { get; private set; }

    public int TrackId { get; private set; }
    public int ClassId { get; private set; }
    public T Score { get; private set; }
    public int FrameId { get; private set; }
    public int StartFrame { get; private set; }
    public int TrackletLen { get; private set; }
    public bool IsActivated { get; private set; }
    public TrackState State { get; private set; }
    public int EndFrame => FrameId;

    /// <summary>Starts a new tracklet; only tracks born in the first frame start activated.</summary>
    public void Activate(int trackId, int frameId)
    {
        TrackId = trackId;
        (_mean, _covariance) = _kalman.Initiate(ToXyah(Detection.Box));
        TrackletLen = 0;
        State = TrackState.Confirmed;
        IsActivated = frameId == 1;
        FrameId = frameId;
        StartFrame = frameId;
    }

    /// <summary>Kalman prediction; a track that is not being tracked has its height velocity zeroed first.</summary>
    public void Predict()
    {
        if (_mean is null || _covariance is null)
        {
            return;
        }

        var mean = _mean;
        if (State != TrackState.Confirmed)
        {
            mean = _mean.Clone();
            mean[7, 0] = NumOps.Zero;
        }

        (_mean, _covariance) = _kalman.Predict(mean, _covariance);
    }

    public void Update(STrack<T> newTrack, int frameId)
    {
        FrameId = frameId;
        TrackletLen++;
        Correct(newTrack);
    }

    public void ReActivate(STrack<T> newTrack, int frameId)
    {
        TrackletLen = 0;
        FrameId = frameId;
        Correct(newTrack);
    }

    public void MarkLost()
    {
        State = TrackState.Lost;
    }

    public void MarkRemoved()
    {
        State = TrackState.Deleted;
    }

    /// <summary>The box as (x1, y1, x2, y2): from the filter state when there is one, else the detection.</summary>
    public BoundingBox<T> Tlbr
    {
        get
        {
            if (_mean is null)
            {
                return Detection.Box;
            }

            T half = NumOps.FromDouble(0.5);
            T h = _mean[3, 0];
            T w = NumOps.Multiply(_mean[2, 0], h);
            T x1 = NumOps.Subtract(_mean[0, 0], NumOps.Multiply(w, half));
            T y1 = NumOps.Subtract(_mean[1, 0], NumOps.Multiply(h, half));
            return new BoundingBox<T>(x1, y1, NumOps.Add(x1, w), NumOps.Add(y1, h), BoundingBoxFormat.XYXY);
        }
    }

    public Track<T> ToTrack()
    {
        var velocityX = _mean is null ? NumOps.Zero : _mean[4, 0];
        var velocityY = _mean is null ? NumOps.Zero : _mean[5, 0];
        return new Track<T>(TrackId, Tlbr, ClassId, Score)
        {
            VelocityX = velocityX,
            VelocityY = velocityY,
            TimeSinceUpdate = 0,
            Age = FrameId - StartFrame + 1,
            Hits = TrackletLen,
            State = State
        };
    }

    private void Correct(STrack<T> newTrack)
    {
        if (_mean is not null && _covariance is not null)
        {
            (_mean, _covariance) = _kalman.Update(_mean, _covariance, ToXyah(newTrack.Detection.Box));
        }

        Detection = newTrack.Detection;
        ClassId = newTrack.ClassId;
        Score = newTrack.Score;
        State = TrackState.Confirmed;
        IsActivated = true;
    }

    /// <summary>(x1, y1, x2, y2) to (centre x, centre y, width / height, height).</summary>
    private static Tensor<T> ToXyah(BoundingBox<T> box)
    {
        T half = NumOps.FromDouble(0.5);
        T w = NumOps.Subtract(box.X2, box.X1);
        T h = NumOps.Subtract(box.Y2, box.Y1);
        return new Tensor<T>(new[]
        {
            NumOps.Add(box.X1, NumOps.Multiply(w, half)),
            NumOps.Add(box.Y1, NumOps.Multiply(h, half)),
            NumOps.Divide(w, h),
            h
        }, new[] { 4 });
    }
}
