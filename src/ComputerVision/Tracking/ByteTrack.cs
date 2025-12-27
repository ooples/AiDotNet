using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Tensors;

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
/// <para>Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
/// Every Detection Box", ECCV 2022</para>
/// </remarks>
public class ByteTrack<T> : ObjectTrackerBase<T>
{
    private readonly List<STrack<T>> _trackedTracks;
    private readonly List<STrack<T>> _lostTracks;
    private readonly List<STrack<T>> _removedTracks;
    private readonly double _trackHighThresh;
    private readonly double _trackLowThresh;
    private readonly double _newTrackThresh;
    private readonly int _trackBuffer;
    private readonly double _matchThresh;

    /// <inheritdoc/>
    public override string Name => "ByteTrack";

    /// <summary>
    /// Creates a new ByteTrack tracker.
    /// </summary>
    public ByteTrack(TrackingOptions<T> options) : base(options)
    {
        _trackedTracks = new List<STrack<T>>();
        _lostTracks = new List<STrack<T>>();
        _removedTracks = new List<STrack<T>>();

        // ByteTrack specific thresholds
        _trackHighThresh = NumOps.ToDouble(options.ConfidenceThreshold);
        _trackLowThresh = 0.1;
        _newTrackThresh = Math.Max(_trackHighThresh, 0.3);
        _trackBuffer = options.MaxAge;
        _matchThresh = 0.8;
    }

    /// <inheritdoc/>
    public override TrackingResult<T> Update(List<Detection<T>> detections)
    {
        var startTime = DateTime.UtcNow;
        FrameCount++;

        // Split detections into high and low confidence
        var highConfDets = new List<Detection<T>>();
        var lowConfDets = new List<Detection<T>>();

        foreach (var det in detections)
        {
            double conf = NumOps.ToDouble(det.Confidence);
            if (conf >= _trackHighThresh)
            {
                highConfDets.Add(det);
            }
            else if (conf >= _trackLowThresh)
            {
                lowConfDets.Add(det);
            }
        }

        // Step 1: Add newly detected tracklets to tracked_stracks
        var unconfirmedTracks = new List<STrack<T>>();
        var trackedTracks = new List<STrack<T>>();

        foreach (var track in _trackedTracks)
        {
            if (!track.IsActivated)
            {
                unconfirmedTracks.Add(track);
            }
            else
            {
                trackedTracks.Add(track);
            }
        }

        // Step 2: Merge tracked tracks and lost tracks
        var sTrackPool = JoinTracks(trackedTracks, _lostTracks);

        // Predict locations
        foreach (var track in sTrackPool)
        {
            track.Predict();
        }

        // Step 3: First association with high confidence detections
        var (matchedPairs1, unmatchedTracks1, unmatchedDets1) =
            Associate(sTrackPool, highConfDets, _matchThresh);

        // Update matched tracks
        foreach (var (trackIdx, detIdx) in matchedPairs1)
        {
            sTrackPool[trackIdx].Update(highConfDets[detIdx], FrameCount);
        }

        // Step 4: Second association with low confidence detections
        var remainingTracks = unmatchedTracks1.Select(i => sTrackPool[i]).ToList();
        var (matchedPairs2, unmatchedTracks2, _) =
            Associate(remainingTracks, lowConfDets, 0.5); // Lower threshold for low conf

        foreach (var (trackIdx, detIdx) in matchedPairs2)
        {
            remainingTracks[trackIdx].Update(lowConfDets[detIdx], FrameCount);
        }

        // Mark remaining as lost
        var stillUnmatched = unmatchedTracks2.Select(i => remainingTracks[i]).ToList();
        foreach (var track in stillUnmatched)
        {
            if (!track.State.Equals(TrackState.Deleted))
            {
                track.MarkLost();
            }
        }

        // Step 5: Deal with unconfirmed tracks
        var (matchedPairs3, unmatchedUnconfirmed, unmatchedDets3) =
            Associate(unconfirmedTracks, highConfDets, 0.7);

        // Filter to only use remaining detections from first high-conf match
        var remainingHighDets = unmatchedDets1.Select(i => highConfDets[i]).ToList();
        var (matchedPairs3b, unmatchedUnconfirmed2, unmatchedDets3b) =
            Associate(unconfirmedTracks, remainingHighDets, 0.7);

        foreach (var (trackIdx, detIdx) in matchedPairs3b)
        {
            unconfirmedTracks[trackIdx].Update(remainingHighDets[detIdx], FrameCount);
        }

        // Remove unconfirmed tracks that weren't matched
        var tracksToRemove = unmatchedUnconfirmed2.Select(i => unconfirmedTracks[i]).ToList();
        foreach (var track in tracksToRemove)
        {
            track.MarkRemoved();
        }

        // Step 6: Init new tracks from unmatched high confidence detections
        foreach (int detIdx in unmatchedDets3b)
        {
            var det = remainingHighDets[detIdx];
            if (NumOps.ToDouble(det.Confidence) >= _newTrackThresh)
            {
                var newTrack = new STrack<T>(NextTrackId++, det, FrameCount, NumOps);
                newTrack.Activate(FrameCount);
                _trackedTracks.Add(newTrack);
            }
        }

        // Update track lists
        UpdateTrackLists();

        // Build result
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

    private (List<(int trackIdx, int detIdx)> matched, List<int> unmatchedTracks, List<int> unmatchedDets)
        Associate(List<STrack<T>> tracks, List<Detection<T>> detections, double iouThresh)
    {
        var matched = new List<(int trackIdx, int detIdx)>();
        var unmatchedTracks = new List<int>();
        var unmatchedDets = new List<int>();

        if (tracks.Count == 0)
        {
            unmatchedDets = Enumerable.Range(0, detections.Count).ToList();
            return (matched, unmatchedTracks, unmatchedDets);
        }

        if (detections.Count == 0)
        {
            unmatchedTracks = Enumerable.Range(0, tracks.Count).ToList();
            return (matched, unmatchedTracks, unmatchedDets);
        }

        // Build IoU cost matrix
        var costMatrix = new double[tracks.Count, detections.Count];

        for (int t = 0; t < tracks.Count; t++)
        {
            var trackBox = tracks[t].GetPredictedBox();

            for (int d = 0; d < detections.Count; d++)
            {
                double iou = ComputeIoU(trackBox, detections[d].Box);
                costMatrix[t, d] = 1.0 - iou;
            }
        }

        // Hungarian assignment
        var assignment = HungarianAssignment(costMatrix);

        var matchedDetSet = new HashSet<int>();
        var matchedTrackSet = new HashSet<int>();

        for (int t = 0; t < assignment.Length; t++)
        {
            int d = assignment[t];
            if (d >= 0 && costMatrix[t, d] < (1.0 - iouThresh))
            {
                matched.Add((t, d));
                matchedTrackSet.Add(t);
                matchedDetSet.Add(d);
            }
        }

        for (int t = 0; t < tracks.Count; t++)
        {
            if (!matchedTrackSet.Contains(t))
            {
                unmatchedTracks.Add(t);
            }
        }

        for (int d = 0; d < detections.Count; d++)
        {
            if (!matchedDetSet.Contains(d))
            {
                unmatchedDets.Add(d);
            }
        }

        return (matched, unmatchedTracks, unmatchedDets);
    }

    private List<STrack<T>> JoinTracks(List<STrack<T>> tracksA, List<STrack<T>> tracksB)
    {
        var result = new List<STrack<T>>(tracksA);
        var existingIds = tracksA.Select(t => t.TrackId).ToHashSet();

        foreach (var track in tracksB)
        {
            if (!existingIds.Contains(track.TrackId))
            {
                result.Add(track);
            }
        }

        return result;
    }

    private void UpdateTrackLists()
    {
        // Move lost tracks that have been lost too long to removed
        var newLost = new List<STrack<T>>();
        foreach (var track in _lostTracks)
        {
            if (FrameCount - track.EndFrame > _trackBuffer)
            {
                track.MarkRemoved();
                _removedTracks.Add(track);
            }
            else
            {
                newLost.Add(track);
            }
        }
        _lostTracks.Clear();
        _lostTracks.AddRange(newLost);

        // Process tracked tracks
        var newTracked = new List<STrack<T>>();
        foreach (var track in _trackedTracks)
        {
            if (track.State == TrackState.Deleted)
            {
                _lostTracks.Add(track);
            }
            else
            {
                newTracked.Add(track);
            }
        }
        _trackedTracks.Clear();
        _trackedTracks.AddRange(newTracked);

        // Move re-activated tracks from lost to tracked
        var reactivated = new List<STrack<T>>();
        foreach (var track in _lostTracks)
        {
            if (track.IsActivated && track.State == TrackState.Confirmed)
            {
                reactivated.Add(track);
            }
        }

        foreach (var track in reactivated)
        {
            _lostTracks.Remove(track);
            _trackedTracks.Add(track);
        }
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _trackedTracks.Clear();
        _lostTracks.Clear();
        _removedTracks.Clear();
    }
}

/// <summary>
/// Single track for ByteTrack (STrack).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class STrack<T>
{
    private readonly INumericOperations<T> _numOps;
    private int _classId;
    private T _confidence;

    // State: [cx, cy, w, h, vx, vy, vw, vh]
    private double[] _state;

    public int TrackId { get; }
    public int StartFrame { get; private set; }
    public int EndFrame { get; private set; }
    public int TrackletLen { get; private set; }
    public bool IsActivated { get; private set; }
    public TrackState State { get; private set; }
    public double Score { get; private set; }

    public STrack(int trackId, Detection<T> detection, int frameId, INumericOperations<T> numOps)
    {
        _numOps = numOps;
        TrackId = trackId;
        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        Score = numOps.ToDouble(detection.Confidence);

        // Initialize state
        var (cx, cy, w, h) = BoxToState(detection.Box);
        _state = new double[] { cx, cy, w, h, 0, 0, 0, 0 };

        StartFrame = frameId;
        EndFrame = frameId;
        TrackletLen = 1;
        IsActivated = false;
        State = TrackState.Tentative;
    }

    public void Activate(int frameId)
    {
        IsActivated = true;
        StartFrame = frameId;
        State = TrackState.Confirmed;
    }

    public void Predict()
    {
        // Apply velocity
        _state[0] += _state[4]; // cx += vx
        _state[1] += _state[5]; // cy += vy
        _state[2] += _state[6]; // w += vw
        _state[3] += _state[7]; // h += vh

        // Ensure positive dimensions
        _state[2] = Math.Max(1, _state[2]);
        _state[3] = Math.Max(1, _state[3]);
    }

    public void Update(Detection<T> detection, int frameId)
    {
        var (cx, cy, w, h) = BoxToState(detection.Box);

        // Update velocity
        _state[4] = 0.6 * _state[4] + 0.4 * (cx - _state[0]);
        _state[5] = 0.6 * _state[5] + 0.4 * (cy - _state[1]);
        _state[6] = 0.6 * _state[6] + 0.4 * (w - _state[2]);
        _state[7] = 0.6 * _state[7] + 0.4 * (h - _state[3]);

        // Update state
        _state[0] = cx;
        _state[1] = cy;
        _state[2] = w;
        _state[3] = h;

        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        Score = _numOps.ToDouble(detection.Confidence);
        EndFrame = frameId;
        TrackletLen++;

        if (!IsActivated)
        {
            Activate(frameId);
        }

        State = TrackState.Confirmed;
    }

    public void MarkLost()
    {
        State = TrackState.Deleted;
    }

    public void MarkRemoved()
    {
        State = TrackState.Deleted;
        IsActivated = false;
    }

    public BoundingBox<T> GetPredictedBox()
    {
        double cx = _state[0];
        double cy = _state[1];
        double w = _state[2];
        double h = _state[3];

        double x1 = cx - w / 2;
        double y1 = cy - h / 2;
        double x2 = cx + w / 2;
        double y2 = cy + h / 2;

        return new BoundingBox<T>(
            _numOps.FromDouble(x1), _numOps.FromDouble(y1),
            _numOps.FromDouble(x2), _numOps.FromDouble(y2),
            BoundingBoxFormat.XYXY);
    }

    public Track<T> ToTrack()
    {
        var box = GetPredictedBox();
        return new Track<T>(TrackId, box, _classId, _confidence)
        {
            VelocityX = _state[4],
            VelocityY = _state[5],
            TimeSinceUpdate = 0,
            Age = TrackletLen,
            Hits = TrackletLen,
            State = State
        };
    }

    private (double cx, double cy, double w, double h) BoxToState(BoundingBox<T> box)
    {
        double x1 = _numOps.ToDouble(box.X1);
        double y1 = _numOps.ToDouble(box.Y1);
        double x2 = _numOps.ToDouble(box.X2);
        double y2 = _numOps.ToDouble(box.Y2);

        double w = x2 - x1;
        double h = y2 - y1;
        double cx = x1 + w / 2;
        double cy = y1 + h / 2;

        return (cx, cy, w, h);
    }
}
