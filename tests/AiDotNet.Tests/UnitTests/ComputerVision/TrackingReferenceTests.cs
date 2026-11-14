using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Tracking;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Reference behaviour of the trackers, hand-traced from the official implementations:
/// abewley/sort (SORT), nwojke/deep_sort (DeepSORT) and ifzhang/ByteTrack (BYTETracker).
/// </summary>
public class TrackingReferenceTests
{
    private const double Tolerance = 1e-9;

    private static Detection<double> Det(double x1, double y1, double x2, double y2, double score, int classId = 0)
        => new(new BoundingBox<double>(x1, y1, x2, y2, BoundingBoxFormat.XYXY), classId, score);

    // ---------------- ByteTrack ----------------

    [Fact]
    public void ByteTrack_FirstAssociation_MatchesWhenIouDistanceIsWithinMatchThresh()
    {
        // Frame 1 creates track 1. Frame 2: bbox_ious (+1 pixel widths) gives IoU 66/176 = 0.375; with score 0.9
        // the fused cost is 1 - 0.3375 = 0.6625 <= 0.8, so the official tracker keeps track 1 (master required
        // IoU > 0.8).
        var tracker = new ByteTrack<double>(new TrackingOptions<double>());
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        var result = tracker.Update(new List<Detection<double>> { Det(5, 0, 15, 10, 0.9) });

        var track = Assert.Single(result.Tracks);
        Assert.Equal(1, track.TrackId);
    }

    [Fact]
    public void ByteTrack_FirstAssociation_RejectsCostAboveMatchThresh()
    {
        // bbox_ious IoU 22/220 = 0.1: fused cost 0.91 > 0.8. Track 1 is lost; the detection starts track 2, which is not
        // activated outside frame 1, so nothing is reported.
        var tracker = new ByteTrack<double>(new TrackingOptions<double>());
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        var result = tracker.Update(new List<Detection<double>> { Det(9, 0, 19, 10, 0.9) });

        Assert.Empty(result.Tracks);
    }

    [Fact]
    public void ByteTrack_FuseScore_ScalesIouByDetectionScore()
    {
        // bbox_ious IoU 44/198 = 2/9, score 0.7: fused cost 1 - 1.4/9 = 0.844 > 0.8 (no match); unfused
        // 7/9 = 0.778 <= 0.8 (match).
        var fused = new ByteTrack<double>(new TrackingOptions<double>());
        fused.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        Assert.Empty(fused.Update(new List<Detection<double>> { Det(7, 0, 17, 10, 0.7) }).Tracks);

        var unfused = new ByteTrack<double>(new ByteTrackOptions<double> { FuseScore = false });
        unfused.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        var track = Assert.Single(unfused.Update(new List<Detection<double>> { Det(7, 0, 17, 10, 0.7) }).Tracks);
        Assert.Equal(1, track.TrackId);
    }

    [Fact]
    public void ByteTrack_IouMatrix_PixelInclusive_MatchesCythonBboxIous()
    {
        // cython_bbox bbox_overlaps: iw = min(x2) - max(x1) + 1, ih likewise, union of (w + 1)(h + 1) areas.
        // (0,0,10,10) vs (6,0,16,10): iw 5, ih 11, overlap 55, areas 121 + 121, IoU 55/187.
        // (0,0,10,10) vs (0,0,4,10): overlap 5 * 11 = 55, areas 121 + 55, IoU 55/121.
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        var b = new Tensor<double>(new double[] { 6, 0, 16, 10, 0, 0, 4, 10 }, new[] { 2, 4 });

        var iou = TrackingAssociation<double>.IouMatrix(a, b, BoxAreaConvention.PixelInclusive);

        Assert.Equal(55.0 / 187.0, iou[0, 0], Tolerance);
        Assert.Equal(55.0 / 121.0, iou[0, 1], Tolerance);
    }

    [Fact]
    public void ByteTrack_IouMatrix_Continuous_MatchesUltralytics()
    {
        // Continuous widths: overlap 4 * 10 = 40 over 100 + 100 - 40, and 4 * 10 over 100 + 40 - 40.
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        var b = new Tensor<double>(new double[] { 6, 0, 16, 10, 0, 0, 4, 10 }, new[] { 2, 4 });

        var iou = TrackingAssociation<double>.IouMatrix(a, b, BoxAreaConvention.Continuous);

        Assert.Equal(0.25, iou[0, 0], Tolerance);
        Assert.Equal(0.4, iou[0, 1], Tolerance);
    }

    [Fact]
    public void ByteTrack_DefaultConvention_IsTheOfficialPixelInclusiveIou()
    {
        // (0,0,10,10) then (6,0,16,10) at score 0.7. Official bbox_ious IoU 55/187: fused cost
        // 1 - 0.7 * 55/187 = 0.794 <= 0.8, so track 1 continues. Continuous IoU 0.25 would give 0.825 and lose it.
        Assert.Equal(BoxAreaConvention.PixelInclusive, new ByteTrackOptions<double>().BoxAreaConvention);
        var tracker = new ByteTrack<double>(new TrackingOptions<double>());
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });

        var track = Assert.Single(tracker.Update(new List<Detection<double>> { Det(6, 0, 16, 10, 0.7) }).Tracks);
        Assert.Equal(1, track.TrackId);
    }

    [Fact]
    public void ByteTrack_ContinuousConvention_UsesUltralyticsIou()
    {
        // Same frames with continuous IoU 0.25: fused cost 1 - 0.175 = 0.825 > 0.8, so track 1 is lost and the
        // new track is not activated outside frame 1; nothing is reported.
        var tracker = new ByteTrack<double>(new ByteTrackOptions<double> { BoxAreaConvention = BoxAreaConvention.Continuous });
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });

        Assert.Empty(tracker.Update(new List<Detection<double>> { Det(6, 0, 16, 10, 0.7) }).Tracks);
    }

    [Fact]
    public void ByteTrack_NewTrack_NeedsTrackThreshPlusPointOne()
    {
        // det_thresh = 0.5 + 0.1: a 0.55 detection is high-score but may not start a track.
        var tracker = new ByteTrack<double>(new TrackingOptions<double>());
        Assert.Empty(tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.55) }).Tracks);

        var accepted = new ByteTrack<double>(new TrackingOptions<double>());
        Assert.Single(accepted.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.65) }).Tracks);
    }

    [Fact]
    public void ByteTrack_ScoreEqualToTrackThresh_JoinsNeitherSet()
    {
        // remain_inds = scores > track_thresh and inds_second = 0.1 < scores < track_thresh.
        var tracker = new ByteTrack<double>(new TrackingOptions<double>());
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        var result = tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.5) });

        // Track 1 finds no detection in either round and is marked lost, so nothing is reported.
        Assert.Empty(result.Tracks);
    }

    [Fact]
    public void ByteTrack_LowScoreDetection_KeepsTrackAliveInSecondRound()
    {
        var tracker = new ByteTrack<double>(new TrackingOptions<double>());
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        var result = tracker.Update(new List<Detection<double>> { Det(1, 0, 11, 10, 0.3) });

        var track = Assert.Single(result.Tracks);
        Assert.Equal(1, track.TrackId);
        Assert.Equal(0.3, track.Confidence, Tolerance);
    }

    [Fact]
    public void ByteTrack_TrackBornAfterFirstFrame_IsReportedOnceConfirmed()
    {
        var tracker = new ByteTrack<double>(new TrackingOptions<double>());
        Assert.Empty(tracker.Update(new List<Detection<double>>()).Tracks);
        Assert.Empty(tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) }).Tracks);

        var track = Assert.Single(tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) }).Tracks);
        Assert.Equal(1, track.TrackId);
    }

    // ---------------- SORT ----------------

    [Fact]
    public void SORT_KalmanUpdate_MatchesFilterpyReference()
    {
        // z1 = (5, 5, 100, 1). After predict, P[0,0] = 10 + 1e4 + 1 = 10011 and P[0,4] = 1e4; with R[0,0] = 1
        // the gain is K[0,0] = 10011/10012 and K[4,0] = 1e4/10012 for the innovation (2, 0, 0, 0).
        var tracker = new SORT<double>(new TrackingOptions<double>());
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        var track = Assert.Single(tracker.Update(new List<Detection<double>> { Det(2, 0, 12, 10, 0.9) }).Tracks);

        Assert.Equal(20000.0 / 10012.0, track.VelocityX, Tolerance);
        Assert.Equal(0.0, track.VelocityY, Tolerance);
        Assert.Equal(2.0 * 10011.0 / 10012.0, track.Box.X1, Tolerance);
        Assert.Equal(10.0 + 2.0 * 10011.0 / 10012.0, track.Box.X2, Tolerance);
    }

    [Fact]
    public void SORT_Association_IsOptimalNotGreedy()
    {
        // IoU: T0-D0 0.818, T0-D1 0.667, T1-D0 0.333, T1-D1 0.111. Greedy takes T0-D0 and strands T1;
        // the optimal assignment pairs T0-D1 and T1-D0, both above 0.3, so no new track appears.
        var tracker = new SORT<double>(new TrackingOptions<double> { MinHits = 1 });
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9), Det(6, 0, 16, 10, 0.9) });
        var result = tracker.Update(new List<Detection<double>> { Det(1, 0, 11, 10, 0.9), Det(-2, 0, 8, 10, 0.9) });

        Assert.Equal(new[] { 1, 2 }, result.Tracks.Select(t => t.TrackId).OrderBy(id => id).ToArray());
    }

    [Fact]
    public void SORT_ReportsOnlyTracksMatchedThisFrame()
    {
        var tracker = new SORT<double>(new TrackingOptions<double> { MinHits = 1, MaxAge = 5 });
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 10, 0.9) });

        // time_since_update must be 0 for a tracker to be reported.
        Assert.Empty(tracker.Update(new List<Detection<double>>()).Tracks);
    }

    // ---------------- DeepSORT ----------------

    [Fact]
    public void XyahKalmanFilter_InitiateAndGate_MatchDeepSortReference()
    {
        // h = 10: std = (2/20 h, 2/20 h, 1e-2, 2/20 h, 10/160 h, 10/160 h, 1e-5, 10/160 h).
        var kf = new XyahKalmanFilter<double>();
        var (mean, covariance) = kf.Initiate(new Tensor<double>(new[] { 5.0, 5.0, 1.0, 10.0 }, new[] { 4 }));

        Assert.Equal(1.0, covariance[0, 0], Tolerance);
        Assert.Equal(1e-4, covariance[2, 2], Tolerance);
        Assert.Equal(0.390625, covariance[4, 4], Tolerance);
        Assert.Equal(1e-10, covariance[6, 6], 1e-15);
        Assert.Equal(0.0, covariance[0, 1], Tolerance);

        // Projected S[0,0] = 1 + (h/20)^2 = 1.25, so a 1-pixel x offset has squared distance 0.8.
        var measurements = new Tensor<double>(new[] { 6.0, 5.0, 1.0, 10.0, 5.0, 5.0, 1.0, 10.0 }, new[] { 2, 4 });
        var distance = kf.GatingDistance(mean, covariance, measurements);
        Assert.Equal(0.8, distance[0], 1e-9);
        Assert.Equal(0.0, distance[1], 1e-9);

        // Predict adds Q: P[0,0] = 1 + 0.390625 + (h/20)^2 = 1.640625.
        var (_, predicted) = kf.Predict(mean, covariance);
        Assert.Equal(1.640625, predicted[0, 0], Tolerance);
        Assert.Equal(0.390625, predicted[0, 4], Tolerance);
    }

    [Fact]
    public void DeepSORT_ConfirmsAfterNInitAndReportsUntilOneMissedFrame()
    {
        var tracker = new DeepSORT<double>(new TrackingOptions<double>());
        var det = new List<Detection<double>> { Det(0, 0, 10, 20, 0.9) };

        Assert.Empty(tracker.Update(det).Tracks);
        Assert.Empty(tracker.Update(det).Tracks);
        Assert.Equal(1, Assert.Single(tracker.Update(det).Tracks).TrackId);

        // deep_sort_app reports confirmed tracks with time_since_update <= 1.
        Assert.Single(tracker.Update(new List<Detection<double>>()).Tracks);
        Assert.Empty(tracker.Update(new List<Detection<double>>()).Tracks);
    }

    [Fact]
    public void DeepSORT_TentativeTrackMissedOnce_IsDeleted()
    {
        var tracker = new DeepSORT<double>(new TrackingOptions<double>());
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 20, 0.9) });
        tracker.Update(new List<Detection<double>>());

        // The tentative track was deleted, so the same box starts a new track (id 2).
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 20, 0.9) });
        tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 20, 0.9) });
        var track = Assert.Single(tracker.Update(new List<Detection<double>> { Det(0, 0, 10, 20, 0.9) }).Tracks);
        Assert.Equal(2, track.TrackId);
    }

    [Fact]
    public void LinearAssignment_WithCostLimit_MatchesLapjvExtendedCost()
    {
        // lapjv(extend_cost=True, cost_limit=0.8): a pair is kept only if its cost is <= 0.8 and pairing
        // beats leaving both sides unmatched (0.4 + 0.4).
        var cost = new Tensor<double>(new[] { 0.1, 0.9, 0.85, 0.7 }, new[] { 2, 2 });
        var (matches, unmatchedRows, unmatchedCols) = TrackingAssociation<double>.LinearAssignment(cost, 0.8);

        Assert.Equal(new[] { (0, 0), (1, 1) }, matches.Select(m => (m.Row, m.Col)).ToArray());
        Assert.Empty(unmatchedRows);
        Assert.Empty(unmatchedCols);

        var far = new Tensor<double>(new[] { 0.81 }, new[] { 1, 1 });
        var (none, rows, cols) = TrackingAssociation<double>.LinearAssignment(far, 0.8);
        Assert.Empty(none);
        Assert.Equal(new[] { 0 }, rows.ToArray());
        Assert.Equal(new[] { 0 }, cols.ToArray());
    }
}
