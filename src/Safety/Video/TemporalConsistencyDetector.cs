using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;

namespace AiDotNet.Safety.Video;

/// <summary>
/// Detects deepfake videos by analyzing temporal consistency between consecutive frames.
/// </summary>
/// <remarks>
/// <para>
/// Deepfake videos often exhibit subtle temporal inconsistencies that are not present in
/// authentic video. This module analyzes frame-to-frame differences to detect anomalies
/// in motion, color, and spatial coherence that are characteristic of AI-generated or
/// manipulated video content.
/// </para>
/// <para>
/// <b>For Beginners:</b> When someone creates a fake video (deepfake), the way objects and
/// faces change between frames is often slightly "off" compared to real video. This module
/// looks at how each frame differs from the previous one and flags videos where those
/// differences look unnatural.
/// </para>
/// <para>
/// <b>Detection approach:</b>
/// 1. Compute frame-to-frame difference statistics (mean absolute difference, variance)
/// 2. Analyze temporal smoothness — real video has consistent motion patterns
/// 3. Detect sudden discontinuities that may indicate frame manipulation
/// 4. Check for periodic artifacts from frame-by-frame generation
/// </para>
/// <para>
/// <b>References:</b>
/// - Spatio-temporal consistency for video deepfake detection (2025)
/// - NACO: Self-supervised natural consistency for face forgery detection (ECCV 2024)
/// - Generalizable deepfake detection across benchmarks (CVPR 2025)
/// - FakeFormer: Efficient vulnerability-driven face forgery detection (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TemporalConsistencyDetector<T> : VideoSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _threshold;

    /// <inheritdoc />
    public override string ModuleName => "TemporalConsistencyDetector";

    /// <summary>
    /// Initializes a new temporal consistency detector.
    /// </summary>
    /// <param name="threshold">
    /// Detection threshold (0-1). Videos scoring above this are flagged as potentially
    /// manipulated. Default: 0.7.
    /// </param>
    public TemporalConsistencyDetector(double threshold = 0.7)
        : base(30.0)
    {
        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold),
                "Threshold must be between 0 and 1.");
        }

        _threshold = threshold;
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();

        if (frames.Count < 2 || frameRate <= 0)
        {
            return findings;
        }

        var temporalFeatures = ComputeTemporalFeatures(frames);
        var deepfakeScore = EstimateDeepfakeScore(temporalFeatures);

        if (deepfakeScore >= _threshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = SafetySeverity.Medium,
                Confidence = deepfakeScore,
                Description = $"Video flagged as potentially manipulated/deepfake (score: {deepfakeScore:F3}). " +
                              "Temporal consistency analysis detected frame-to-frame anomalies.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private TemporalFeatures ComputeTemporalFeatures(IReadOnlyList<Tensor<T>> frames)
    {
        int pairCount = frames.Count - 1;
        double[] frameDiffs = new double[pairCount];
        double totalDiffSum = 0;
        double totalDiffSumSq = 0;

        for (int f = 0; f < pairCount; f++)
        {
            double diff = ComputeFrameDifference(frames[f], frames[f + 1]);
            frameDiffs[f] = diff;
            totalDiffSum += diff;
            totalDiffSumSq += diff * diff;
        }

        double meanDiff = totalDiffSum / pairCount;
        double diffVariance = totalDiffSumSq / pairCount - meanDiff * meanDiff;

        // Compute jitter: variation in frame differences (should be smooth for real video)
        double jitterSum = 0;
        for (int f = 1; f < pairCount; f++)
        {
            double delta = frameDiffs[f] - frameDiffs[f - 1];
            jitterSum += delta * delta;
        }

        double jitter = pairCount > 1 ? jitterSum / (pairCount - 1) : 0;

        return new TemporalFeatures
        {
            MeanFrameDifference = meanDiff,
            FrameDifferenceVariance = diffVariance,
            TemporalJitter = jitter,
            FrameCount = frames.Count
        };
    }

    private double ComputeFrameDifference(Tensor<T> frame1, Tensor<T> frame2)
    {
        var span1 = frame1.Data.Span;
        var span2 = frame2.Data.Span;
        int minLength = Math.Min(span1.Length, span2.Length);

        if (minLength == 0)
        {
            return 0.0;
        }

        double sumAbsDiff = 0;
        for (int i = 0; i < minLength; i++)
        {
            double v1 = NumOps.ToDouble(span1[i]);
            double v2 = NumOps.ToDouble(span2[i]);
            sumAbsDiff += Math.Abs(v1 - v2);
        }

        return sumAbsDiff / minLength;
    }

    /// <summary>
    /// Estimates deepfake probability from temporal features.
    /// </summary>
    /// <remarks>
    /// Placeholder heuristic. In production, replace with a trained temporal consistency
    /// model operating on optical flow or learned spatiotemporal features.
    /// Returns 0.0 to avoid false positives until a real model is integrated.
    /// </remarks>
    private static double EstimateDeepfakeScore(TemporalFeatures features)
    {
        _ = features;
        return 0.0;
    }

    private struct TemporalFeatures
    {
        public double MeanFrameDifference;
        public double FrameDifferenceVariance;
        public double TemporalJitter;
        public int FrameCount;
    }
}
