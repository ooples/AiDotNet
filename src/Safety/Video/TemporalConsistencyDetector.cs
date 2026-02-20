using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

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
/// 5. Measure temporal jitter — variation in frame difference magnitudes
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

    private readonly T _threshold;

    // Pre-computed constants
    private static readonly T Zero = NumOps.Zero;
    private static readonly T One = NumOps.One;

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

        _threshold = NumOps.FromDouble(threshold);
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
        T deepfakeScore = EstimateDeepfakeScore(temporalFeatures);

        if (NumOps.GreaterThanOrEquals(deepfakeScore, _threshold))
        {
            double scoreDouble = NumOps.ToDouble(deepfakeScore);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = SafetySeverity.Medium,
                Confidence = scoreDouble,
                Description = $"Video flagged as potentially manipulated/deepfake (score: {scoreDouble:F3}). " +
                              $"Temporal jitter: {NumOps.ToDouble(temporalFeatures.TemporalJitter):F4}, " +
                              $"Discontinuity ratio: {NumOps.ToDouble(temporalFeatures.DiscontinuityRatio):F4}, " +
                              $"Periodicity: {NumOps.ToDouble(temporalFeatures.Periodicity):F4}.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private TemporalFeatures ComputeTemporalFeatures(IReadOnlyList<Tensor<T>> frames)
    {
        int pairCount = frames.Count - 1;
        var frameDiffs = new Vector<T>(pairCount);
        T totalDiffSum = Zero;
        T totalDiffSumSq = Zero;

        for (int f = 0; f < pairCount; f++)
        {
            T diff = ComputeFrameDifference(frames[f], frames[f + 1]);
            frameDiffs[f] = diff;
            totalDiffSum = NumOps.Add(totalDiffSum, diff);
            totalDiffSumSq = NumOps.Add(totalDiffSumSq, NumOps.Multiply(diff, diff));
        }

        T pairCountT = NumOps.FromDouble(pairCount);
        T meanDiff = NumOps.Divide(totalDiffSum, pairCountT);
        T diffVariance = NumOps.Subtract(
            NumOps.Divide(totalDiffSumSq, pairCountT),
            NumOps.Multiply(meanDiff, meanDiff));

        // Compute jitter: variation in frame differences (should be smooth for real video)
        T jitterSum = Zero;
        for (int f = 1; f < pairCount; f++)
        {
            T delta = NumOps.Subtract(frameDiffs[f], frameDiffs[f - 1]);
            jitterSum = NumOps.Add(jitterSum, NumOps.Multiply(delta, delta));
        }

        T jitter = pairCount > 1
            ? NumOps.Divide(jitterSum, NumOps.FromDouble(pairCount - 1))
            : Zero;

        // Detect discontinuities: frames where difference is > 3x the mean
        int discontinuities = 0;
        T threeX = NumOps.Multiply(NumOps.FromDouble(3.0), meanDiff);
        T epsilon = NumOps.FromDouble(1e-10);
        T effectiveThreshold = NumOps.GreaterThan(threeX, epsilon) ? threeX : epsilon;

        for (int f = 0; f < pairCount; f++)
        {
            if (NumOps.GreaterThan(frameDiffs[f], effectiveThreshold))
            {
                discontinuities++;
            }
        }

        T discontinuityRatio = NumOps.Divide(NumOps.FromDouble(discontinuities), pairCountT);

        // Detect periodicity in frame differences (deepfakes often have periodic patterns)
        T periodicity = ComputePeriodicity(frameDiffs, pairCount);

        return new TemporalFeatures
        {
            MeanFrameDifference = meanDiff,
            FrameDifferenceVariance = diffVariance,
            TemporalJitter = jitter,
            DiscontinuityRatio = discontinuityRatio,
            Periodicity = periodicity,
            FrameCount = frames.Count
        };
    }

    private T ComputeFrameDifference(Tensor<T> frame1, Tensor<T> frame2)
    {
        var span1 = frame1.Data.Span;
        var span2 = frame2.Data.Span;

        if (span1.Length == 0 || span2.Length == 0)
        {
            return Zero;
        }

        if (span1.Length != span2.Length)
        {
            throw new ArgumentException(
                $"Frame tensors have different sizes ({span1.Length} vs {span2.Length}). " +
                "All video frames must have the same dimensions.");
        }

        T sumAbsDiff = Zero;
        for (int i = 0; i < span1.Length; i++)
        {
            T diff = NumOps.Subtract(span1[i], span2[i]);
            sumAbsDiff = NumOps.Add(sumAbsDiff, NumOps.Abs(diff));
        }

        return NumOps.Divide(sumAbsDiff, NumOps.FromDouble(span1.Length));
    }

    /// <summary>
    /// Detects periodicity in frame difference signal via autocorrelation.
    /// Deepfakes often produce periodic frame-to-frame artifacts due to per-frame generation.
    /// </summary>
    private static T ComputePeriodicity(Vector<T> frameDiffs, int count)
    {
        if (count < 6) return Zero; // Need enough frames for meaningful autocorrelation

        // Compute mean
        T sum = Zero;
        for (int i = 0; i < count; i++)
        {
            sum = NumOps.Add(sum, frameDiffs[i]);
        }
        T mean = NumOps.Divide(sum, NumOps.FromDouble(count));

        // Compute autocorrelation at lag 0 (variance)
        T r0 = Zero;
        for (int i = 0; i < count; i++)
        {
            T d = NumOps.Subtract(frameDiffs[i], mean);
            r0 = NumOps.Add(r0, NumOps.Multiply(d, d));
        }

        T epsilon = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(r0, epsilon)) return Zero;

        // Find max autocorrelation for lags 2 to count/2 (skip lag 1 which is trivially high)
        T maxNormAutocorr = Zero;
        int maxLag = count / 2;

        for (int lag = 2; lag <= maxLag; lag++)
        {
            T rLag = Zero;
            for (int i = 0; i < count - lag; i++)
            {
                T d1 = NumOps.Subtract(frameDiffs[i], mean);
                T d2 = NumOps.Subtract(frameDiffs[i + lag], mean);
                rLag = NumOps.Add(rLag, NumOps.Multiply(d1, d2));
            }

            T normAutocorr = NumOps.Divide(rLag, r0);
            if (NumOps.GreaterThan(normAutocorr, maxNormAutocorr))
            {
                maxNormAutocorr = normAutocorr;
            }
        }

        // Periodicity: strong autocorrelation at some lag indicates periodic pattern
        // Natural video rarely shows strong periodicity in frame differences
        return maxNormAutocorr;
    }

    /// <summary>
    /// Estimates deepfake probability from temporal features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Weighted combination of temporal anomaly indicators:
    /// </para>
    /// <para>
    /// 1. <b>Temporal jitter (30%)</b>: Real video has smooth motion. Deepfakes show erratic
    ///    frame-to-frame changes due to independent per-frame generation. High jitter relative
    ///    to mean frame difference is suspicious.
    /// </para>
    /// <para>
    /// 2. <b>Discontinuity ratio (25%)</b>: Real video rarely has sudden jumps. A high ratio
    ///    of frames with differences exceeding 3x the mean indicates manipulation.
    /// </para>
    /// <para>
    /// 3. <b>Periodicity (25%)</b>: GAN-based deepfakes often introduce periodic artifacts
    ///    that appear as autocorrelation peaks in the frame difference signal.
    /// </para>
    /// <para>
    /// 4. <b>Variance anomaly (20%)</b>: The ratio of variance to mean in frame differences
    ///    differs between real and synthetic video. Real video follows more predictable patterns.
    /// </para>
    /// </remarks>
    private static T EstimateDeepfakeScore(TemporalFeatures features)
    {
        T epsilon = NumOps.FromDouble(1e-10);
        T meanDiff = features.MeanFrameDifference;

        // 1. Jitter score: ratio of jitter to mean^2 (coefficient of variation of differences)
        // Real video: smooth temporal changes → low jitter relative to mean
        // Deepfake: erratic changes → high jitter
        T meanSq = NumOps.Multiply(meanDiff, meanDiff);
        T effectiveMeanSq = NumOps.GreaterThan(meanSq, epsilon) ? meanSq : epsilon;
        T jitterRatio = NumOps.Divide(features.TemporalJitter, effectiveMeanSq);
        // Map: jitterRatio of 0 → 0, jitterRatio of 2+ → 1.0
        T jitterScore = Clamp01(NumOps.Divide(jitterRatio, NumOps.FromDouble(2.0)));

        // 2. Discontinuity score: direct mapping from ratio
        // Any discontinuity ratio > 0.15 is highly suspicious
        T discontinuityScore = Clamp01(
            NumOps.Divide(features.DiscontinuityRatio, NumOps.FromDouble(0.15)));

        // 3. Periodicity score: autocorrelation peak > 0.3 is suspicious
        // Natural video rarely exceeds 0.2 periodicity
        T periodicityScore = Clamp01(
            NumOps.Divide(features.Periodicity, NumOps.FromDouble(0.5)));

        // 4. Variance anomaly: high variance relative to mean indicates irregular motion
        T effectiveMean = NumOps.GreaterThan(meanDiff, epsilon) ? meanDiff : epsilon;
        T cvSquared = NumOps.Divide(features.FrameDifferenceVariance, NumOps.Multiply(effectiveMean, effectiveMean));
        // CV^2 > 1.0 is suspicious for frame differences in real video
        T varianceScore = Clamp01(NumOps.Divide(cvSquared, NumOps.FromDouble(2.0)));

        // Weighted combination
        T w1 = NumOps.FromDouble(0.30);
        T w2 = NumOps.FromDouble(0.25);
        T w3 = NumOps.FromDouble(0.25);
        T w4 = NumOps.FromDouble(0.20);

        T score = NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(w1, jitterScore),
                NumOps.Multiply(w2, discontinuityScore)),
            NumOps.Add(
                NumOps.Multiply(w3, periodicityScore),
                NumOps.Multiply(w4, varianceScore)));

        return Clamp01(score);
    }

    private static T Clamp01(T value)
    {
        if (NumOps.LessThan(value, Zero)) return Zero;
        if (NumOps.GreaterThan(value, One)) return One;
        return value;
    }

    private struct TemporalFeatures
    {
        public T MeanFrameDifference;
        public T FrameDifferenceVariance;
        public T TemporalJitter;
        public T DiscontinuityRatio;
        public T Periodicity;
        public int FrameCount;
    }
}
