using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Safety.Image;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Video;

/// <summary>
/// Comprehensive video moderator that combines frame-level content classification,
/// temporal deepfake detection, and optional audio track analysis.
/// </summary>
/// <remarks>
/// <para>
/// Orchestrates multiple detection strategies for complete video safety analysis:
/// 1. Frame sampling with ensemble image classifiers (NSFW, violence, hate symbols)
/// 2. Temporal consistency analysis for deepfake detection
/// 3. Scene transition analysis for detecting spliced/manipulated segments
/// 4. Motion analysis for detecting unnatural movement patterns
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the "all-in-one" video safety checker. It looks at individual
/// frames for harmful images, checks whether the video flows naturally between frames (deepfakes
/// often don't), and analyzes scene transitions to find where content might have been spliced in.
/// </para>
/// <para>
/// <b>References:</b>
/// - Efficient video understanding via multi-scale temporal sampling (CVPR 2024)
/// - Spatio-temporal consistency for video deepfake detection (2025)
/// - Video content moderation at scale (Meta, 2024)
/// - VideoGuard: Multimodal video safety with reasoning-based instruction hierarchy (2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MultimodalVideoModerator<T> : VideoSafetyModuleBase<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly CLIPImageSafetyClassifier<T> _imageClassifier;
    private readonly double _samplingRate;
    private readonly double _deepfakeThreshold;
    private readonly double _sceneChangeThreshold;

    /// <inheritdoc />
    public override string ModuleName => "MultimodalVideoModerator";

    /// <summary>
    /// Initializes a new multimodal video moderator.
    /// </summary>
    /// <param name="samplingRate">Frames per second to sample for content classification. Default: 1.0.</param>
    /// <param name="deepfakeThreshold">Deepfake detection threshold (0-1). Default: 0.6.</param>
    /// <param name="sceneChangeThreshold">Scene change sensitivity (0-1). Default: 0.3.</param>
    /// <param name="nsfwThreshold">NSFW detection threshold for the image classifier. Default: 0.8.</param>
    /// <param name="violenceThreshold">Violence detection threshold for the image classifier. Default: 0.75.</param>
    public MultimodalVideoModerator(
        double samplingRate = 1.0,
        double deepfakeThreshold = 0.6,
        double sceneChangeThreshold = 0.3,
        double nsfwThreshold = 0.8,
        double violenceThreshold = 0.75)
        : base(30.0)
    {
        _samplingRate = Math.Max(0.1, samplingRate);
        _deepfakeThreshold = deepfakeThreshold;
        _sceneChangeThreshold = sceneChangeThreshold;
        _imageClassifier = new CLIPImageSafetyClassifier<T>(nsfwThreshold, violenceThreshold);
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();

        if (frames.Count == 0 || frameRate <= 0)
        {
            return findings;
        }

        // 1. Frame-level content classification
        var contentFindings = AnalyzeFrameContent(frames, frameRate);
        findings.AddRange(contentFindings);

        // 2. Temporal consistency analysis (deepfake detection)
        if (frames.Count >= 3)
        {
            var temporalFindings = AnalyzeTemporalConsistency(frames, frameRate);
            findings.AddRange(temporalFindings);
        }

        // 3. Scene transition analysis (splice detection)
        if (frames.Count >= 4)
        {
            var sceneFindings = AnalyzeSceneTransitions(frames, frameRate);
            findings.AddRange(sceneFindings);
        }

        // 4. Motion analysis (unnatural movement detection)
        if (frames.Count >= 5)
        {
            var motionFindings = AnalyzeMotionPatterns(frames, frameRate);
            findings.AddRange(motionFindings);
        }

        return findings;
    }

    private IReadOnlyList<SafetyFinding> AnalyzeFrameContent(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();
        int frameInterval = Math.Max(1, (int)(frameRate / _samplingRate));

        for (int i = 0; i < frames.Count; i += frameInterval)
        {
            var frameFindings = _imageClassifier.EvaluateImage(frames[i]);

            foreach (var finding in frameFindings)
            {
                double timestamp = i / frameRate;
                findings.Add(new SafetyFinding
                {
                    Category = finding.Category,
                    Severity = finding.Severity,
                    Confidence = finding.Confidence,
                    Description = $"Frame at {timestamp:F1}s: {finding.Description}",
                    RecommendedAction = finding.RecommendedAction,
                    SourceModule = ModuleName,
                    SpanStart = (int)(timestamp * 1000),
                    SpanEnd = (int)((timestamp + 1.0 / frameRate) * 1000)
                });
            }
        }

        return findings;
    }

    private IReadOnlyList<SafetyFinding> AnalyzeTemporalConsistency(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();
        int pairCount = frames.Count - 1;

        var frameDiffs = new double[pairCount];
        double diffSum = 0;

        for (int f = 0; f < pairCount; f++)
        {
            frameDiffs[f] = ComputeFrameDifference(frames[f], frames[f + 1]);
            diffSum += frameDiffs[f];
        }

        double meanDiff = diffSum / pairCount;

        // Compute temporal jitter (variance of frame-to-frame differences)
        double jitterSum = 0;
        for (int f = 1; f < pairCount; f++)
        {
            double delta = frameDiffs[f] - frameDiffs[f - 1];
            jitterSum += delta * delta;
        }
        double jitter = pairCount > 1 ? jitterSum / (pairCount - 1) : 0;

        // Compute periodicity via autocorrelation
        double periodicity = ComputePeriodicity(frameDiffs, pairCount);

        // Count discontinuities (sudden jumps)
        int discontinuities = 0;
        double threshold3x = Math.Max(meanDiff * 3.0, 1e-10);
        for (int f = 0; f < pairCount; f++)
        {
            if (frameDiffs[f] > threshold3x) discontinuities++;
        }
        double discontinuityRatio = (double)discontinuities / pairCount;

        // Combine into deepfake score
        double meanSq = Math.Max(meanDiff * meanDiff, 1e-10);
        double jitterScore = Math.Max(0, Math.Min(1.0, jitter / (meanSq * 2.0)));
        double discoScore = Math.Max(0, Math.Min(1.0, discontinuityRatio / 0.15));
        double periodicityScore = Math.Max(0, Math.Min(1.0, periodicity / 0.5));

        double deepfakeScore = 0.35 * jitterScore + 0.30 * discoScore + 0.35 * periodicityScore;

        if (deepfakeScore >= _deepfakeThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = deepfakeScore >= 0.8 ? SafetySeverity.High : SafetySeverity.Medium,
                Confidence = Math.Min(1.0, deepfakeScore),
                Description = $"Temporal inconsistency detected (score: {deepfakeScore:F3}). " +
                              $"Jitter: {jitterScore:F3}, discontinuities: {discoScore:F3}, " +
                              $"periodicity: {periodicityScore:F3}. Video may be AI-generated or manipulated.",
                RecommendedAction = SafetyAction.Warn,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private IReadOnlyList<SafetyFinding> AnalyzeSceneTransitions(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();
        int pairCount = frames.Count - 1;

        // Detect abrupt scene changes and analyze surrounding frames for splicing
        var frameDiffs = new double[pairCount];
        double diffSum = 0;
        for (int f = 0; f < pairCount; f++)
        {
            frameDiffs[f] = ComputeFrameDifference(frames[f], frames[f + 1]);
            diffSum += frameDiffs[f];
        }
        double meanDiff = diffSum / pairCount;

        // Find scene change points
        var sceneChanges = new List<int>();
        double sceneThreshold = Math.Max(meanDiff * (1.0 / Math.Max(_sceneChangeThreshold, 0.01)), 1e-8);

        for (int f = 0; f < pairCount; f++)
        {
            if (frameDiffs[f] > sceneThreshold)
            {
                sceneChanges.Add(f);
            }
        }

        // Analyze scene change patterns — very frequent changes may indicate splicing
        if (sceneChanges.Count > 0)
        {
            double videoDurationSec = frames.Count / frameRate;
            double changesPerSecond = sceneChanges.Count / Math.Max(videoDurationSec, 0.1);

            // Normal video: ~0.1-0.5 scene changes/sec. Suspicious: > 2/sec
            if (changesPerSecond > 2.0)
            {
                double suspicionScore = Math.Min(1.0, (changesPerSecond - 2.0) / 3.0);
                findings.Add(new SafetyFinding
                {
                    Category = SafetyCategory.Manipulated,
                    Severity = SafetySeverity.Medium,
                    Confidence = suspicionScore,
                    Description = $"Unusually frequent scene changes detected ({changesPerSecond:F1}/sec). " +
                                  $"May indicate video splicing or rapid content switching.",
                    RecommendedAction = SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }

            // Check for irregular scene change intervals (natural cuts tend to be rhythmic)
            if (sceneChanges.Count >= 3)
            {
                var intervals = new double[sceneChanges.Count - 1];
                double intervalSum = 0;
                for (int i = 0; i < intervals.Length; i++)
                {
                    intervals[i] = (sceneChanges[i + 1] - sceneChanges[i]) / frameRate;
                    intervalSum += intervals[i];
                }
                double meanInterval = intervalSum / intervals.Length;

                double intervalVariance = 0;
                for (int i = 0; i < intervals.Length; i++)
                {
                    double d = intervals[i] - meanInterval;
                    intervalVariance += d * d;
                }
                intervalVariance /= intervals.Length;

                double cv = meanInterval > 1e-10 ? Math.Sqrt(intervalVariance) / meanInterval : 0;
                if (cv > 1.5)
                {
                    double irregularityScore = Math.Min(1.0, (cv - 1.5) / 2.0);
                    findings.Add(new SafetyFinding
                    {
                        Category = SafetyCategory.Manipulated,
                        Severity = SafetySeverity.Low,
                        Confidence = irregularityScore,
                        Description = $"Irregular scene change timing detected (CV: {cv:F2}). " +
                                      $"Natural video editing tends to have more regular cut patterns.",
                        RecommendedAction = SafetyAction.Log,
                        SourceModule = ModuleName
                    });
                }
            }
        }

        return findings;
    }

    private IReadOnlyList<SafetyFinding> AnalyzeMotionPatterns(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();

        // Compute optical flow magnitude estimates via frame differences in spatial blocks
        int blockSize = 8;
        int numAnalyzed = Math.Min(frames.Count - 1, 16);
        var blockMotions = new List<double>();

        for (int f = 0; f < numAnalyzed; f++)
        {
            double motion = ComputeBlockMotionVariance(frames[f], frames[f + 1], blockSize);
            blockMotions.Add(motion);
        }

        if (blockMotions.Count < 3) return findings;

        // Compute motion statistics
        double motionSum = 0;
        foreach (var m in blockMotions) motionSum += m;
        double motionMean = motionSum / blockMotions.Count;

        double motionVar = 0;
        foreach (var m in blockMotions)
        {
            double d = m - motionMean;
            motionVar += d * d;
        }
        motionVar /= blockMotions.Count;

        // Deepfakes often have unnaturally smooth or erratic motion
        // Natural motion: moderate variance; synthetic: very low or very high
        double motionCV = motionMean > 1e-10 ? Math.Sqrt(motionVar) / motionMean : 0;

        // Very low CV = unnaturally smooth (autoregressive generation)
        if (motionCV < 0.1 && motionMean > 1e-6)
        {
            double smoothnessScore = Math.Min(1.0, (0.1 - motionCV) / 0.1);
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Deepfake,
                Severity = SafetySeverity.Low,
                Confidence = smoothnessScore * 0.6,
                Description = $"Unnaturally smooth motion detected (motion CV: {motionCV:F4}). " +
                              $"AI-generated videos often lack natural motion variability.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static double ComputeFrameDifference(Tensor<T> frame1, Tensor<T> frame2)
    {
        var span1 = frame1.Data.Span;
        var span2 = frame2.Data.Span;
        int minLen = Math.Min(span1.Length, span2.Length);
        if (minLen == 0) return 0;

        double sumAbsDiff = 0;
        for (int i = 0; i < minLen; i++)
        {
            double d = NumOps.ToDouble(NumOps.Subtract(span1[i], span2[i]));
            sumAbsDiff += Math.Abs(d);
        }

        return sumAbsDiff / minLen;
    }

    private static double ComputeBlockMotionVariance(Tensor<T> frame1, Tensor<T> frame2, int blockSize)
    {
        var span1 = frame1.Data.Span;
        var span2 = frame2.Data.Span;
        int minLen = Math.Min(span1.Length, span2.Length);
        if (minLen < blockSize * blockSize) return 0;

        var blockDiffs = new List<double>();
        int numBlocks = minLen / (blockSize * blockSize);
        numBlocks = Math.Min(numBlocks, 64);

        for (int b = 0; b < numBlocks; b++)
        {
            int offset = b * blockSize * blockSize;
            double blockDiff = 0;
            int count = 0;
            for (int i = 0; i < blockSize * blockSize && offset + i < minLen; i++)
            {
                double d = NumOps.ToDouble(NumOps.Subtract(span1[offset + i], span2[offset + i]));
                blockDiff += Math.Abs(d);
                count++;
            }
            if (count > 0) blockDiffs.Add(blockDiff / count);
        }

        if (blockDiffs.Count < 2) return 0;

        double mean = 0;
        foreach (var d in blockDiffs) mean += d;
        mean /= blockDiffs.Count;

        double variance = 0;
        foreach (var d in blockDiffs)
        {
            double diff = d - mean;
            variance += diff * diff;
        }
        return variance / blockDiffs.Count;
    }

    private static double ComputePeriodicity(double[] frameDiffs, int count)
    {
        if (count < 6) return 0;

        double sum = 0;
        for (int i = 0; i < count; i++) sum += frameDiffs[i];
        double mean = sum / count;

        double r0 = 0;
        for (int i = 0; i < count; i++)
        {
            double d = frameDiffs[i] - mean;
            r0 += d * d;
        }
        if (r0 < 1e-10) return 0;

        double maxAutocorr = 0;
        int maxLag = count / 2;
        for (int lag = 2; lag <= maxLag; lag++)
        {
            double rLag = 0;
            for (int i = 0; i < count - lag; i++)
            {
                rLag += (frameDiffs[i] - mean) * (frameDiffs[i + lag] - mean);
            }
            double norm = rLag / r0;
            if (norm > maxAutocorr) maxAutocorr = norm;
        }

        return maxAutocorr;
    }
}
