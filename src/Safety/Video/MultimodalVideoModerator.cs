using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Harmful YouTube Video Detection: A Taxonomy of Online Harm and MLLMs as Alternative Annotators",
    "https://arxiv.org/abs/2411.05854",
    Year = 2024,
    Authors = "Claire Wonjeong Jo, Miki Wesolowska, Magdalena Wojcieszak")]
public class MultimodalVideoModerator<T> : VideoSafetyModuleBase<T>
{

    private readonly CLIPImageSafetyClassifier<T> _imageClassifier;
    private readonly double _deepfakeThreshold;
    private readonly double _sceneChangeThreshold;

    /// <inheritdoc />
    public override string ModuleName => "MultimodalVideoModerator";

    /// <summary>
    /// Initializes a new multimodal video moderator.
    /// </summary>
    /// <param name="deepfakeThreshold">Deepfake detection threshold (0-1). Default: 0.6.</param>
    /// <param name="sceneChangeThreshold">Scene change sensitivity (0-1). Default: 0.3.</param>
    /// <param name="nsfwThreshold">NSFW detection threshold for the image classifier. Default: 0.8.</param>
    /// <param name="violenceThreshold">Violence detection threshold for the image classifier. Default: 0.75.</param>
    public MultimodalVideoModerator(
        double deepfakeThreshold = 0.6,
        double sceneChangeThreshold = 0.3,
        double nsfwThreshold = 0.8,
        double violenceThreshold = 0.75)
        : base(30.0)
    {
        _deepfakeThreshold = deepfakeThreshold;
        _sceneChangeThreshold = sceneChangeThreshold;
        _imageClassifier = new CLIPImageSafetyClassifier<T>(nsfwThreshold, violenceThreshold);
    }

    /// <inheritdoc />
    /// <summary>
    /// Number of image frames sampled per video, matching the paper's annotation budget: "14 image
    /// frames, 1 thumbnail, and text metadata" were fed to the model for each of 19,422 videos.
    /// </summary>
    public const int TaxonomyFrameBudget = 14;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Samples the paper's fixed budget of <see cref="TaxonomyFrameBudget"/> frames plus a
    /// thumbnail, gathers every signal those frames raise, folds the signals into the six-category
    /// harm taxonomy via <see cref="HarmTaxonomyMap"/>, and emits ONE video-level finding per harm
    /// category present. Categories are NON-MUTUALLY EXCLUSIVE, so a video may be reported under
    /// several at once — the paper's second taxonomy principle, illustrated there by a video that
    /// "narrates hate speech towards women while showing clips of women being punched" being both
    /// hate and harassment and physical harm.
    /// </para>
    /// <para>
    /// The forensic analyses below (temporal consistency, scene transitions, motion) are retained
    /// and routed through the same taxonomy: manipulation signals evidence Information harm, which
    /// is where the paper places deceptive content.
    /// </para>
    /// <para>
    /// <b>Not implemented, deliberately:</b> the paper's annotator also consumes text metadata
    /// (title, channel name, description, transcript) and audio. This interface receives frames
    /// only, so the text and audio pathways of the taxonomy cannot be evaluated here and no attempt
    /// is made to fake them. Its majority-of-three vote ("three API keys for GPT and three
    /// crowdworkers, selecting the majority answer from each") is likewise absent: that exists to
    /// control sampling randomness in an LLM annotator, and repeating a deterministic classifier
    /// three times would produce three identical votes.
    /// </para>
    /// </remarks>
    public override IReadOnlyList<SafetyFinding> EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();
        if (frames is null || frames.Count == 0 || frameRate <= 0) return findings;

        // Signals gathered from every modality this interface exposes, then folded into the taxonomy.
        var signals = new List<SafetyFinding>();
        signals.AddRange(AnalyzeFrameContent(frames, frameRate));
        if (frames.Count >= 3) signals.AddRange(AnalyzeTemporalConsistency(frames, frameRate));
        if (frames.Count >= 4) signals.AddRange(AnalyzeSceneTransitions(frames, frameRate));
        if (frames.Count >= 5) signals.AddRange(AnalyzeMotionPatterns(frames, frameRate));

        // Multi-label fold: one finding per harm category, carrying its strongest evidence.
        var byHarm = new Dictionary<HarmCategory, (double Confidence, SafetyFinding Exemplar, int Count)>();
        foreach (var signal in signals)
        {
            var harm = HarmTaxonomyMap.ToHarmCategory(signal.Category);
            if (harm is null) continue;

            if (byHarm.TryGetValue(harm.Value, out var existing))
            {
                byHarm[harm.Value] = signal.Confidence > existing.Confidence
                    ? (signal.Confidence, signal, existing.Count + 1)
                    : (existing.Confidence, existing.Exemplar, existing.Count + 1);
            }
            else
            {
                byHarm[harm.Value] = (signal.Confidence, signal, 1);
            }
        }

        double videoMilliseconds = frames.Count / frameRate * 1000.0;
        foreach (HarmCategory harm in Enum.GetValues<HarmCategory>())
        {
            if (!byHarm.TryGetValue(harm, out var evidence)) continue;

            findings.Add(new SafetyFinding
            {
                Category = evidence.Exemplar.Category,
                Severity = evidence.Exemplar.Severity,
                Confidence = evidence.Confidence,
                Description =
                    $"{harm} harm detected from {evidence.Count} signal(s) across " +
                    $"{Math.Min(TaxonomyFrameBudget, frames.Count)} sampled frame(s) + thumbnail. " +
                    evidence.Exemplar.Description,
                RecommendedAction = evidence.Exemplar.RecommendedAction,
                SourceModule = ModuleName,
                SpanStart = 0,
                SpanEnd = (int)videoMilliseconds,
            });
        }

        return findings;
    }

    private IReadOnlyList<SafetyFinding> AnalyzeFrameContent(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();

        // The paper's fixed annotation budget: 14 image frames plus 1 thumbnail per video, however
        // long the video is. Index 0 stands in for the thumbnail, which on a video platform is a
        // separate asset this interface does not receive; the remaining budget is spread evenly.
        var sampled = new List<int> { 0 };
        int budget = Math.Min(TaxonomyFrameBudget, frames.Count);
        for (int k = 0; k < budget; k++)
        {
            int idx = (int)((long)k * frames.Count / budget);
            if (!sampled.Contains(idx)) sampled.Add(idx);
        }

        foreach (int i in sampled)
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
