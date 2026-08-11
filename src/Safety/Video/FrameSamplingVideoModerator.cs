using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Safety.Image;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Video;

/// <summary>
/// Video content moderator built on Temporal Segment Networks: sparse segment-wise sampling with a
/// segmental consensus over snippet scores, producing one video-level verdict per category.
/// </summary>
/// <remarks>
/// <para>
/// Wang et al., "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition"
/// (ECCV 2016, arXiv:1608.00859). The video is divided into <c>K</c> segments of EQUAL DURATION and
/// one snippet is sampled from each; a shared classifier scores every snippet; a segmental consensus
/// combines those scores into a single video-level score, which is only then thresholded. In the
/// paper's notation (Eq. 1):
/// </para>
/// <para>
/// <c>TSN(T_1, ..., T_K) = H(G(F(T_1; W), ..., F(T_K; W)))</c>
/// </para>
/// <para>
/// where <c>F</c> is the shared-weight classifier applied to snippet <c>T_k</c>, <c>G</c> is the
/// segmental consensus and <c>H</c> is the prediction. The ordering is the substance of the method:
/// consensus is applied BEFORE the prediction function, so the model is supervised at video level
/// rather than frame level.
/// </para>
/// <para>
/// <b>Why this differs from fixed-rate sampling.</b> Sampling at a fixed FPS makes the number of
/// examined frames scale with duration, so a long video is judged on many more samples than a short
/// one, and each frame is decided independently — one twitchy frame flags the whole video. Segment
/// -wise sampling examines a CONSTANT K snippets spread evenly across the video however long it is,
/// which is what "sparse temporal sampling" means, and consensus-before-decision means the verdict
/// reflects the video as a whole. The paper's motivation is exactly this: dense frame-level
/// sampling is both redundant and unable to model long-range structure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of checking a frame every second — which examines 10 frames of a
/// 10-second clip and 600 of a 10-minute one — this splits any video into a few equal chunks and
/// looks at one frame from each. The separate opinions are then averaged into a single score for
/// the whole video before deciding. That makes the decision stable: one odd-looking frame no longer
/// condemns an entire video, and a genuinely problematic video is caught because several chunks
/// agree.
/// </para>
/// <para>
/// <b>Sampling at inference.</b> The paper samples snippets randomly within each segment during
/// training and uses fixed sampling at test time. This module is inference-only and must be
/// reproducible — the same video has to yield the same verdict every time it is scanned — so it
/// takes the CENTRE frame of each segment.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Temporal Segment Networks: Towards Good Practices for Deep Action Recognition",
    "https://arxiv.org/abs/1608.00859",
    Year = 2016,
    Authors = "Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool")]
public partial class FrameSamplingVideoModerator<T> : VideoSafetyModuleBase<T>
{
    /// <summary>
    /// Number of segments K used when the caller does not specify one. The paper: "the number of
    /// snippets K is set to 3 according to previous works on temporal modeling".
    /// </summary>
    public const int DefaultSegmentCount = 3;

    private readonly CLIPImageSafetyClassifier<T> _imageClassifier;
    private readonly int _segmentCount;
    private readonly SegmentalConsensus _consensus;
    private readonly double _nsfwThreshold;
    private readonly double _violenceThreshold;

    /// <inheritdoc />
    public override string ModuleName => "FrameSamplingVideoModerator";

    /// <summary>
    /// Gets the number of equal-duration segments the video is divided into.
    /// </summary>
    public int SegmentCount => _segmentCount;

    /// <summary>
    /// Number of frames the last <see cref="EvaluateVideo"/> call actually classified.
    /// </summary>
    /// <remarks>
    /// THE COST PROPERTY, OBSERVABLE. SegmentCount only reports what the constructor was handed; it
    /// stays 3 even if the sampler degenerates into scanning every frame. This reports what was
    /// really examined -- one snippet per segment, so min(SegmentCount, frames.Count) and NOT a
    /// function of video length. That is the whole claim of sparse temporal sampling, and it is
    /// otherwise untestable from outside.
    /// </remarks>
    public int LastSampledFrameCount { get; private set; }

    /// <summary>
    /// Gets the segmental consensus function used to combine snippet scores.
    /// </summary>
    public SegmentalConsensus Consensus => _consensus;

    /// <summary>
    /// Initializes a TSN-style video moderator.
    /// </summary>
    /// <param name="segmentCount">
    /// Number of equal-duration segments K. Default <see cref="DefaultSegmentCount"/> (3, the
    /// paper's value). One snippet is examined per segment regardless of video length.
    /// </param>
    /// <param name="consensus">
    /// Segmental consensus function G. Default <see cref="SegmentalConsensus.Average"/>, which the
    /// paper found best.
    /// </param>
    /// <param name="nsfwThreshold">NSFW threshold applied to the CONSENSUS score.</param>
    /// <param name="violenceThreshold">Violence threshold applied to the CONSENSUS score.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="segmentCount"/> is not positive.</exception>
    public FrameSamplingVideoModerator(
        int segmentCount = DefaultSegmentCount,
        SegmentalConsensus consensus = SegmentalConsensus.Average,
        double nsfwThreshold = 0.8,
        double violenceThreshold = 0.75)
        : base(30.0)
    {
        if (segmentCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(segmentCount),
                segmentCount, "Segment count K must be positive.");
        }

        // THE THRESHOLDS GET THE SAME TREATMENT AS THE SEGMENT COUNT. Both were accepted
        // unvalidated, and both fail SILENTLY: above 1 no confidence can ever reach the threshold so
        // the category is unreachable, below 0 every score clears it so the category always fires. A
        // moderation threshold that silently disables a category is the worst shape of
        // misconfiguration to ship, so it fails at construction instead.
        if (nsfwThreshold < 0.0 || nsfwThreshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(nsfwThreshold), nsfwThreshold,
                "Threshold must be within [0, 1]. Above 1 the NSFW category can never fire; below 0 it always fires.");
        }

        if (violenceThreshold < 0.0 || violenceThreshold > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(violenceThreshold), violenceThreshold,
                "Threshold must be within [0, 1]. Above 1 the violence category can never fire; below 0 it always fires.");
        }

        _segmentCount = segmentCount;
        _consensus = consensus;
        _nsfwThreshold = nsfwThreshold;
        _violenceThreshold = violenceThreshold;
        _imageClassifier = new CLIPImageSafetyClassifier<T>(nsfwThreshold, violenceThreshold);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Implements Eq. 1: score every segment's snippet with the shared classifier, apply the
    /// segmental consensus per category, and emit one video-level finding per category that clears
    /// its threshold. Per-snippet evidence is retained in the description so a reviewer can still
    /// see where in the video the signal came from.
    /// </remarks>
    public override IReadOnlyList<SafetyFinding> EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();
        if (frames is null || frames.Count == 0 || frameRate <= 0) return findings;

        // --- Sparse temporal sampling: K segments of equal duration, one snippet from each ---
        int segments = Math.Min(_segmentCount, frames.Count);
        LastSampledFrameCount = segments;
        var snippetIndices = new int[segments];
        for (int k = 0; k < segments; k++)
        {
            // Segment k spans [k*N/K, (k+1)*N/K); take its centre so the scan is reproducible.
            int start = (int)((long)k * frames.Count / segments);
            int end = (int)((long)(k + 1) * frames.Count / segments);
            if (end <= start) end = start + 1;
            snippetIndices[k] = Math.Min(frames.Count - 1, start + (end - start) / 2);
        }

        // --- F(T_k; W): the SHARED classifier scores each snippet independently ---
        var perSnippet = new List<IReadOnlyList<SafetyFinding>>(segments);
        for (int k = 0; k < segments; k++)
        {
            perSnippet.Add(_imageClassifier.EvaluateImage(frames[snippetIndices[k]]));
        }

        // --- G: segmental consensus, per category, over the snippet scores ---
        // A category absent from a snippet scored zero for that snippet: it was examined and found
        // clean. Treating it as missing instead would average only over the snippets that fired and
        // turn a single flagged snippet into a full-confidence video verdict, which is precisely the
        // frame-level behaviour TSN replaces.
        var categories = new List<SafetyCategory>();
        foreach (var snippet in perSnippet)
        {
            foreach (var finding in snippet)
            {
                if (!categories.Contains(finding.Category)) categories.Add(finding.Category);
            }
        }

        foreach (var category in categories)
        {
            var scores = new double[segments];
            SafetyFinding? exemplar = null;
            var firedAt = new List<int>();

            for (int k = 0; k < segments; k++)
            {
                foreach (var finding in perSnippet[k])
                {
                    if (finding.Category != category) continue;
                    if (finding.Confidence > scores[k]) scores[k] = finding.Confidence;
                    exemplar ??= finding;
                    if (!firedAt.Contains(k)) firedAt.Add(k);
                }
            }

            double consensusScore = ApplyConsensus(scores, _consensus);
            double threshold = ThresholdFor(category);
            // Two separate conditions, deliberately. `consensusScore < threshold` keeps the owned
        // categories' gate exactly as it was -- a score EQUAL to the configured threshold still
        // clears it -- while `<= 0` expresses the "must be non-zero" requirement that the
        // double.Epsilon sentinel used to smuggle into the threshold value. Folding them into one
        // comparison would have silently tightened the nsfw/violence gates.
        if (exemplar is null || consensusScore <= 0.0 || consensusScore < threshold) continue;

            var timestamps = new List<string>(firedAt.Count);
            foreach (int k in firedAt)
            {
                timestamps.Add((snippetIndices[k] / frameRate).ToString("F1") + "s");
            }

            double videoSeconds = frames.Count / frameRate;
            findings.Add(new SafetyFinding
            {
                Category = exemplar.Category,
                Severity = exemplar.Severity,
                Confidence = consensusScore,
                Description =
                    $"Video flagged by {_consensus} consensus over {segments} segment(s): " +
                    $"score {consensusScore:F3} >= {threshold:F2}, from {firedAt.Count}/{segments} " +
                    $"segment(s) at {string.Join(", ", timestamps)}. {exemplar.Description}",
                RecommendedAction = exemplar.RecommendedAction,
                SourceModule = ModuleName,
                // The verdict is video-level, so the span is the whole video rather than one frame.
                SpanStart = 0,
                SpanEnd = (int)(videoSeconds * 1000),
            });
        }

        return findings;
    }

    /// <summary>Applies the segmental consensus function G to one category's snippet scores.</summary>
    private static double ApplyConsensus(double[] scores, SegmentalConsensus consensus)
    {
        if (scores.Length == 0) return 0.0;

        switch (consensus)
        {
            case SegmentalConsensus.Max:
            {
                double max = scores[0];
                for (int i = 1; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
                return max;
            }

            case SegmentalConsensus.WeightedAverage:
            {
                // Linearly increasing weights normalized to sum to 1, so later segments count more.
                double weighted = 0.0, weightSum = 0.0;
                for (int i = 0; i < scores.Length; i++)
                {
                    double w = i + 1;
                    weighted += w * scores[i];
                    weightSum += w;
                }
                return weightSum > 0 ? weighted / weightSum : 0.0;
            }

            case SegmentalConsensus.Average:
            default:
            {
                double sum = 0.0;
                for (int i = 0; i < scores.Length; i++) sum += scores[i];
                return sum / scores.Length;
            }
        }
    }

    /// <summary>
    /// The threshold the consensus score must clear for a category.
    /// </summary>
    /// <remarks>
    /// The per-snippet classifier has already applied its own thresholds, so any score reaching here
    /// cleared them at snippet level. Consensus is a second, video-level gate; requiring the
    /// averaged score to still clear the snippet threshold is what stops one flagged segment out of
    /// K from condemning the whole video.
    /// </remarks>
    private double ThresholdFor(SafetyCategory category)
    {
        if (category == SafetyCategory.SexualExplicit) return _nsfwThreshold;
        if (category == SafetyCategory.ViolenceGraphic) return _violenceThreshold;

        // A category this module does not own a threshold for already cleared the snippet-level
        // gate, so the only requirement left is that the consensus be non-zero. Returning
        // double.Epsilon expressed that as a magnitude bound, which it is not -- it is the smallest
        // positive subnormal, and reading it as "just above zero" invites a float-comparison bug the
        // moment anyone adjusts the comparison. Returning 0 states the intent directly, and the
        // caller compares with > so a zero consensus is still rejected.
        return 0.0;
    }
}
