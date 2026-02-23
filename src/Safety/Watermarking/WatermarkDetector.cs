using AiDotNet.Enums;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Unified watermark detector that combines multiple watermark detection strategies.
/// </summary>
/// <remarks>
/// <para>
/// Runs sampling, lexical, and syntactic watermark detectors in parallel and aggregates
/// their scores to determine whether text contains any type of watermark.
/// </para>
/// <para>
/// <b>For Beginners:</b> This detector checks for all known types of text watermarks
/// at once. It combines multiple detection methods to be more accurate than any single
/// method alone.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class WatermarkDetector<T> : TextWatermarkerBase<T>
{
    private readonly SamplingWatermarker<T> _sampling;
    private readonly LexicalWatermarker<T> _lexical;
    private readonly SyntacticWatermarker<T> _syntactic;

    /// <inheritdoc />
    public override string ModuleName => "WatermarkDetector";

    /// <summary>
    /// Initializes a new composite watermark detector.
    /// </summary>
    public WatermarkDetector() : base(0.5)
    {
        _sampling = new SamplingWatermarker<T>();
        _lexical = new LexicalWatermarker<T>();
        _syntactic = new SyntacticWatermarker<T>();
    }

    /// <inheritdoc />
    public override double DetectWatermark(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return 0;

        double samplingScore = _sampling.DetectWatermark(text);
        double lexicalScore = _lexical.DetectWatermark(text);
        double syntacticScore = _syntactic.DetectWatermark(text);

        // Take the maximum score across all detectors
        return Math.Max(samplingScore, Math.Max(lexicalScore, syntacticScore));
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        double score = DetectWatermark(text);

        if (score >= 0.3)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = score >= 0.7 ? SafetySeverity.Medium : SafetySeverity.Info,
                Confidence = score,
                Description = $"Text watermark detected by composite detector (confidence: {score:F3}). " +
                              "Content may be AI-generated with embedded provenance watermark.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }
}
