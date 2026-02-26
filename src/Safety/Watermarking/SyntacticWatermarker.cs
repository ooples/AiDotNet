using AiDotNet.Enums;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Text watermarker that embeds watermarks through syntactic structure rearrangement.
/// </summary>
/// <remarks>
/// <para>
/// Encodes watermark bits through syntactic choices: active vs passive voice, clause
/// ordering, comma placement patterns, and sentence structure variations. Detection
/// analyzes the statistical distribution of syntactic patterns against expected distributions.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker changes the structure of sentences without
/// changing their meaning. For example, "The dog bit the man" vs "The man was bitten
/// by the dog" both mean the same thing but encode different watermark bits.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SyntacticWatermarker<T> : TextWatermarkerBase<T>
{
    // Structural patterns that encode bits
    private static readonly string[] ActivePatterns = { " is ", " are ", " was ", " were ", " has ", " have " };
    private static readonly string[] PassiveIndicators = { " by the ", " by a ", " been ", " being " };
    private static readonly string[] ConjunctionBefore = { ", and ", ", but ", ", or ", ", yet " };
    private static readonly string[] ConjunctionAfter = { " and ", " but ", " or ", " yet " };

    /// <inheritdoc />
    public override string ModuleName => "SyntacticWatermarker";

    /// <summary>
    /// Initializes a new syntactic watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Unused for syntactic approach; retained for API consistency.</param>
    public SyntacticWatermarker(double watermarkStrength = 0.5) : base(watermarkStrength) { }

    /// <inheritdoc />
    public override double DetectWatermark(string text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Length < 50) return 0;

        string lower = text.ToLowerInvariant();

        // Analyze passive vs active voice ratio
        int activeCount = 0;
        int passiveCount = 0;

        foreach (string p in ActivePatterns)
        {
            int idx = 0;
            while ((idx = lower.IndexOf(p, idx, StringComparison.Ordinal)) >= 0)
            {
                activeCount++;
                idx += p.Length;
            }
        }

        foreach (string p in PassiveIndicators)
        {
            int idx = 0;
            while ((idx = lower.IndexOf(p, idx, StringComparison.Ordinal)) >= 0)
            {
                passiveCount++;
                idx += p.Length;
            }
        }

        // Analyze conjunction placement (comma-before vs no comma)
        int commaConj = 0;
        int noCommaConj = 0;

        foreach (string p in ConjunctionBefore)
        {
            int idx = 0;
            while ((idx = lower.IndexOf(p, idx, StringComparison.Ordinal)) >= 0)
            {
                commaConj++;
                idx += p.Length;
            }
        }

        foreach (string p in ConjunctionAfter)
        {
            int idx = 0;
            while ((idx = lower.IndexOf(p, idx, StringComparison.Ordinal)) >= 0)
            {
                noCommaConj++;
                idx += p.Length;
            }
        }

        // Compute deviation from natural distribution
        int totalVoice = activeCount + passiveCount;
        int totalConj = commaConj + noCommaConj;

        if (totalVoice < 3 && totalConj < 3) return 0;

        double voiceDeviation = 0;
        if (totalVoice >= 3)
        {
            // Natural English: ~85% active, ~15% passive
            double passiveRate = (double)passiveCount / totalVoice;
            voiceDeviation = Math.Abs(passiveRate - 0.15) / 0.3;
        }

        double conjDeviation = 0;
        if (totalConj >= 3)
        {
            // Natural: ~40% comma-before, ~60% without comma
            double commaRate = (double)commaConj / totalConj;
            conjDeviation = Math.Abs(commaRate - 0.4) / 0.3;
        }

        double combined = Math.Max(voiceDeviation, conjDeviation);
        return Math.Max(0, Math.Min(1.0, combined));
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateText(string text)
    {
        var findings = new List<SafetyFinding>();
        double score = DetectWatermark(text);

        if (score >= 0.4)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = score,
                Description = $"Syntactic watermark pattern detected (confidence: {score:F3}). " +
                              "Unusual voice/structure distribution consistent with syntactic watermarking.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }
}
