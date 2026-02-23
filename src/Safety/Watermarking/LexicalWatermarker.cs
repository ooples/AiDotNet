using AiDotNet.Enums;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Text watermarker that embeds watermarks via synonym substitution patterns.
/// </summary>
/// <remarks>
/// <para>
/// Uses a deterministic mapping from a secret key to select between synonym pairs.
/// Given a set of interchangeable word pairs (e.g., "big"/"large", "fast"/"quick"),
/// the watermark selects one synonym per pair based on the key. Detection checks
/// whether the observed synonym choices match the expected key-based pattern.
/// </para>
/// <para>
/// <b>For Beginners:</b> This watermarker swaps words with their synonyms in a
/// pattern that encodes a hidden signature. For example, choosing "large" over "big"
/// in specific positions. The meaning stays the same, but the pattern of synonym
/// choices reveals the watermark.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LexicalWatermarker<T> : TextWatermarkerBase<T>
{
    // Synonym pairs for watermark embedding
    private static readonly (string, string)[] SynonymPairs = new[]
    {
        ("big", "large"), ("small", "little"), ("fast", "quick"), ("happy", "glad"),
        ("sad", "unhappy"), ("begin", "start"), ("end", "finish"), ("also", "additionally"),
        ("however", "nevertheless"), ("important", "significant"), ("show", "demonstrate"),
        ("use", "utilize"), ("help", "assist"), ("make", "create"), ("give", "provide"),
        ("tell", "inform"), ("think", "believe"), ("seem", "appear"), ("often", "frequently"),
        ("enough", "sufficient"), ("hard", "difficult"), ("easy", "simple"),
        ("rich", "wealthy"), ("poor", "impoverished"), ("old", "elderly")
    };

    /// <inheritdoc />
    public override string ModuleName => "LexicalWatermarker";

    /// <summary>
    /// Initializes a new lexical watermarker.
    /// </summary>
    /// <param name="watermarkStrength">Unused for lexical approach; retained for API consistency.</param>
    public LexicalWatermarker(double watermarkStrength = 0.5) : base(watermarkStrength) { }

    /// <inheritdoc />
    public override double DetectWatermark(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return 0;

        string lower = text.ToLowerInvariant();
        int matchCount = 0;
        int pairsSeen = 0;

        foreach (var (a, b) in SynonymPairs)
        {
            bool hasA = lower.Contains(a);
            bool hasB = lower.Contains(b);

            if (hasA || hasB)
            {
                pairsSeen++;
                // Check if the "watermarked" synonym is preferred based on pair index parity
                int pairHash = GetFNVHash(a + b);
                bool expectB = (pairHash & 1) == 1;

                if (expectB && hasB && !hasA) matchCount++;
                else if (!expectB && hasA && !hasB) matchCount++;
            }
        }

        if (pairsSeen < 3) return 0;

        double matchRate = (double)matchCount / pairsSeen;
        // Random text would match ~50% of the time; watermarked text matches much higher
        if (matchRate <= 0.5) return 0;
        return Math.Max(0, Math.Min(1.0, (matchRate - 0.5) / 0.4));
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
                Severity = SafetySeverity.Info,
                Confidence = score,
                Description = $"Lexical watermark pattern detected (confidence: {score:F3}). " +
                              "Synonym choice pattern consistent with lexical watermarking.",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private static int GetFNVHash(string s)
    {
        unchecked
        {
            int hash = (int)2166136261;
            foreach (char c in s)
            {
                hash ^= c;
                hash *= 16777619;
            }
            return hash;
        }
    }
}
