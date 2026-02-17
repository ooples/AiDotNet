namespace AiDotNet.Safety.Text;

/// <summary>
/// Abstract base class for hallucination detection modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for hallucination detectors including claim
/// extraction utilities and common scoring logic. Concrete implementations provide
/// the actual detection algorithm (reference-based, self-consistency, triplet, entailment).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all hallucination
/// detectors. Each detector type extends this and adds its own way of checking
/// whether an AI made something up.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class HallucinationDetectorBase<T> : TextSafetyModuleBase<T>, IHallucinationDetector<T>
{
    /// <inheritdoc />
    public abstract double GetHallucinationScore(string text);

    /// <inheritdoc />
    public virtual IReadOnlyList<SafetyFinding> EvaluateAgainstReference(string generatedText, string referenceText)
    {
        if (string.IsNullOrWhiteSpace(referenceText))
        {
            // No reference available — fall back to standard evaluation
            return EvaluateText(generatedText);
        }

        // Default reference-based evaluation: extract claims from generated text
        // and check whether they appear in the reference text.
        var findings = new List<SafetyFinding>();
        var claims = ExtractClaims(generatedText);
        string referenceLower = referenceText.ToLowerInvariant();

        foreach (string claim in claims)
        {
            if (claim.Length < 10) continue; // Skip very short fragments

            // Simple reference check: if the claim text doesn't appear in the reference,
            // it may be hallucinated. Subclasses should override with semantic matching.
            if (!referenceLower.Contains(claim.ToLowerInvariant()))
            {
                findings.Add(new SafetyFinding
                {
                    Category = Enums.SafetyCategory.Hallucination,
                    Severity = Enums.SafetySeverity.Medium,
                    Confidence = 0.5,
                    Description = $"Claim not found in reference text: \"{(claim.Length > 80 ? claim.Substring(0, 80) + "..." : claim)}\"",
                    RecommendedAction = Enums.SafetyAction.Warn,
                    SourceModule = ModuleName
                });
            }
        }

        return findings.Count > 0 ? findings : EvaluateText(generatedText);
    }

    /// <summary>
    /// Splits text into individual claims (sentences) for per-claim evaluation.
    /// </summary>
    protected static string[] ExtractClaims(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<string>();

        var raw = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        var trimmed = new string[raw.Length];
        for (int i = 0; i < raw.Length; i++)
        {
            trimmed[i] = raw[i].Trim();
        }
        return trimmed;
    }
}
