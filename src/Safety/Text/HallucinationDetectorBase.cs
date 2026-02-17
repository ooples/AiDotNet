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
        // Default implementation delegates to standard text evaluation.
        // Subclasses that support reference-based checking should override this.
        return EvaluateText(generatedText);
    }

    /// <summary>
    /// Splits text into individual claims (sentences) for per-claim evaluation.
    /// </summary>
    protected static string[] ExtractClaims(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<string>();

        return text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
    }
}
