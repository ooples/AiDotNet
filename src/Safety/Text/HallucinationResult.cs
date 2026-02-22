namespace AiDotNet.Safety.Text;

/// <summary>
/// Detailed result from hallucination detection with per-claim verdicts.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> HallucinationResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class HallucinationResult
{
    /// <summary>Overall hallucination score (0.0 = grounded, 1.0 = fabricated).</summary>
    public double OverallScore { get; init; }

    /// <summary>Whether the content is likely hallucinated.</summary>
    public bool IsHallucinated { get; init; }

    /// <summary>Per-claim verdicts with explanations.</summary>
    public IReadOnlyList<ClaimVerdict> Claims { get; init; } = Array.Empty<ClaimVerdict>();
}

/// <summary>
/// Verdict for a single claim in a hallucination check.
/// </summary>
public class ClaimVerdict
{
    /// <summary>The claim text.</summary>
    public string Claim { get; init; } = string.Empty;

    /// <summary>Whether this claim is supported by the reference/context.</summary>
    public bool IsSupported { get; init; }

    /// <summary>Confidence in the verdict (0.0-1.0).</summary>
    public double Confidence { get; init; }

    /// <summary>Explanation of the verdict.</summary>
    public string Explanation { get; init; } = string.Empty;
}
