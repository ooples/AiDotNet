namespace AiDotNet.Safety.Text;

/// <summary>
/// Detailed result from copyright and memorization detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> CopyrightResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class CopyrightResult
{
    /// <summary>Overall memorization score (0.0 = original, 1.0 = memorized).</summary>
    public double MemorizationScore { get; init; }

    /// <summary>Whether the content likely infringes copyright.</summary>
    public bool IsCopyrightViolation { get; init; }

    /// <summary>Detected overlapping segments with known copyrighted content.</summary>
    public IReadOnlyList<CopyrightMatch> Matches { get; init; } = Array.Empty<CopyrightMatch>();
}
