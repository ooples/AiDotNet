namespace AiDotNet.Safety.Text;

/// <summary>
/// A match between generated text and potentially copyrighted content.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> CopyrightMatch provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class CopyrightMatch
{
    /// <summary>The matching text segment.</summary>
    public string MatchedText { get; init; } = string.Empty;

    /// <summary>Start character offset in the generated text.</summary>
    public int StartIndex { get; init; }

    /// <summary>Similarity score (0.0-1.0).</summary>
    public double Similarity { get; init; }

    /// <summary>Source of the potential copyright match, if known.</summary>
    public string Source { get; init; } = string.Empty;
}
