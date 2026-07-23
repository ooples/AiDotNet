namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for fuzzy entity matching in Private Set Intersection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When entity IDs across parties aren't perfectly identical
/// (e.g., "John Smith" vs "Jon Smith", or "123-45-6789" vs "123456789"), fuzzy matching
/// finds approximate matches. These options control how approximate the matching can be.</para>
///
/// <para>Example configuration for matching patient names across hospitals:
/// <code>
/// var fuzzyOptions = new FuzzyMatchOptions
/// {
///     Strategy = FuzzyMatchStrategy.EditDistance,
///     Threshold = 2.0,     // Allow up to 2 character edits
///     CaseSensitive = false // "john" matches "John"
/// };
/// </code>
/// </para>
/// </remarks>
public class FuzzyMatchOptions
{
    /// <summary>
    /// Gets or sets the fuzzy matching strategy to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This selects which algorithm measures similarity between IDs.
    /// EditDistance is a good default for typos. Phonetic is better for name matching.
    /// NGram and Jaccard are good for longer strings with partial matches.</para>
    /// </remarks>
    public FuzzyMatchStrategy Strategy { get; set; } = FuzzyMatchStrategy.Exact;

    /// <summary>
    /// Gets or sets the similarity threshold for fuzzy matching.
    /// </summary>
    /// <remarks>
    /// <para>Interpretation depends on the strategy:</para>
    /// <list type="bullet">
    /// <item><description><b>EditDistance:</b> Maximum number of edits allowed (e.g., 2 means up to 2 character changes).</description></item>
    /// <item><description><b>NGram/Jaccard:</b> Minimum similarity score from 0.0 to 1.0 (e.g., 0.8 means 80% similar).</description></item>
    /// <item><description><b>Phonetic:</b> Ignored (phonetic codes either match or don't).</description></item>
    /// <item><description><b>Exact:</b> Ignored.</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> A higher threshold means stricter matching (fewer false positives but more missed matches).
    /// Start with 2 for EditDistance or 0.8 for NGram/Jaccard and adjust based on your data quality.</para>
    /// </remarks>
    public double Threshold { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the n-gram size for NGram matching strategy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> N-grams split a string into overlapping pieces of length N.
    /// For example, "hello" with N=2 becomes: {"he", "el", "ll", "lo"}.
    /// Smaller N catches more typos but increases false positives. 2 or 3 is typical.</para>
    /// </remarks>
    public int NGramSize { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether string comparisons are case-sensitive.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When false, "John" and "john" are treated as the same.
    /// Usually you want case-insensitive matching for names and identifiers.</para>
    /// </remarks>
    public bool CaseSensitive { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize whitespace before matching.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, extra spaces are removed and standardized.
    /// "John  Smith" becomes "John Smith" before comparison.</para>
    /// </remarks>
    public bool NormalizeWhitespace { get; set; } = true;
}
