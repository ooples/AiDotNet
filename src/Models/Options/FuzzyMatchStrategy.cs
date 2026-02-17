namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the similarity strategy used for fuzzy entity matching in PSI.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When entity IDs aren't perfectly identical across parties
/// (e.g., "John Smith" vs "Jon Smith"), fuzzy matching finds approximate matches.
/// Each strategy uses a different notion of "similarity":</para>
/// <list type="bullet">
/// <item><description><b>Exact:</b> IDs must be identical (no fuzzy matching).</description></item>
/// <item><description><b>EditDistance:</b> Counts the minimum character edits (insertions, deletions, substitutions) needed to transform one string into another.</description></item>
/// <item><description><b>Phonetic:</b> Matches strings that sound alike (e.g., "Smith" and "Smyth").</description></item>
/// <item><description><b>NGram:</b> Compares overlapping character sequences of length N.</description></item>
/// <item><description><b>Jaccard:</b> Measures set overlap between character or token sets.</description></item>
/// </list>
/// </remarks>
public enum FuzzyMatchStrategy
{
    /// <summary>
    /// Exact string equality. No fuzzy matching is performed.
    /// </summary>
    Exact = 0,

    /// <summary>
    /// Levenshtein edit distance. Counts minimum single-character edits
    /// (insertions, deletions, substitutions) to transform one string into another.
    /// A threshold defines the maximum allowable distance for a match.
    /// </summary>
    EditDistance = 1,

    /// <summary>
    /// Phonetic matching using Soundex or Double Metaphone algorithms.
    /// Groups strings by pronunciation, matching "Stephen" with "Steven".
    /// </summary>
    Phonetic = 2,

    /// <summary>
    /// Character n-gram similarity. Decomposes strings into overlapping character
    /// subsequences and measures the overlap. Good for handling typos and OCR errors.
    /// </summary>
    NGram = 3,

    /// <summary>
    /// Jaccard similarity coefficient between token or character sets.
    /// Measures |A intersect B| / |A union B|, ranging from 0.0 (no overlap) to 1.0 (identical).
    /// </summary>
    Jaccard = 4
}
