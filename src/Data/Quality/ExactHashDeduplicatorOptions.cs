namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for exact hash-based deduplication.
/// </summary>
public sealed class ExactHashDeduplicatorOptions
{
    /// <summary>Normalize whitespace before hashing. Default is true.</summary>
    public bool NormalizeWhitespace { get; set; } = true;
    /// <summary>Convert to lowercase before hashing. Default is true.</summary>
    public bool CaseInsensitive { get; set; } = true;
}
