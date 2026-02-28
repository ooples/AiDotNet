namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for language identification filtering.
/// </summary>
/// <remarks>
/// Uses character n-gram frequency profiles for language detection.
/// </remarks>
public sealed class LanguageIdFilterOptions
{
    /// <summary>Target language codes to keep (e.g., "en", "fr"). Default is ["en"].</summary>
    public string[] TargetLanguages { get; set; } = ["en"];
    /// <summary>Minimum confidence score to accept detection. Default is 0.8.</summary>
    public double MinConfidence { get; set; } = 0.8;
    /// <summary>N-gram size for character-level language profiles. Default is 3.</summary>
    public int ProfileNGramSize { get; set; } = 3;
    /// <summary>Maximum number of n-grams to keep in language profile. Default is 300.</summary>
    public int MaxProfileSize { get; set; } = 300;
    /// <summary>Minimum text length (in characters) for reliable detection. Default is 50.</summary>
    public int MinTextLength { get; set; } = 50;
}
