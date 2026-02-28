namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for perplexity-based text quality filtering.
/// </summary>
public sealed class PerplexityFilterOptions
{
    /// <summary>Maximum perplexity threshold. Documents above this are filtered out. Default is 1000.</summary>
    public double MaxPerplexity { get; set; } = 1000.0;
    /// <summary>Minimum perplexity threshold. Documents below this may be boilerplate. Default is 0 (disabled).</summary>
    public double MinPerplexity { get; set; } = 0.0;
    /// <summary>N-gram order for the language model. Default is 3.</summary>
    public int NGramOrder { get; set; } = 3;
    /// <summary>Smoothing factor for unseen n-grams. Default is 1.0 (Laplace smoothing).</summary>
    public double SmoothingFactor { get; set; } = 1.0;
}
