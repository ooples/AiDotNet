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

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (MaxPerplexity <= 0) throw new ArgumentOutOfRangeException(nameof(MaxPerplexity), "MaxPerplexity must be positive.");
        if (MinPerplexity < 0) throw new ArgumentOutOfRangeException(nameof(MinPerplexity), "MinPerplexity must be non-negative.");
        if (NGramOrder <= 0) throw new ArgumentOutOfRangeException(nameof(NGramOrder), "NGramOrder must be positive.");
        if (SmoothingFactor <= 0) throw new ArgumentOutOfRangeException(nameof(SmoothingFactor), "SmoothingFactor must be positive.");
    }
}
