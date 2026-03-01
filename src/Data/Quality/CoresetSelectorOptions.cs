namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for coreset selection.
/// </summary>
/// <remarks>
/// Coreset selection finds a small representative subset of the data that approximates
/// the full dataset for training, reducing compute while preserving model quality.
/// </remarks>
public sealed class CoresetSelectorOptions
{
    /// <summary>Target size of the coreset as a fraction of the original dataset. Default is 0.1 (10%).</summary>
    public double SelectionRatio { get; set; } = 0.1;
    /// <summary>Strategy for coreset selection. Default is Greedy (facility location).</summary>
    public CoresetStrategy Strategy { get; set; } = CoresetStrategy.Greedy;
    /// <summary>Random seed for reproducibility. Default is null (random).</summary>
    public int? Seed { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (SelectionRatio <= 0 || SelectionRatio > 1) throw new ArgumentOutOfRangeException(nameof(SelectionRatio), "SelectionRatio must be in (0, 1].");
    }
}

/// <summary>
/// Strategy for selecting coreset samples.
/// </summary>
public enum CoresetStrategy
{
    /// <summary>Greedy facility location: iteratively pick the point that maximizes coverage.</summary>
    Greedy,
    /// <summary>k-Center: minimize the maximum distance from any point to its nearest selected point.</summary>
    KCenter,
    /// <summary>Random sampling baseline.</summary>
    Random
}
