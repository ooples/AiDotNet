namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for data pruning.
/// </summary>
/// <remarks>
/// Data pruning removes easy or redundant samples based on training signals
/// (e.g., loss, confidence, forgetting events), keeping only the most informative examples.
/// </remarks>
public sealed class DataPrunerOptions
{
    /// <summary>Fraction of data to prune (remove). Default is 0.3 (30%).</summary>
    public double PruneRatio { get; set; } = 0.3;
    /// <summary>Strategy for selecting samples to prune. Default is HighConfidence.</summary>
    public PruneStrategy Strategy { get; set; } = PruneStrategy.HighConfidence;
    /// <summary>Minimum number of epochs before pruning scores are reliable. Default is 5.</summary>
    public int MinEpochsForScoring { get; set; } = 5;

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (PruneRatio < 0 || PruneRatio > 1) throw new ArgumentOutOfRangeException(nameof(PruneRatio), "PruneRatio must be between 0 and 1.");
        if (MinEpochsForScoring < 0) throw new ArgumentOutOfRangeException(nameof(MinEpochsForScoring), "MinEpochsForScoring must be non-negative.");
    }
}

/// <summary>
/// Strategy for selecting samples to prune.
/// </summary>
public enum PruneStrategy
{
    /// <summary>Remove samples with consistently high confidence (easy examples).</summary>
    HighConfidence,
    /// <summary>Remove samples never forgotten during training (already well-learned).</summary>
    NeverForgotten,
    /// <summary>Remove samples with lowest EL2N (error L2 norm) scores.</summary>
    LowEL2N,
    /// <summary>Remove samples with lowest GraNd (gradient norm) scores.</summary>
    LowGraNd
}
