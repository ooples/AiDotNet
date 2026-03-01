namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for active learning query strategies.
/// </summary>
/// <remarks>
/// Active learning selects the most informative unlabeled samples for annotation,
/// maximizing model improvement per labeling dollar spent.
/// </remarks>
public sealed class ActiveLearningQueryStrategyOptions
{
    /// <summary>Number of samples to select per query round. Default is 100.</summary>
    public int QueryBatchSize { get; set; } = 100;
    /// <summary>Query strategy for sample selection. Default is Uncertainty.</summary>
    public QueryStrategy Strategy { get; set; } = QueryStrategy.Uncertainty;
    /// <summary>Number of Monte Carlo dropout forward passes for BALD. Default is 10.</summary>
    public int NumMcDropoutPasses { get; set; } = 10;
    /// <summary>Random seed for reproducibility. Default is null (random).</summary>
    public int? Seed { get; set; }

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (QueryBatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(QueryBatchSize), "QueryBatchSize must be positive.");
        if (NumMcDropoutPasses <= 0) throw new ArgumentOutOfRangeException(nameof(NumMcDropoutPasses), "NumMcDropoutPasses must be positive.");
    }
}

/// <summary>
/// Strategy for selecting samples in active learning.
/// </summary>
public enum QueryStrategy
{
    /// <summary>Select samples the model is most uncertain about (highest entropy).</summary>
    Uncertainty,
    /// <summary>Select samples with the smallest margin between top-2 predictions.</summary>
    Margin,
    /// <summary>Select samples where the model's top prediction has lowest confidence.</summary>
    LeastConfidence,
    /// <summary>Bayesian Active Learning by Disagreement (MC Dropout-based).</summary>
    BALD,
    /// <summary>Random sampling baseline.</summary>
    Random
}
