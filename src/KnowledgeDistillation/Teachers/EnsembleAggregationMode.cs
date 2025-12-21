namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Defines how ensemble predictions are aggregated.
/// </summary>
public enum EnsembleAggregationMode
{
    /// <summary>
    /// Weighted average of teacher logits (most common).
    /// </summary>
    WeightedAverage,

    /// <summary>
    /// Geometric mean of teacher logits (for multiplicative ensembles).
    /// </summary>
    GeometricMean,

    /// <summary>
    /// Element-wise maximum (for pessimistic ensembles).
    /// </summary>
    Maximum,

    /// <summary>
    /// Element-wise median (robust to outliers).
    /// </summary>
    Median
}
