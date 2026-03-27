namespace AiDotNet.Enums;

/// <summary>
/// The expected format of model outputs that a loss function operates on.
/// </summary>
public enum OutputType
{
    /// <summary>Probability values in [0, 1] (after Softmax/Sigmoid).</summary>
    Probabilities,
    /// <summary>Raw logits (unbounded real values, before Softmax).</summary>
    Logits,
    /// <summary>Continuous real values (regression outputs).</summary>
    Continuous,
    /// <summary>Binary values (0 or 1).</summary>
    Binary,
    /// <summary>Distance/similarity values.</summary>
    Distances
}
