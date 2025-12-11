namespace AiDotNet.Configuration;

/// <summary>
/// Type of exploration decay schedule.
/// </summary>
public enum ExplorationDecayType
{
    /// <summary>
    /// Linear decay from initial to final value.
    /// </summary>
    Linear,

    /// <summary>
    /// Exponential decay (faster initial decay, slower later).
    /// </summary>
    Exponential,

    /// <summary>
    /// Cosine annealing schedule.
    /// </summary>
    Cosine
}
