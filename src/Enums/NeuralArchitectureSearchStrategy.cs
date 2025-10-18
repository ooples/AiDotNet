namespace AiDotNet.Enums;

/// <summary>
/// Specifies the strategy used for neural architecture search.
/// </summary>
public enum NeuralArchitectureSearchStrategy
{
    /// <summary>
    /// Uses evolutionary algorithms to search for optimal architectures.
    /// </summary>
    Evolutionary,

    /// <summary>
    /// Uses reinforcement learning to search for optimal architectures.
    /// </summary>
    ReinforcementLearning,

    /// <summary>
    /// Uses gradient-based optimization (DARTS) to search for optimal architectures.
    /// </summary>
    GradientBased,

    /// <summary>
    /// Uses random search as a baseline.
    /// </summary>
    RandomSearch,

    /// <summary>
    /// Uses Bayesian optimization to search for optimal architectures.
    /// </summary>
    BayesianOptimization
}
