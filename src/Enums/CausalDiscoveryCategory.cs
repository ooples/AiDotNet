namespace AiDotNet.Enums;

/// <summary>
/// Categories of causal discovery algorithms based on their methodology.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Causal discovery algorithms can be grouped by how they work.
/// Some test statistical independence, some optimize a score, some use continuous math,
/// and some combine multiple approaches. This enum helps you understand and filter
/// algorithms by their methodology.
/// </para>
/// </remarks>
public enum CausalDiscoveryCategory
{
    /// <summary>
    /// Continuous optimization methods that formulate DAG learning as a smooth optimization problem.
    /// Includes NOTEARS, DAGMA, and GOLEM.
    /// </summary>
    ContinuousOptimization,

    /// <summary>
    /// Score-based search methods that evaluate candidate DAGs using a scoring function (e.g., BIC, BDeu).
    /// Includes GES, Hill Climbing, and Tabu Search.
    /// </summary>
    ScoreBasedSearch,

    /// <summary>
    /// Constraint-based methods that use conditional independence tests to build the graph skeleton.
    /// Includes PC, FCI, and MMPC.
    /// </summary>
    ConstraintBased,

    /// <summary>
    /// Hybrid methods that combine constraint-based and score-based approaches.
    /// Includes MMHC and GFCI.
    /// </summary>
    Hybrid,

    /// <summary>
    /// Functional causal model methods that exploit properties of the noise distribution (e.g., non-Gaussianity).
    /// Includes LiNGAM and Additive Noise Models.
    /// </summary>
    Functional,

    /// <summary>
    /// Time series causal discovery methods that account for temporal ordering and lagged effects.
    /// Includes Granger Causality, PCMCI, and DYNOTEARS.
    /// </summary>
    TimeSeries,

    /// <summary>
    /// Deep learning methods that use neural networks for structure learning.
    /// Includes DAG-GNN, GraNDAG, and DECI.
    /// </summary>
    DeepLearning,

    /// <summary>
    /// Bayesian methods that maintain a posterior distribution over graph structures.
    /// Includes Order MCMC and DiBS.
    /// </summary>
    Bayesian,

    /// <summary>
    /// Information-theoretic methods that use entropy and mutual information measures.
    /// Includes Transfer Entropy and oCSE.
    /// </summary>
    InformationTheoretic,

    /// <summary>
    /// Specialized methods that use unique mathematical formulations (e.g., integer linear programming).
    /// </summary>
    Specialized
}
