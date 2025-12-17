namespace AiDotNet.Enums;

/// <summary>
/// Defines the search strategy used to explore AutoML candidate configurations.
/// </summary>
/// <remarks>
/// <para>
/// AutoML can use different strategies to decide which candidate model configurations to try next.
/// The best choice depends on budget, search-space shape (continuous vs categorical), and how expensive each trial is.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is how AutoML decides what to try next:
/// <list type="bullet">
/// <item><description><see cref="RandomSearch"/> tries random settings (simple and surprisingly strong).</description></item>
/// <item><description><see cref="BayesianOptimization"/> tries to learn which settings work best and focus on them.</description></item>
/// <item><description><see cref="Evolutionary"/> evolves good settings over time (useful for discrete/conditional knobs).</description></item>
/// <item><description><see cref="MultiFidelity"/> uses short runs first and only gives more budget to promising trials.</description></item>
/// </list>
/// </para>
/// </remarks>
public enum AutoMLSearchStrategy
{
    /// <summary>
    /// Random search baseline.
    /// </summary>
    RandomSearch,

    /// <summary>
    /// Bayesian optimization (typically Gaussian-process or TPE style).
    /// </summary>
    BayesianOptimization,

    /// <summary>
    /// Evolutionary / genetic search.
    /// </summary>
    Evolutionary,

    /// <summary>
    /// Multi-fidelity search (e.g., HyperBand/ASHA-style scheduling).
    /// </summary>
    MultiFidelity
}

