using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Interface for causal structure learning algorithms that discover Directed Acyclic Graphs (DAGs) from data.
/// </summary>
/// <remarks>
/// <para>
/// Causal discovery algorithms analyze observational data to infer the causal structure — a DAG where
/// edges represent direct causal relationships between variables. Unlike correlation analysis, these
/// algorithms attempt to determine the direction of causation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This interface defines the contract for algorithms that figure out
/// cause-and-effect relationships from data. Given a dataset with multiple variables, the algorithm
/// produces a graph showing which variables directly cause changes in other variables.
///
/// For example, given data about weather, traffic, and commute time, a causal discovery algorithm
/// might find: Weather → Traffic → Commute Time (weather causes traffic, which causes longer commutes).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public interface ICausalDiscoveryAlgorithm<T>
{
    /// <summary>
    /// Gets the display name of this algorithm.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the methodological category of this algorithm.
    /// </summary>
    CausalDiscoveryCategory Category { get; }

    /// <summary>
    /// Gets whether this algorithm can handle latent (unobserved) confounders.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Latent confounders are hidden variables that affect two or more observed
    /// variables. For example, "genetic predisposition" might affect both "exercise frequency" and
    /// "heart health" but may not be directly measured in your data.
    /// </para>
    /// </remarks>
    bool SupportsLatentConfounders { get; }

    /// <summary>
    /// Gets whether this algorithm is designed for time series data.
    /// </summary>
    bool SupportsTimeSeries { get; }

    /// <summary>
    /// Gets whether this algorithm can discover nonlinear causal relationships.
    /// </summary>
    bool SupportsNonlinear { get; }

    /// <summary>
    /// Gets whether this algorithm supports mixed (continuous and discrete) data types.
    /// </summary>
    bool SupportsMixedData { get; }

    /// <summary>
    /// Discovers causal structure from an observational data matrix.
    /// </summary>
    /// <param name="data">Data matrix of shape [n_samples, n_variables].</param>
    /// <param name="featureNames">Optional variable/feature names. If null, default names are generated.</param>
    /// <returns>A <see cref="CausalGraph{T}"/> representing the discovered causal structure.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in your data as a matrix where each row is an observation and each
    /// column is a variable. The algorithm will return a graph showing the causal relationships
    /// it discovered. Optionally provide names for your variables so the graph is easier to interpret.
    /// </para>
    /// </remarks>
    CausalGraph<T> DiscoverStructure(Matrix<T> data, string[]? featureNames = null);

    /// <summary>
    /// Discovers causal structure with a designated target variable for directed analysis.
    /// </summary>
    /// <param name="data">Data matrix of shape [n_samples, n_variables].</param>
    /// <param name="target">Target variable vector for directed causal analysis.</param>
    /// <param name="featureNames">Optional variable/feature names.</param>
    /// <returns>A <see cref="CausalGraph{T}"/> with the target as a distinguished node.</returns>
    CausalGraph<T> DiscoverStructure(Matrix<T> data, Vector<T> target, string[]? featureNames = null);
}
