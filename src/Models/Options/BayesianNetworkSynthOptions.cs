namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Bayesian Network Synthesis, a statistical approach that
/// learns a directed acyclic graph (DAG) structure and conditional probability tables
/// to generate synthetic tabular data via ancestral sampling.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Bayesian Network Synthesis operates in three phases:
/// - <b>Structure learning</b>: Discovers a DAG using greedy hill-climbing with BIC scoring
/// - <b>Parameter estimation</b>: Estimates conditional probability tables (CPTs) from the data
/// - <b>Ancestral sampling</b>: Generates data by sampling from root nodes down through the DAG
/// </para>
/// <para>
/// <b>For Beginners:</b> This method creates a probabilistic model of your data:
///
/// Think of a family tree of features — some features "depend on" others.
/// For example, in a health dataset:
/// 1. Age has no parents (sampled first)
/// 2. Blood pressure depends on Age
/// 3. Medication depends on Blood pressure
///
/// The model learns these dependency chains and samples new data following
/// the same parent-to-child order, producing statistically coherent rows.
///
/// Unlike neural network generators (CTGAN, TVAE), this uses classical statistics,
/// making it faster to train and more interpretable, though less flexible for
/// complex distributions.
///
/// Example:
/// <code>
/// var options = new BayesianNetworkSynthOptions&lt;double&gt;
/// {
///     MaxParents = 3,
///     NumBins = 20
/// };
/// var bnSynth = new BayesianNetworkSynthGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class BayesianNetworkSynthOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of parents per node in the DAG.
    /// </summary>
    /// <value>Maximum parents, defaulting to 3. Higher values allow more complex dependencies but increase computation.</value>
    public int MaxParents { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of discretization bins for continuous features.
    /// </summary>
    /// <value>Number of bins, defaulting to 20.</value>
    public int NumBins { get; set; } = 20;

    /// <summary>
    /// Gets or sets the maximum number of structure learning iterations.
    /// </summary>
    /// <value>Maximum iterations, defaulting to 100.</value>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the Laplace smoothing constant for CPT estimation.
    /// </summary>
    /// <value>Smoothing constant, defaulting to 1.0. Prevents zero-probability entries in CPTs.</value>
    public double LaplaceSmoothing { get; set; } = 1.0;
}
