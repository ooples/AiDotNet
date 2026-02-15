using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for causal structure discovery.
/// </summary>
/// <remarks>
/// <para>
/// These options control which causal discovery algorithm is used and how it behaves.
/// All properties are nullable with industry-standard defaults applied internally when null.
/// </para>
/// <para>
/// <b>For Beginners:</b> You can use these options to configure how the causal discovery
/// algorithm works. If you leave everything as null, sensible defaults will be used.
/// The most important option is <see cref="Algorithm"/> which determines which method is used.
///
/// Example:
/// <code>
/// builder.ConfigureCausalDiscovery(options => {
///     options.Algorithm = CausalDiscoveryAlgorithmType.NOTEARSLinear;
///     options.SparsityPenalty = 0.1;   // encourage sparse graphs
///     options.EdgeThreshold = 0.3;     // prune weak edges
/// });
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("CausalDiscovery")]
public class CausalDiscoveryOptions
{
    /// <summary>
    /// Which causal discovery algorithm to use. Default: null (auto-select based on data characteristics).
    /// </summary>
    /// <remarks>
    /// <para>When null, the system selects an appropriate algorithm based on data size and type.
    /// For small-medium datasets (&lt;1000 variables), NOTEARS Linear is used.
    /// For larger datasets, DAGMA Linear is preferred for speed.</para>
    /// </remarks>
    public CausalDiscoveryAlgorithmType? Algorithm { get; set; }

    /// <summary>
    /// Significance level for conditional independence tests. Default: null (0.05).
    /// </summary>
    /// <remarks>
    /// <para>Used by constraint-based methods (PC, FCI, MMPC). Lower values are more conservative
    /// (fewer edges). Typical range: 0.01 to 0.10.</para>
    /// </remarks>
    public double? SignificanceLevel { get; set; }

    /// <summary>
    /// L1 sparsity penalty (lambda1). Default: null (algorithm-specific default).
    /// </summary>
    /// <remarks>
    /// <para>Controls the sparsity of the learned graph. Higher values produce sparser (fewer edges) graphs.
    /// NOTEARS default: 0.1. DAGMA default: 0.03. Set to 0 for no sparsity penalty.</para>
    /// </remarks>
    public double? SparsityPenalty { get; set; }

    /// <summary>
    /// Edge weight threshold for pruning. Default: null (0.3).
    /// </summary>
    /// <remarks>
    /// <para>After optimization, edges with absolute weight below this threshold are removed.
    /// The NOTEARS paper uses 0.3 as default. Lower values keep more edges; higher values
    /// produce cleaner graphs.</para>
    /// </remarks>
    public double? EdgeThreshold { get; set; }

    /// <summary>
    /// Maximum conditioning set size for constraint-based methods. Default: null (3).
    /// </summary>
    /// <remarks>
    /// <para>Limits the size of conditioning sets tested in PC/FCI algorithms.
    /// Higher values are more thorough but exponentially slower.</para>
    /// </remarks>
    public int? MaxConditioningSetSize { get; set; }

    /// <summary>
    /// Maximum number of outer iterations for optimization-based methods. Default: null (100).
    /// </summary>
    public int? MaxIterations { get; set; }

    /// <summary>
    /// Convergence tolerance for the acyclicity constraint h(W). Default: null (1e-8).
    /// </summary>
    /// <remarks>
    /// <para>The algorithm stops when h(W) &lt; this value, meaning the graph is essentially acyclic.</para>
    /// </remarks>
    public double? AcyclicityTolerance { get; set; }

    /// <summary>
    /// Maximum penalty parameter (rho_max) for augmented Lagrangian methods. Default: null (1e+16).
    /// </summary>
    public double? MaxPenalty { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null (non-deterministic).
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Variable/feature names to label the graph nodes. Default: null (auto-generated X0, X1, ...).
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Maximum number of parents per node. Default: null (unlimited).
    /// </summary>
    /// <remarks>
    /// <para>Constraining the in-degree can improve both speed and interpretability.</para>
    /// </remarks>
    public int? MaxParents { get; set; }

    /// <summary>
    /// Whether to also use the discovered causal graph for feature selection in preprocessing.
    /// Default: null (false).
    /// </summary>
    /// <remarks>
    /// <para>When true, the discovered causal parents of the target variable are used to
    /// select features for the downstream prediction model (Option A + B combined).</para>
    /// </remarks>
    public bool? UseForFeatureSelection { get; set; }

    /// <summary>
    /// Loss type for continuous optimization methods. Default: null ("l2").
    /// </summary>
    /// <remarks>
    /// <para>Supported values: "l2" (least squares), "logistic", "poisson".
    /// Matches the NOTEARS reference implementation.</para>
    /// </remarks>
    public string? LossType { get; set; }
}
