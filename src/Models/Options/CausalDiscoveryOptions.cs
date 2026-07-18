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
public class CausalDiscoveryOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values (all options null → each algorithm's own default applies).</summary>
    public CausalDiscoveryOptions() { }

    /// <summary>Initializes a new instance by copying every option from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public CausalDiscoveryOptions(CausalDiscoveryOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Algorithm = other.Algorithm;
        SignificanceLevel = other.SignificanceLevel;
        ConfoundingEvidenceCutoff = other.ConfoundingEvidenceCutoff;
        SparsityPenalty = other.SparsityPenalty;
        EdgeThreshold = other.EdgeThreshold;
        MaxConditioningSetSize = other.MaxConditioningSetSize;
        MaxIterations = other.MaxIterations;
        AcyclicityTolerance = other.AcyclicityTolerance;
        MaxPenalty = other.MaxPenalty;
        FeatureNames = other.FeatureNames is null ? null : (string[])other.FeatureNames.Clone();
        MaxParents = other.MaxParents;
        UseForFeatureSelection = other.UseForFeatureSelection;
        LossType = other.LossType;
        LearningRate = other.LearningRate;
        MaxLag = other.MaxLag;
        HiddenUnits = other.HiddenUnits;
        InnerIterations = other.InnerIterations;
        MaxRank = other.MaxRank;
        CorrelationThreshold = other.CorrelationThreshold;
        DirectionalityAsymmetryThreshold = other.DirectionalityAsymmetryThreshold;
        ConcavityParameter = other.ConcavityParameter;
        SobolevWeight = other.SobolevWeight;
        MaxSegments = other.MaxSegments;
        InitScale = other.InitScale;
        MaxEpochs = other.MaxEpochs;
        InitialLogVariance = other.InitialLogVariance;
        DefaultKlWeight = other.DefaultKlWeight;
        MaxKlWeight = other.MaxKlWeight;
        UseKlWarmUp = other.UseKlWarmUp;
    }

    /// <summary>
    /// Which causal discovery algorithm to use. Default: null (auto-select based on data characteristics).
    /// </summary>
    /// <remarks>
    /// <para>When null, the system selects an appropriate algorithm based on data size and type.
    /// For small datasets (≤50 variables), NOTEARS Linear is used.
    /// For medium datasets (≤200 variables), DAGMA Linear is preferred for speed.
    /// For large datasets (&gt;200 variables), FGES is used for scalability.</para>
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
    /// Fraction (0..1) of a candidate's direction-evidence that may point the "wrong" way before RCD
    /// treats the remaining variables as latently confounded and stops. Default: null (0.05).
    /// </summary>
    /// <value>
    /// A finite value in <c>[0, 1]</c>, or <see langword="null"/> to use the calibrated default of
    /// <c>0.05</c>.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This knob controls how eagerly RCD declares "these variables share
    /// a hidden common cause, stop here." Lower values stop sooner (more cautious about hidden
    /// confounders); higher values demand stronger wrong-way evidence before stopping.</para>
    /// <para><b>Reference:</b> Maeda &amp; Shimizu, "RCD: Repetitive Causal Discovery of Linear
    /// Non-Gaussian Acyclic Models with Latent Confounders" (AISTATS 2020). The <c>0.05</c> default is
    /// a conservative operating point on the scale-free confounding-ratio described below (a clean
    /// root scores ≈ 0; a symmetric latently-confounded pair ≈ 0.5), leaving wide margin above the
    /// clean-root floor while still flagging genuine common-cause structure.</para>
    /// <para>Used by <c>RCDAlgorithm</c>. RCD scores each candidate root by the DirectLiNGAM entropy
    /// criterion (<c>DiffMutualInfo</c>): positive evidence means the candidate is a cause, negative
    /// means it is an effect. The confounding score is the SCALE-FREE ratio
    /// <c>Σ min(0, DiffMI)² / Σ DiffMI²</c> — the fraction of the best candidate's squared
    /// direction-evidence that indicates it is actually an effect of some other variable. A clean root
    /// scores ≈ 0 (all evidence points outward); a latently-confounded set has no clean root, so even
    /// the best candidate carries substantial wrong-way evidence (≈ 0.5 for a symmetric common-cause
    /// pair). When the score exceeds this cutoff the remaining variables are flagged confounded and
    /// left unordered (per RCD). Because it is a ratio in [0, 1] it needs no per-dataset rescaling —
    /// unlike a raw <c>DiffMI²</c> sum, whose magnitude drifts with sample size and non-Gaussianity.
    /// Lower values stop more eagerly (more conservative about confounding); typical range 0.02–0.20.</para>
    /// </remarks>
    public double? ConfoundingEvidenceCutoff { get; set; }

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

    // Seed is inherited from ModelOptions (random seed for reproducibility; null = non-deterministic).

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

    /// <summary>
    /// Learning rate for deep learning-based causal discovery methods. Default: null (1e-3).
    /// </summary>
    /// <remarks>
    /// <para>Controls the step size in gradient-based optimization for neural network methods
    /// (DAG-GNN, GraN-DAG, DECI, CGNN, CausalVAE, etc.).</para>
    /// </remarks>
    public double? LearningRate { get; set; }

    /// <summary>
    /// Maximum lag order for time-series causal discovery methods. Default: null (3).
    /// </summary>
    /// <remarks>
    /// <para>Controls how many past time steps are considered for lagged effects in methods
    /// like VAR-LiNGAM, DYNOTEARS, PCMCI, and Granger causality.</para>
    /// </remarks>
    public int? MaxLag { get; set; }

    /// <summary>
    /// Number of hidden units in neural network layers for deep learning methods. Default: null (64).
    /// </summary>
    /// <remarks>
    /// <para>Controls the capacity of neural networks used in deep learning-based causal discovery.
    /// Larger values can model more complex relationships but require more data.</para>
    /// </remarks>
    public int? HiddenUnits { get; set; }

    /// <summary>
    /// Number of inner gradient descent steps per outer augmented Lagrangian iteration. Default: null (algorithm-specific).
    /// </summary>
    /// <remarks>
    /// <para>Used by MCSL (default: 30), NOTEARS Low-Rank (default: 20), and similar algorithms.
    /// Higher values improve convergence per outer step but increase computation time.</para>
    /// </remarks>
    public int? InnerIterations { get; set; }

    /// <summary>
    /// Maximum rank for low-rank matrix factorizations. Default: null (10).
    /// </summary>
    /// <remarks>
    /// <para>Used by NOTEARS Low-Rank (W = A*B^T factorization).
    /// Higher rank allows more complex graphs but increases parameters quadratically.</para>
    /// </remarks>
    public int? MaxRank { get; set; }

    /// <summary>
    /// Correlation threshold for edge inclusion in constraint-based time-series methods. Default: null (0.1).
    /// </summary>
    /// <remarks>
    /// <para>Used by LPCMCI, TSFCI, and other time-series methods to filter weak correlations
    /// before running conditional independence tests.</para>
    /// </remarks>
    public double? CorrelationThreshold { get; set; }

    /// <summary>
    /// Minimum relative asymmetry (in [0, 1]) between the two cross-map skills of a pair before CCM
    /// prunes the weaker reverse direction as a reconstruction artifact. Default: null (0.2).
    /// </summary>
    /// <value>
    /// A finite value in <c>[0, 1]</c>, or <see langword="null"/> to use CCM's default threshold of <c>0.2</c>.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lower values remove weak reverse links more aggressively; higher values
    /// retain more bidirectional links.</para>
    /// <para><b>Reference:</b> Sugihara et al., "Detecting Causality in Complex Ecosystems," Science, 2012.</para>
    /// <para>Used by <see cref="AiDotNet.CausalDiscovery.TimeSeries.CCMAlgorithm{T}"/>. For a pair
    /// (X, Y) with forward/backward cross-map skills f and b, the relative asymmetry is
    /// |f - b| / (max(f, b) + eps). When it meets or exceeds this threshold the dominant direction is
    /// kept and the weaker one is zeroed; below it, both directions are retained as genuine
    /// bidirectional coupling. Raising the value keeps only more strongly asymmetric edges (fewer
    /// pruned reverse links); lowering it prunes more aggressively. Must be within [0, 1].</para>
    /// </remarks>
    public double? DirectionalityAsymmetryThreshold { get; set; }

    /// <summary>
    /// Concavity parameter (gamma) for MCP/SCAD penalty functions. Default: null (algorithm-specific).
    /// </summary>
    /// <remarks>
    /// <para>Used by CCDr and other methods using non-convex penalties.
    /// Controls the transition from L1 to constant penalty. Typical values: 1.5-3.7.</para>
    /// </remarks>
    public double? ConcavityParameter { get; set; }

    /// <summary>
    /// Sobolev regularization weight for NOTEARS Sobolev. Default: null (0.1).
    /// </summary>
    /// <remarks>
    /// <para>Controls the strength of the Sobolev smoothness penalty, which encourages
    /// smoothly varying causal functions. Distinct from <see cref="SparsityPenalty"/> (L1).</para>
    /// </remarks>
    public double? SobolevWeight { get; set; }

    /// <summary>
    /// Maximum number of segments for nonstationary time-series methods. Default: null (3).
    /// </summary>
    /// <remarks>
    /// <para>Used by NTS-NOTEARS for change-point-based partitioning.
    /// Higher values allow more regime changes to be detected.</para>
    /// </remarks>
    public int? MaxSegments { get; set; }

    /// <summary>
    /// Initialization scale for weight matrices or low-rank factors. Default: null (algorithm-specific).
    /// </summary>
    /// <remarks>
    /// <para>Controls the magnitude of initial weight values in optimization-based methods.
    /// For NOTEARS Low-Rank, this sets the scale of A,B factors (A*B^T ≈ scale² * rank).
    /// Too small values slow convergence; too large values cause acyclicity constraint issues.</para>
    /// </remarks>
    public double? InitScale { get; set; }

    /// <summary>
    /// Maximum number of training epochs for deep learning-based methods. Default: null (algorithm-specific).
    /// </summary>
    /// <remarks>
    /// <para>Used by CGNN, GraN-DAG, TCDF, and other neural network methods. Higher values allow
    /// more training but risk overfitting on small datasets.</para>
    /// </remarks>
    public int? MaxEpochs { get; set; }

    /// <summary>
    /// Initial log-variance for variational parameters in GAE and similar methods. Default: null (-4.0).
    /// </summary>
    /// <remarks>
    /// <para>Controls how tightly the initial posterior is centered around the mean.
    /// More negative values create a tighter initial posterior (lower variance).</para>
    /// </remarks>
    public double? InitialLogVariance { get; set; }

    /// <summary>
    /// Default KL divergence weight for variational methods (GAE, CausalVAE). Default: null (0.01).
    /// </summary>
    /// <remarks>
    /// <para>Scales the KL divergence term in the ELBO loss. Higher values enforce stronger
    /// regularization toward the prior but can hurt reconstruction quality.</para>
    /// </remarks>
    public double? DefaultKlWeight { get; set; }

    /// <summary>
    /// Maximum KL divergence weight after warm-up. Default: null (0.25).
    /// </summary>
    /// <remarks>
    /// <para>When KL warm-up is enabled, the KL weight ramps linearly from <see cref="DefaultKlWeight"/>
    /// to this value over the first 25% of training epochs to prevent posterior collapse.</para>
    /// </remarks>
    public double? MaxKlWeight { get; set; }

    /// <summary>
    /// Whether to use KL weight warm-up schedule. Default: null (true).
    /// </summary>
    /// <remarks>
    /// <para>When true, the KL weight ramps linearly from <see cref="DefaultKlWeight"/> to
    /// <see cref="MaxKlWeight"/> over the first 25% of epochs. When false, uses a fixed weight.</para>
    /// </remarks>
    public bool? UseKlWarmUp { get; set; }
}
