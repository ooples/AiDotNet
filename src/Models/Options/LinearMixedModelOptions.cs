namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for mixed-effects (hierarchical/multilevel) models.
/// </summary>
/// <remarks>
/// <para>
/// Mixed-effects models handle data with natural grouping or clustering by estimating
/// both population-level (fixed) effects and group-level (random) effects. They properly
/// account for correlation within groups and provide valid inference.
/// </para>
/// <para>
/// <b>For Beginners:</b> Mixed-effects models are for data that has "groups" or "clusters".
///
/// Examples of grouped data:
/// - Students nested in schools
/// - Patients nested in hospitals
/// - Repeated measurements on individuals
/// - Products sold in different stores
///
/// Why use mixed models instead of regular regression?
/// 1. Properly accounts for non-independence within groups
/// 2. Gives you valid standard errors and p-values
/// 3. Shares information across groups (partial pooling)
/// 4. Estimates how much variation is between vs. within groups
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LinearMixedModelOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the estimation method.
    /// </summary>
    /// <value>Default is REML.</value>
    /// <remarks>
    /// <para>
    /// REML (Restricted Maximum Likelihood) is preferred for variance component estimation
    /// as it corrects for bias in maximum likelihood. ML (Maximum Likelihood) is needed
    /// for comparing models with different fixed effects.
    /// </para>
    /// <para><b>For Beginners:</b> REML gives less biased variance estimates. Use ML only
    /// when comparing models with different fixed effects using likelihood ratio tests.
    /// </para>
    /// </remarks>
    public MixedEffectsEstimationMethod EstimationMethod { get; set; } = MixedEffectsEstimationMethod.REML;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization.
    /// </summary>
    /// <value>Default is 100.</value>
    private int _maxIterations = 100;
    public int MaxIterations
    {
        get => _maxIterations;
        set
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(MaxIterations), "MaxIterations must be > 0.");
            }
            _maxIterations = value;
        }
    }

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    private double _tolerance = 1e-6;
    public double Tolerance
    {
        get => _tolerance;
        set
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(Tolerance), "Tolerance must be > 0.");
            }
            _tolerance = value;
        }
    }

    /// <summary>
    /// Gets or sets whether to compute confidence intervals for variance components.
    /// </summary>
    /// <value>Default is true.</value>
    public bool ComputeVarianceCI { get; set; } = true;

    /// <summary>
    /// Gets or sets the confidence level for intervals.
    /// </summary>
    /// <value>Default is 0.95 (95% confidence).</value>
    private double _confidenceLevel = 0.95;
    public double ConfidenceLevel
    {
        get => _confidenceLevel;
        set
        {
            if (value <= 0 || value >= 1)
            {
                throw new ArgumentOutOfRangeException(nameof(ConfidenceLevel), "ConfidenceLevel must be between 0 and 1.");
            }
            _confidenceLevel = value;
        }
    }

    /// <summary>
    /// Gets or sets whether to use the bounded optimization for variance components.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// When true, variance components are constrained to be non-negative.
    /// </remarks>
    public bool BoundVarianceComponents { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to compute predicted random effects (BLUPs).
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// BLUPs (Best Linear Unbiased Predictors) are the estimated random effects for each group.
    /// </remarks>
    public bool ComputeBLUPs { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to compute marginal and conditional R-squared.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// Marginal R-squared reflects fixed effects only.
    /// Conditional R-squared includes both fixed and random effects.
    /// </remarks>
    public bool ComputeRSquared { get; set; } = true;

    /// <summary>
    /// Gets or sets the optimizer to use.
    /// </summary>
    /// <value>Default is EM algorithm.</value>
    public MixedEffectsOptimizer Optimizer { get; set; } = MixedEffectsOptimizer.EM;

    /// <summary>
    /// Gets or sets whether to use singular value decomposition for numerical stability.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseSVD { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to print verbose output during fitting.
    /// </summary>
    public bool Verbose { get; set; }
}

/// <summary>
/// Estimation methods for mixed-effects models.
/// </summary>
public enum MixedEffectsEstimationMethod
{
    /// <summary>
    /// Restricted Maximum Likelihood - preferred for variance estimation.
    /// </summary>
    REML,

    /// <summary>
    /// Maximum Likelihood - use for model comparison.
    /// </summary>
    ML
}

/// <summary>
/// Optimization algorithms for mixed-effects model fitting.
/// </summary>
public enum MixedEffectsOptimizer
{
    /// <summary>
    /// Expectation-Maximization algorithm.
    /// </summary>
    EM,

    /// <summary>
    /// Newton-Raphson optimization.
    /// </summary>
    NewtonRaphson,

    /// <summary>
    /// Fisher scoring algorithm.
    /// </summary>
    FisherScoring
}
