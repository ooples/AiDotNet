namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Mixed-Effects (Hierarchical/Multilevel) Models.
/// </summary>
/// <remarks>
/// <para>
/// Mixed-effects models contain both fixed effects (population-level parameters) and random
/// effects (group-level variations). They're essential for analyzing hierarchical or clustered
/// data where observations are not independent.
/// </para>
/// <para>
/// <b>For Beginners:</b> Mixed-effects models are perfect when your data has a natural grouping structure:
///
/// <b>Examples:</b>
/// - Students within schools: Students in the same school share characteristics
/// - Patients within hospitals: Patients at the same hospital have similar care patterns
/// - Measurements over time: Repeated measurements from the same person are correlated
/// - Products within brands: Products from the same brand share brand-level qualities
///
/// <b>Why not just use regular regression?</b>
/// Regular regression assumes all observations are independent. But students at the same
/// school are more similar to each other than students at different schools. Ignoring this
/// leads to overconfident predictions and incorrect conclusions.
///
/// <b>Two types of effects:</b>
/// - <b>Fixed effects:</b> Population-level patterns (e.g., "on average, studying 1 more hour
///   increases test scores by 5 points")
/// - <b>Random effects:</b> Group-level variations (e.g., "some schools have higher baseline
///   scores than others")
///
/// The model estimates both the population patterns AND how much groups vary from this pattern.
/// </para>
/// </remarks>
public class MixedEffectsModelOptions
{
    /// <summary>
    /// Gets or sets the maximum number of iterations for optimization.
    /// </summary>
    /// <value>Default is 100.</value>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the optimization method.
    /// </summary>
    /// <value>Default is REML.</value>
    public MixedEffectsOptimization OptimizationMethod { get; set; } = MixedEffectsOptimization.REML;

    /// <summary>
    /// Gets or sets whether to include random intercepts.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// Random intercepts allow each group to have its own baseline level.
    /// </remarks>
    public bool IncludeRandomIntercept { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include random slopes.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// Random slopes allow the effect of predictors to vary across groups.
    /// For example, the effect of study time on test scores might be stronger in some schools.
    /// </remarks>
    public bool IncludeRandomSlopes { get; set; } = false;

    /// <summary>
    /// Gets or sets the indices of features to include as random slopes (if enabled).
    /// </summary>
    /// <value>Default is null (all features if random slopes enabled).</value>
    public int[]? RandomSlopeFeatures { get; set; }

    /// <summary>
    /// Gets or sets the covariance structure for random effects.
    /// </summary>
    /// <value>Default is Unstructured.</value>
    public MixedEffectsCovarianceStructure CovarianceStructure { get; set; } = MixedEffectsCovarianceStructure.Unstructured;

    /// <summary>
    /// Gets or sets whether to use robust standard errors.
    /// </summary>
    /// <value>Default is false.</value>
    public bool UseRobustStandardErrors { get; set; } = false;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether to center features before fitting.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// Centering improves numerical stability and makes intercepts more interpretable.
    /// </remarks>
    public bool CenterFeatures { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum number of observations per group.
    /// </summary>
    /// <value>Default is 2.</value>
    /// <remarks>
    /// Groups with fewer observations than this are handled specially or excluded.
    /// </remarks>
    public int MinObservationsPerGroup { get; set; } = 2;
}

/// <summary>
/// Optimization methods for mixed-effects models.
/// </summary>
public enum MixedEffectsOptimization
{
    /// <summary>
    /// Maximum Likelihood estimation. Provides unbiased fixed effects but biased variance estimates.
    /// </summary>
    ML,

    /// <summary>
    /// Restricted Maximum Likelihood. Provides unbiased variance estimates. Recommended for most cases.
    /// </summary>
    REML
}

/// <summary>
/// Covariance structures for random effects.
/// </summary>
public enum MixedEffectsCovarianceStructure
{
    /// <summary>
    /// Unstructured covariance. Most flexible, but requires the most parameters.
    /// </summary>
    Unstructured,

    /// <summary>
    /// Diagonal covariance (no correlation between random effects).
    /// </summary>
    Diagonal,

    /// <summary>
    /// Compound symmetry (equal correlations).
    /// </summary>
    CompoundSymmetry,

    /// <summary>
    /// Identity covariance (scaled identity matrix).
    /// </summary>
    Identity
}
