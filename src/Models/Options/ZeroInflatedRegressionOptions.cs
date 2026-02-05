namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Zero-Inflated regression models.
/// </summary>
/// <remarks>
/// <para>
/// Zero-Inflated models handle count data that has more zeros than a standard count
/// distribution (Poisson, Negative Binomial) would predict. They model the data as
/// a mixture of a point mass at zero and a count distribution.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sometimes you're counting things, but there are lots of zeros:
/// - Number of insurance claims (most people have zero claims)
/// - Number of fish caught (many fishers catch nothing)
/// - Number of cigarettes smoked (many people don't smoke)
///
/// Regular Poisson regression doesn't handle this well because it assumes a specific
/// relationship between the mean and variance, and can't account for "excess zeros."
///
/// Zero-Inflated models solve this by saying:
/// "There are two types of zeros - some come from a zero-generating process
/// (people who NEVER smoke), and some come from the count process
/// (smokers who happened to smoke 0 cigarettes today)."
///
/// This gives more accurate predictions and proper uncertainty estimates.
/// </para>
/// </remarks>
public class ZeroInflatedRegressionOptions
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
    /// Gets or sets the base count distribution family.
    /// </summary>
    /// <value>Default is Poisson.</value>
    public ZeroInflatedDistributionFamily DistributionFamily { get; set; } = ZeroInflatedDistributionFamily.Poisson;

    /// <summary>
    /// Gets or sets whether to model the zero-inflation probability as a function of features.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// When true, the probability of being a "structural zero" (always zero) can vary
    /// with the features. When false, a single probability is estimated.
    /// </remarks>
    public bool ModelZeroInflation { get; set; } = true;

    /// <summary>
    /// Gets or sets the link function for the count model.
    /// </summary>
    /// <value>Default is Log.</value>
    public ZeroInflatedCountLink CountLink { get; set; } = ZeroInflatedCountLink.Log;

    /// <summary>
    /// Gets or sets the link function for the zero-inflation model.
    /// </summary>
    /// <value>Default is Logit.</value>
    public ZeroInflatedZeroLink ZeroLink { get; set; } = ZeroInflatedZeroLink.Logit;

    /// <summary>
    /// Gets or sets whether to use regularization.
    /// </summary>
    /// <value>Default is true.</value>
    public bool UseRegularization { get; set; } = true;

    /// <summary>
    /// Gets or sets the regularization strength.
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double RegularizationStrength { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }
}

/// <summary>
/// Base count distributions for zero-inflated models.
/// </summary>
public enum ZeroInflatedDistributionFamily
{
    /// <summary>
    /// Poisson distribution. Good when variance roughly equals mean.
    /// </summary>
    Poisson,

    /// <summary>
    /// Negative Binomial distribution. Good for overdispersed count data
    /// (variance greater than mean).
    /// </summary>
    NegativeBinomial
}

/// <summary>
/// Link functions for the count component of zero-inflated models.
/// </summary>
public enum ZeroInflatedCountLink
{
    /// <summary>
    /// Log link: η = log(μ). Most common for count data.
    /// </summary>
    Log,

    /// <summary>
    /// Square root link: η = √μ.
    /// </summary>
    SquareRoot,

    /// <summary>
    /// Identity link: η = μ.
    /// </summary>
    Identity
}

/// <summary>
/// Link functions for the zero-inflation component.
/// </summary>
public enum ZeroInflatedZeroLink
{
    /// <summary>
    /// Logit link: η = log(π/(1-π)).
    /// </summary>
    Logit,

    /// <summary>
    /// Probit link: η = Φ⁻¹(π).
    /// </summary>
    Probit,

    /// <summary>
    /// Complementary log-log link: η = log(-log(1-π)).
    /// </summary>
    CLogLog
}
