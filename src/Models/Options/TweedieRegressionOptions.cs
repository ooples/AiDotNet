namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Tweedie Regression, a flexible generalized linear model that encompasses
/// several distributions (Poisson, Gamma, Inverse Gaussian) as special cases.
/// </summary>
/// <remarks>
/// <para>
/// Tweedie regression is a powerful family of distributions where variance is proportional to a power
/// of the mean: Var(Y) = φ × μ^p. The power parameter p determines which distribution family is used:
/// - p = 0: Normal/Gaussian (variance independent of mean)
/// - p = 1: Poisson (variance = mean)
/// - 1 &lt; p &lt; 2: Compound Poisson-Gamma (excellent for data with exact zeros and positive continuous values)
/// - p = 2: Gamma (variance = mean²)
/// - p = 3: Inverse Gaussian (variance = mean³)
/// </para>
/// <para><b>For Beginners:</b> Tweedie regression is like having a "dial" that lets you choose
/// how the variability in your data relates to the average.
///
/// It's especially useful for:
/// - Insurance claims: Many zeros (no claim) plus positive amounts (actual claims)
/// - Rainfall data: Many dry days (zero) plus positive rainfall amounts
/// - Any situation where you have both exact zeros and positive continuous values
/// - When you're not sure whether Poisson or Gamma is the right choice
///
/// The key advantage is that Tweedie with p between 1 and 2 naturally handles data that has:
/// - Exact zeros (like Poisson)
/// - Positive continuous values (like Gamma)
/// This makes it ideal for insurance, healthcare costs, and many business applications.
/// </para>
/// </remarks>
public class TweedieRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the power parameter (p) that determines the variance-mean relationship.
    /// </summary>
    /// <value>The power parameter, defaulting to 1.5 (compound Poisson-Gamma).</value>
    /// <remarks>
    /// <para>
    /// The power parameter p determines how variance scales with the mean:
    /// - p = 0: Normal/Gaussian distribution
    /// - p = 1: Poisson distribution (for counts)
    /// - 1 &lt; p &lt; 2: Compound Poisson-Gamma (handles zeros and positive continuous)
    /// - p = 2: Gamma distribution (positive continuous)
    /// - p = 3: Inverse Gaussian distribution (heavy-tailed positive continuous)
    ///
    /// Note: p cannot be in the range (0, 1) as this does not correspond to a valid distribution.
    /// </para>
    /// <para><b>For Beginners:</b> The power parameter is like a tuning dial:
    /// - Set p = 1.5 (default) if your data has zeros mixed with positive values
    /// - Set p = 1 if your data is pure counts (0, 1, 2, 3, ...)
    /// - Set p = 2 if your data is always positive with no zeros
    /// - Set p between 1.1 and 1.9 to find the best fit for your specific data
    ///
    /// The default of 1.5 is a good starting point for insurance and similar data.
    /// </para>
    /// </remarks>
    public double PowerParameter { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the IRLS algorithm.
    /// </summary>
    /// <value>The maximum iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// The Tweedie regression model is fitted using Iteratively Reweighted Least Squares (IRLS).
    /// This controls how many iterations are allowed before stopping.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how many refinement steps the algorithm takes.
    /// The default of 100 works well for most datasets.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance for the IRLS algorithm.
    /// </summary>
    /// <value>The convergence threshold, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the change in coefficients between iterations is less than
    /// this value, indicating convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how precise the solution needs to be.
    /// The default of 0.000001 is suitable for most applications.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the link function type for the Tweedie GLM.
    /// </summary>
    /// <value>The link function type, defaulting to Log.</value>
    /// <remarks>
    /// <para>
    /// The link function relates the linear predictor to the expected value of the response.
    /// - Log: ln(μ) = Xβ, most commonly used, ensures positive predictions
    /// - Power: μ^(1-p) = Xβ, the canonical link for Tweedie
    /// - Identity: μ = Xβ (requires constraints to ensure positivity when p > 0)
    /// </para>
    /// <para><b>For Beginners:</b> The link function determines how predictions are computed:
    /// - Log link (default): Most common, always gives positive predictions
    /// - Power link: The "natural" link for Tweedie, sometimes gives better fits
    /// - Identity link: Use with caution, may predict negative values
    ///
    /// Stick with Log link unless you have a specific reason to change it.
    /// </para>
    /// </remarks>
    public TweedieLinkFunction LinkFunction { get; set; } = TweedieLinkFunction.Log;

    /// <summary>
    /// Gets or sets the matrix decomposition type for solving the linear system.
    /// </summary>
    /// <value>The decomposition type, defaulting to QR.</value>
    /// <remarks>
    /// <para>
    /// QR decomposition is numerically stable and works well for most cases.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the mathematical method for solving equations.
    /// The default QR method works well in most situations.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Qr;

    /// <summary>
    /// Gets or sets the initial dispersion parameter estimate.
    /// </summary>
    /// <value>The initial dispersion estimate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The dispersion parameter φ scales the variance: Var(Y) = φ × μ^p.
    /// It is estimated during fitting.
    /// </para>
    /// <para><b>For Beginners:</b> The dispersion parameter measures how spread out the
    /// data is relative to the mean raised to the power p. It's automatically estimated
    /// during training, but you can provide an initial guess if you have domain knowledge.
    /// </para>
    /// </remarks>
    public double InitialDispersion { get; set; } = 1.0;

    /// <summary>
    /// Validates the options, ensuring the power parameter is valid.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when the power parameter is in the invalid range (0, 1).</exception>
    /// <remarks>
    /// <para>
    /// The Tweedie distribution is only defined for p ≤ 0 or p ≥ 1. Values in (0, 1) do not
    /// correspond to valid probability distributions.
    /// </para>
    /// <para><b>For Beginners:</b> The power parameter cannot be between 0 and 1 (exclusive)
    /// because this doesn't make mathematical sense for the Tweedie distribution.
    /// Use 0 for normal data, 1 for count data, or values ≥ 1 for positive continuous data.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (PowerParameter > 0 && PowerParameter < 1)
        {
            throw new ArgumentException(
                $"Power parameter must be ≤ 0 or ≥ 1 for the Tweedie distribution. Got {PowerParameter}. " +
                "Use p=0 for Normal, p=1 for Poisson, 1<p<2 for compound Poisson-Gamma, p=2 for Gamma, p=3 for Inverse Gaussian.");
        }
    }
}

/// <summary>
/// Link functions for Tweedie regression.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The link function transforms the expected response to the scale
/// where linear prediction happens. Different links make different assumptions about
/// how predictors affect the response.
/// </para>
/// </remarks>
public enum TweedieLinkFunction
{
    /// <summary>
    /// Log link: ln(μ) = Xβ. Most commonly used, ensures positive predictions.
    /// </summary>
    Log,

    /// <summary>
    /// Power link: μ^(1-p) = Xβ. The canonical link for Tweedie distribution.
    /// For p=1, this reduces to Log link. For p=2, this is Inverse link.
    /// </summary>
    Power,

    /// <summary>
    /// Identity link: μ = Xβ. Linear predictions, may require constraints for p > 0.
    /// </summary>
    Identity
}
