namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Gamma Regression, a generalized linear model for positive continuous data.
/// </summary>
/// <remarks>
/// <para>
/// Gamma Regression is suited for modeling positive continuous response variables, especially those that
/// are right-skewed and where variance increases with the mean. It uses either a log link function or
/// inverse link function. Common applications include insurance claims, hospital lengths of stay,
/// income modeling, and any situation where the response must be strictly positive.
/// </para>
/// <para><b>For Beginners:</b> Gamma Regression is designed for predicting positive continuous values
/// where the data tends to be skewed with a long right tail.
///
/// It's useful when:
/// - Your target values are always positive (never zero or negative)
/// - Larger values tend to be more variable than smaller values
/// - The data is right-skewed (most values are small, but some are very large)
///
/// Examples:
/// - Insurance claim amounts (can't be negative, large claims are more variable)
/// - Time until an event occurs (always positive)
/// - Income (always positive, highly variable for higher earners)
/// - Costs and prices
///
/// Unlike linear regression which can predict negative values, Gamma regression
/// naturally ensures predictions are always positive.
/// </para>
/// </remarks>
public class GammaRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations for the IRLS algorithm.
    /// </summary>
    /// <value>The maximum iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// The Gamma regression model is fitted using Iteratively Reweighted Least Squares (IRLS).
    /// This controls how many iterations are allowed before stopping.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how many refinement steps the algorithm takes.
    /// The default of 100 works well for most datasets. Increase it for complex data or if
    /// you get convergence warnings.
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
    /// The default of 0.000001 is suitable for most applications. Use a smaller value
    /// for more precision, or a larger value for faster training.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the link function type for the Gamma GLM.
    /// </summary>
    /// <value>The link function type, defaulting to Log.</value>
    /// <remarks>
    /// <para>
    /// The link function relates the linear predictor to the expected value of the response.
    /// - Log: ln(μ) = Xβ, predictions are exp(Xβ), always positive
    /// - Inverse: 1/μ = Xβ, the canonical link for Gamma
    /// - Identity: μ = Xβ (requires constraints to ensure positivity)
    /// </para>
    /// <para><b>For Beginners:</b> The link function determines how predictions are computed:
    /// - Log link (default): Most common, ensures positive predictions naturally
    /// - Inverse link: The "canonical" link for Gamma, sometimes gives better fits
    /// - Identity link: Use with caution, may predict negative values
    ///
    /// Stick with Log link unless you have a specific reason to change it.
    /// </para>
    /// </remarks>
    public GammaLinkFunction LinkFunction { get; set; } = GammaLinkFunction.Log;

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
    /// The dispersion parameter (φ) in the Gamma distribution controls the variance.
    /// Variance = φ × μ². It is estimated during fitting.
    /// </para>
    /// <para><b>For Beginners:</b> The dispersion parameter measures how spread out the
    /// data is relative to its mean. It's automatically estimated during training,
    /// but you can provide an initial guess if you have domain knowledge.
    /// </para>
    /// </remarks>
    public double InitialDispersion { get; set; } = 1.0;
}

/// <summary>
/// Link functions for Gamma regression.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The link function transforms the expected response to the scale
/// where linear prediction happens. Different links make different assumptions about
/// how predictors affect the response.
/// </para>
/// </remarks>
public enum GammaLinkFunction
{
    /// <summary>
    /// Log link: ln(μ) = Xβ. Most commonly used, ensures positive predictions.
    /// </summary>
    Log,

    /// <summary>
    /// Inverse link: 1/μ = Xβ. The canonical link for Gamma distribution.
    /// </summary>
    Inverse,

    /// <summary>
    /// Identity link: μ = Xβ. Linear predictions, may require constraints.
    /// </summary>
    Identity
}
