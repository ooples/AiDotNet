namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Inverse Gaussian Regression, a generalized linear model for positive continuous data
/// with variance proportional to the cube of the mean.
/// </summary>
/// <remarks>
/// <para>
/// The Inverse Gaussian distribution (also known as Wald distribution) is appropriate for modeling
/// positive continuous response variables, particularly those with heavy right tails. The variance
/// is proportional to μ³, making it suitable when larger values have much more variability than
/// smaller values.
/// </para>
/// <para><b>For Beginners:</b> Inverse Gaussian regression is designed for predicting positive continuous
/// values where the data has a heavy right tail (extreme large values are possible).
///
/// It's useful when:
/// - Your target values are always positive (never zero or negative)
/// - Larger values tend to be much more variable than smaller values
/// - The data has a heavier tail than Gamma distribution
///
/// Examples:
/// - Response times in cognitive experiments
/// - Time until failure for certain mechanical systems
/// - First passage times in physics
/// - Waiting times in queuing systems
///
/// Compared to Gamma regression, Inverse Gaussian assumes even more variability for large values.
/// </para>
/// </remarks>
public class InverseGaussianRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations for the IRLS algorithm.
    /// </summary>
    /// <value>The maximum iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// The Inverse Gaussian regression model is fitted using Iteratively Reweighted Least Squares (IRLS).
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
    /// Gets or sets the link function type for the Inverse Gaussian GLM.
    /// </summary>
    /// <value>The link function type, defaulting to Log.</value>
    /// <remarks>
    /// <para>
    /// The link function relates the linear predictor to the expected value of the response.
    /// - Log: ln(μ) = Xβ, most commonly used, ensures positive predictions
    /// - InverseSquared: -1/(2μ²) = Xβ, the canonical link for Inverse Gaussian
    /// - Inverse: 1/μ = Xβ, alternative link
    /// - Identity: μ = Xβ (requires constraints to ensure positivity)
    /// </para>
    /// <para><b>For Beginners:</b> The link function determines how predictions are computed:
    /// - Log link (default): Most common, ensures positive predictions naturally
    /// - InverseSquared link: The "canonical" link for Inverse Gaussian, sometimes better fits
    /// - Inverse link: Alternative that's also commonly used
    /// - Identity link: Use with caution, may predict negative values
    ///
    /// Stick with Log link unless you have a specific reason to change it.
    /// </para>
    /// </remarks>
    public InverseGaussianLinkFunction LinkFunction { get; set; } = InverseGaussianLinkFunction.Log;

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
    /// The dispersion parameter (φ = 1/λ) in the Inverse Gaussian distribution controls the variance.
    /// Variance = φ × μ³. It is estimated during fitting.
    /// </para>
    /// <para><b>For Beginners:</b> The dispersion parameter measures how spread out the
    /// data is relative to its mean cubed. It's automatically estimated during training,
    /// but you can provide an initial guess if you have domain knowledge.
    /// </para>
    /// </remarks>
    public double InitialDispersion { get; set; } = 1.0;
}

/// <summary>
/// Link functions for Inverse Gaussian regression.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The link function transforms the expected response to the scale
/// where linear prediction happens. Different links make different assumptions about
/// how predictors affect the response.
/// </para>
/// </remarks>
public enum InverseGaussianLinkFunction
{
    /// <summary>
    /// Log link: ln(μ) = Xβ. Most commonly used, ensures positive predictions.
    /// </summary>
    Log,

    /// <summary>
    /// Inverse squared link: -1/(2μ²) = Xβ. The canonical link for Inverse Gaussian distribution.
    /// </summary>
    InverseSquared,

    /// <summary>
    /// Inverse link: 1/μ = Xβ. Alternative link function.
    /// </summary>
    Inverse,

    /// <summary>
    /// Identity link: μ = Xβ. Linear predictions, may require constraints.
    /// </summary>
    Identity
}
