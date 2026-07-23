namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Gaussian Process Regression, a flexible non-parametric approach to regression
/// that provides uncertainty estimates along with predictions.
/// </summary>
/// <remarks>
/// <para>
/// Gaussian Process Regression (GPR) is a powerful machine learning technique that models the target function
/// as a sample from a Gaussian process. Unlike many other regression methods, GPR not only provides point
/// predictions but also uncertainty estimates, making it valuable for applications where understanding
/// prediction confidence is important.
/// </para>
/// <para><b>For Beginners:</b> Think of Gaussian Process Regression as drawing a smooth curve through your data
/// points, but instead of just giving you one "best" curve, it gives you a range of possible curves with
/// information about which ones are more likely. This is like a weather forecast that says "70°F with a 90%
/// chance of being between 65-75°F" rather than just "70°F." This ability to express uncertainty makes GPR
/// especially useful when you need to know not just what your model predicts, but how confident it is in those
/// predictions. GPR works well with small to medium datasets and can capture complex patterns without requiring
/// you to specify the exact form of the relationship beforehand.</para>
/// </remarks>
public class GaussianProcessRegressionOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the assumed noise level in the observations.
    /// </summary>
    /// <value>The noise level, defaulting to 0.00001 (1e-5).</value>
    /// <remarks>
    /// <para>
    /// This parameter represents the expected amount of random noise or measurement error in the target values.
    /// It adds a small value to the diagonal of the covariance matrix to improve numerical stability and account
    /// for noise in the observations. Higher values assume more noise in the data and lead to smoother predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This setting tells the model how much random error or "noise" to expect in your
    /// data. With the default value of 0.00001, the model assumes your data is very precise with minimal random
    /// fluctuations. Think of it like the difference between measuring something with a ruler versus a high-precision
    /// laser - the laser measurements have less noise. If your data comes from real-world measurements that might
    /// contain errors or random variations, you should increase this value (perhaps to 0.01 or 0.1). Higher values
    /// make the model less likely to try fitting to every tiny fluctuation in your data, resulting in smoother
    /// predictions. If your model is overfitting (following the training data too closely), increasing this value
    /// often helps.</para>
    /// </remarks>
    public double NoiseLevel { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to automatically optimize the hyperparameters (length scale and signal variance)
    /// based on the training data.
    /// </summary>
    /// <value>Whether to optimize hyperparameters, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When set to true, the model will use maximum likelihood estimation to find optimal values for the
    /// length scale and signal variance hyperparameters. This can improve model performance but increases
    /// training time. When false, the model uses the manually specified values.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the model should automatically tune its
    /// internal parameters to best fit your data. With the default value of false, you need to manually set
    /// the LengthScale and SignalVariance parameters. Setting this to true is like enabling "auto-tune" - the
    /// model will try different parameter values and pick the ones that work best for your specific data. This
    /// automatic optimization usually improves results but makes training slower. For beginners, it's often
    /// best to set this to true unless you have specific knowledge about appropriate parameter values for your
    /// problem or need faster training times.</para>
    /// </remarks>
    public bool OptimizeHyperparameters { get; set; } = false;

    /// <summary>
    /// Gets or sets the length scale parameter for the Gaussian Process kernel.
    /// </summary>
    /// <value>The length scale, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The length scale controls how rapidly the correlation between points decreases with distance.
    /// Smaller values make the model more sensitive to small-scale variations, creating more complex and
    /// wiggly functions. Larger values make the model focus on broader trends, creating smoother functions.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how "wiggly" or "smooth" your model's predictions will be.
    /// With the default value of 1.0, the model uses a standard level of smoothness. Think of it like adjusting
    /// the flexibility of a rubber band - a smaller length scale (like 0.1) creates a very flexible model that
    /// can make sharp turns and follow the data closely, while a larger length scale (like 10.0) creates a stiffer
    /// model that makes gentler, more gradual changes. If your data has rapid changes or fine structure, use a
    /// smaller length scale. If your data follows broader, smoother trends, use a larger length scale. If
    /// OptimizeHyperparameters is true, this value is just used as a starting point for optimization.</para>
    /// </remarks>
    public double LengthScale { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the signal variance parameter for the Gaussian Process kernel.
    /// </summary>
    /// <value>The signal variance, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The signal variance controls the overall magnitude of variations in the function. Larger values allow
    /// for larger deviations from the mean function, while smaller values constrain the function to stay
    /// closer to the mean. This parameter affects the overall scale of the uncertainty estimates.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much your model's predictions can vary overall.
    /// With the default value of 1.0, the model allows for a standard level of variation. Think of it like
    /// adjusting the volume on a speaker - a higher signal variance (like 2.0) allows the model to make more
    /// dramatic changes in its predictions, while a lower value (like 0.5) keeps predictions more conservative
    /// and closer to the average. If your target values have a wide range or large fluctuations, you might
    /// increase this value. If your target values stay within a narrow range, you might decrease it. If
    /// OptimizeHyperparameters is true, this value is just used as a starting point for optimization.</para>
    /// </remarks>
    public double SignalVariance { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the type of matrix decomposition to use for numerical stability.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to Cholesky.</value>
    /// <remarks>
    /// <para>
    /// Gaussian Process Regression involves inverting a covariance matrix, which can be numerically challenging.
    /// Different decomposition methods offer trade-offs between speed, stability, and accuracy. Cholesky
    /// decomposition is generally fast and stable for well-conditioned matrices, while other methods may be
    /// more robust for ill-conditioned matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical technique used to solve the internal
    /// equations in the model. With the default value of Cholesky, the model uses a standard method that works
    /// well in most cases. Think of it like choosing which route to take to a destination - different methods
    /// have different trade-offs in terms of speed and reliability. For most users, the default Cholesky
    /// decomposition is the best choice. You might consider changing this only if you encounter numerical
    /// errors or if you're working with a very large dataset where computational efficiency becomes critical.
    /// Unless you have a background in numerical linear algebra, it's best to leave this at the default setting.</para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}
