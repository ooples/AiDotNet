namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Elastic Net Regression (combined L1 and L2 regularization).
/// </summary>
/// <remarks>
/// <para>
/// Elastic Net combines the penalties of Ridge (L2) and Lasso (L1) regression, providing
/// a balance between feature selection (from L1) and handling correlated features (from L2).
/// </para>
/// <para>
/// The objective function minimized is:
/// (1/2n) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) * ||w||^2 / 2
/// </para>
/// <para><b>For Beginners:</b> Elastic Net gives you the best of both Ridge and Lasso.
///
/// Lasso (L1) is great for feature selection but has a limitation: when features are
/// highly correlated, it tends to arbitrarily pick one and zero out the others.
///
/// Ridge (L2) handles correlated features well but doesn't do feature selection -
/// all features keep non-zero coefficients.
///
/// Elastic Net combines both:
/// - It can still set coefficients to zero (like Lasso) for feature selection
/// - It groups correlated features together (like Ridge) instead of picking one arbitrarily
///
/// When to use Elastic Net:
/// - When you have correlated features and want feature selection
/// - When Lasso's behavior on correlated features is problematic
/// - When you're not sure whether Ridge or Lasso is better
///
/// The l1_ratio parameter controls the mix:
/// - l1_ratio = 1.0: Pure Lasso (L1 only)
/// - l1_ratio = 0.0: Pure Ridge (L2 only)
/// - l1_ratio = 0.5: Equal mix of L1 and L2 (default)
///
/// Note: If your features are on different scales, consider normalizing your data
/// before training using INormalizer implementations like ZScoreNormalizer or MinMaxNormalizer.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class ElasticNetRegressionOptions<T> : RegressionOptions<T>
{
    private double _alpha = 1.0;
    private double _l1Ratio = 0.5;
    private int _maxIterations = 1000;
    private double _tolerance = 1e-4;

    /// <summary>
    /// Gets or sets the overall regularization strength. Must be a non-negative value.
    /// </summary>
    /// <value>The regularization parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the overall strength of regularization. It multiplies both
    /// the L1 and L2 penalties. Larger values result in stronger regularization.
    /// </para>
    /// <para><b>For Beginners:</b> Alpha controls the overall regularization strength.
    ///
    /// Think of it as a volume knob for regularization:
    /// - Alpha = 0.0: No regularization (ordinary least squares)
    /// - Alpha = 1.0: Moderate regularization (default)
    /// - Alpha = 10.0: Strong regularization
    ///
    /// Higher alpha means more shrinkage and potentially more features set to zero.
    /// Use cross-validation to find the optimal value for your data.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is negative.</exception>
    public double Alpha
    {
        get => _alpha;
        set
        {
            if (value < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Alpha must be non-negative.");
            }
            _alpha = value;
        }
    }

    /// <summary>
    /// Gets or sets the ratio of L1 penalty in the combined penalty. Must be between 0 and 1.
    /// </summary>
    /// <value>The L1 ratio, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the mix between L1 (Lasso) and L2 (Ridge) penalties:
    /// - 1.0 = Pure Lasso (L1 only)
    /// - 0.0 = Pure Ridge (L2 only)
    /// - 0.5 = Equal mix (default)
    /// </para>
    /// <para><b>For Beginners:</b> L1Ratio controls the balance between feature selection and stability.
    ///
    /// The effects of different values:
    /// - L1Ratio = 1.0: Pure Lasso - maximum sparsity, may have issues with correlated features
    /// - L1Ratio = 0.5: Balanced - good default for most problems
    /// - L1Ratio = 0.1: Mostly Ridge - keeps more features, better for correlated features
    /// - L1Ratio = 0.0: Pure Ridge - no feature selection, all features kept
    ///
    /// Tips:
    /// - Start with 0.5 and adjust based on results
    /// - If you need feature selection but have correlated features, try 0.3-0.7
    /// - Use cross-validation to find the optimal value
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is outside [0, 1] range.</exception>
    public double L1Ratio
    {
        get => _l1Ratio;
        set
        {
            if (value < 0 || value > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "L1Ratio must be between 0 and 1.");
            }
            _l1Ratio = value;
        }
    }

    /// <summary>
    /// Gets or sets the maximum number of iterations for the coordinate descent algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// Elastic Net uses coordinate descent optimization. This parameter limits
    /// the number of iterations to prevent infinite loops.
    /// </para>
    /// <para><b>For Beginners:</b> This sets how many times the algorithm can try to improve.
    /// The default of 1000 is usually sufficient. Increase if you see convergence warnings.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int MaxIterations
    {
        get => _maxIterations;
        set
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "MaxIterations must be positive.");
            }
            _maxIterations = value;
        }
    }

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 1e-4.</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the maximum change in coefficients falls below this threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how precise the solution needs to be.
    /// The default is good for most applications.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public double Tolerance
    {
        get => _tolerance;
        set
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "Tolerance must be positive.");
            }
            _tolerance = value;
        }
    }

    /// <summary>
    /// Gets or sets whether to use warm starting for cross-validation.
    /// </summary>
    /// <value>True to use warm starting; false otherwise. Default is true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the previous solution is used as the starting point for retraining,
    /// which can significantly speed up cross-validation.
    /// </para>
    /// <para><b>For Beginners:</b> Keep this enabled (default) for faster training when
    /// trying different parameter values.
    /// </para>
    /// </remarks>
    public bool WarmStart { get; set; } = true;
}
