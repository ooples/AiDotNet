namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Ridge Regression (L2 regularized linear regression).
/// </summary>
/// <remarks>
/// <para>
/// Ridge Regression extends ordinary least squares regression by adding an L2 penalty term
/// to the loss function. This penalty shrinks the coefficients toward zero, helping to prevent
/// overfitting, especially when dealing with multicollinearity (highly correlated features) or
/// when the number of features is large relative to the number of samples.
/// </para>
/// <para>
/// The objective function minimized is: ||y - Xw||^2 + alpha * ||w||^2
/// where alpha controls the strength of the regularization.
/// </para>
/// <para><b>For Beginners:</b> Ridge Regression is like standard linear regression with a "shrinkage" effect.
///
/// Imagine you're fitting a line to data points, but some of your features are noisy or redundant.
/// Without regularization, the model might give these noisy features large coefficients, leading to
/// poor predictions on new data (overfitting).
///
/// Ridge Regression solves this by:
/// - Adding a penalty for large coefficient values
/// - Shrinking all coefficients toward zero (but never exactly to zero)
/// - Making the model more stable and generalizable
///
/// When to use Ridge Regression:
/// - When you have many features that might be correlated
/// - When you want to prevent overfitting without removing features
/// - When all features are expected to contribute to the prediction
///
/// The "alpha" parameter controls how much shrinkage is applied:
/// - Higher alpha = more shrinkage = simpler model (might underfit)
/// - Lower alpha = less shrinkage = more complex model (might overfit)
///
/// Note: If your features are on different scales, consider normalizing your data
/// before training using INormalizer implementations like ZScoreNormalizer or MinMaxNormalizer.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class RidgeRegressionOptions<T> : RegressionOptions<T>
{
    private double _alpha = 1.0;

    /// <summary>
    /// Gets or sets the regularization strength (alpha). Must be a non-negative value.
    /// </summary>
    /// <value>The regularization parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the strength of the L2 regularization penalty. Larger values
    /// specify stronger regularization, which results in smaller coefficient values and
    /// potentially underfitting. Smaller values allow the model to fit the training data
    /// more closely but may lead to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> Alpha controls how much the model prioritizes simplicity versus
    /// fitting the data closely.
    ///
    /// Think of it like a dial:
    /// - Alpha = 0.0: No regularization (equivalent to ordinary least squares)
    /// - Alpha = 1.0: Moderate regularization (default, good starting point)
    /// - Alpha = 10.0: Strong regularization (very simple model)
    /// - Alpha = 100.0: Very strong regularization (coefficients very close to zero)
    ///
    /// Tips for choosing alpha:
    /// - Start with the default (1.0) and evaluate model performance
    /// - If the model overfits (good on training, poor on test), increase alpha
    /// - If the model underfits (poor on both training and test), decrease alpha
    /// - Use cross-validation to find the optimal value systematically
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
    /// Gets or sets the type of matrix decomposition used to solve the ridge regression equations.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to Cholesky.</value>
    /// <remarks>
    /// <para>
    /// Ridge Regression has a closed-form solution that requires solving a linear system.
    /// Different decomposition methods offer trade-offs between speed and numerical stability.
    /// Cholesky is the fastest for positive definite matrices (which the regularized normal
    /// equation always produces). SVD is more robust but slower.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how the mathematical equations are solved internally.
    ///
    /// The default (Cholesky) is fastest and works well for most problems. You might consider:
    /// - Cholesky: Fast, stable, recommended for most cases
    /// - SVD: More robust for ill-conditioned problems (when features are highly correlated)
    /// - QR: Good balance between speed and numerical stability
    ///
    /// Unless you encounter numerical issues, the default is fine.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}
