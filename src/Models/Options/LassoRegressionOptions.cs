namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Lasso Regression (L1 regularized linear regression).
/// </summary>
/// <remarks>
/// <para>
/// Lasso (Least Absolute Shrinkage and Selection Operator) Regression extends ordinary least squares
/// by adding an L1 penalty term to the loss function. Unlike Ridge Regression (L2), Lasso can shrink
/// coefficients exactly to zero, effectively performing automatic feature selection.
/// </para>
/// <para>
/// The objective function minimized is: (1/2n) * ||y - Xw||^2 + alpha * ||w||_1
/// where alpha controls the strength of the regularization and ||w||_1 is the L1 norm (sum of absolute values).
/// </para>
/// <para><b>For Beginners:</b> Lasso Regression is like Ridge Regression but with a key difference:
/// it can completely eliminate unimportant features.
///
/// While Ridge Regression shrinks coefficients toward zero but never quite reaches it,
/// Lasso can set coefficients exactly to zero. This makes Lasso useful for:
/// - Feature selection: Identifying which features actually matter
/// - Sparse models: Creating simpler models with fewer non-zero coefficients
/// - High-dimensional data: When you have many features but suspect only a few are relevant
///
/// Example scenario:
/// - You have 100 features for predicting house prices
/// - Only 10 of them actually matter (location, size, etc.)
/// - Lasso will automatically set the other 90 coefficients to zero
/// - This gives you a simpler, more interpretable model
///
/// The trade-off:
/// - Lasso requires iterative optimization (slower than Ridge)
/// - If features are highly correlated, Lasso might arbitrarily pick one and zero out others
/// - For groups of correlated features, consider ElasticNet instead
///
/// Note: If your features are on different scales, consider normalizing your data
/// before training using INormalizer implementations like ZScoreNormalizer or MinMaxNormalizer.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class LassoRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization strength (alpha). Must be a positive value.
    /// </summary>
    /// <value>The regularization parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the strength of the L1 regularization penalty. Larger values
    /// result in more coefficients being set to zero (sparser models). Smaller values allow
    /// the model to fit the training data more closely with more non-zero coefficients.
    /// </para>
    /// <para><b>For Beginners:</b> Alpha controls how aggressively Lasso eliminates features.
    ///
    /// The effect of alpha:
    /// - Alpha = 0.0: No regularization (ordinary least squares, all features kept)
    /// - Alpha = 0.1: Light regularization (most features kept)
    /// - Alpha = 1.0: Moderate regularization (default, good starting point)
    /// - Alpha = 10.0: Strong regularization (many features eliminated)
    ///
    /// Tips for choosing alpha:
    /// - Start with the default (1.0) and examine which features have non-zero coefficients
    /// - If too many features are eliminated, decrease alpha
    /// - If the model still overfits, increase alpha
    /// - Use cross-validation with a range of alpha values to find the optimal value
    /// </para>
    /// </remarks>
    public double Alpha { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the coordinate descent algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// Lasso requires iterative optimization using coordinate descent. This parameter limits
    /// the number of iterations to prevent infinite loops. If convergence is not achieved
    /// within this limit, the algorithm stops with the current solution.
    /// </para>
    /// <para><b>For Beginners:</b> This sets how many times the algorithm can try to improve the solution.
    ///
    /// The coordinate descent algorithm works by:
    /// 1. Looking at each coefficient one at a time
    /// 2. Adjusting it to reduce the error
    /// 3. Repeating until no more improvement can be made
    ///
    /// Usually, convergence happens well before 1000 iterations. You might increase this if:
    /// - You have many features and see warnings about not converging
    /// - Your tolerance is very strict
    ///
    /// You might decrease this if:
    /// - Training is too slow and you're okay with an approximate solution
    /// - You want to limit computational time
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 1e-4.</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the maximum change in coefficients between iterations
    /// falls below this threshold. Smaller values result in more precise solutions
    /// but require more iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how precise the solution needs to be.
    ///
    /// Think of it like a target:
    /// - Tolerance = 1e-4 (0.0001): Good enough for most purposes (default)
    /// - Tolerance = 1e-6: Very precise, but takes longer
    /// - Tolerance = 1e-2: Fast but less accurate
    ///
    /// For most applications, the default is fine. Only change this if:
    /// - You need very high precision (decrease tolerance)
    /// - Training is too slow (increase tolerance)
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets whether to use warm starting for cross-validation.
    /// </summary>
    /// <value>True to use warm starting; false otherwise. Default is true.</value>
    /// <remarks>
    /// <para>
    /// When warm starting is enabled and the model is retrained with a different alpha,
    /// the previous solution is used as the starting point. This can significantly
    /// speed up cross-validation when testing multiple alpha values.
    /// </para>
    /// <para><b>For Beginners:</b> Warm starting speeds up training when trying different alpha values.
    ///
    /// Imagine you're training models with alpha = 0.1, 0.5, 1.0, 2.0:
    /// - Without warm start: Each model starts from scratch
    /// - With warm start: Each model starts from where the previous one ended
    ///
    /// This is faster because solutions for similar alpha values are similar.
    /// Keep this enabled (default) unless you have a specific reason to disable it.
    /// </para>
    /// </remarks>
    public bool WarmStart { get; set; } = true;
}
