namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Kernel Ridge Regression, which combines ridge regression with the kernel trick
/// to model non-linear relationships in data.
/// </summary>
/// <remarks>
/// <para>
/// Kernel Ridge Regression (KRR) extends standard ridge regression by applying the "kernel trick" to
/// implicitly map input features into a higher-dimensional space where linear relationships can better
/// capture complex patterns in the data. This allows the model to learn non-linear relationships while
/// still maintaining the computational benefits and regularization of ridge regression.
/// </para>
/// <para><b>For Beginners:</b> Kernel Ridge Regression is a powerful technique that helps your model
/// capture complex, non-linear patterns in your data.
/// 
/// Imagine you have data points that can't be fit well with a straight line. For example, the relationship
/// between a car's speed and its fuel efficiency might be more of a U-shape (very efficient at moderate
/// speeds, less efficient at very low or very high speeds). Kernel Ridge Regression can help model these
/// curved relationships.
/// 
/// The "kernel trick" is like giving your model special glasses that let it see your data transformed in
/// a way that makes complex patterns easier to find. Instead of trying to fit a straight line to a curve,
/// it transforms the problem so that a straight line in the transformed space corresponds to a curve in
/// the original space.
/// 
/// The "ridge" part helps prevent overfitting by keeping the model from becoming too complex, similar to
/// how guardrails keep a car on the road.
/// 
/// This class inherits from NonLinearRegressionOptions, so all the general non-linear regression settings
/// are also available. The additional settings specific to Kernel Ridge Regression let you fine-tune how
/// the algorithm balances fitting the data versus keeping the model simple.</para>
/// </remarks>
public class KernelRidgeRegressionOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the regularization parameter (lambda) for Kernel Ridge Regression.
    /// </summary>
    /// <value>The regularization parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the strength of regularization applied to the model. Higher values result in
    /// stronger regularization, which helps prevent overfitting but may cause underfitting if set too high.
    /// Lower values allow the model to fit the training data more closely but may lead to overfitting if
    /// set too low.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the model prioritizes simplicity versus
    /// fitting the training data perfectly. With the default value of 1.0, the model tries to balance these
    /// two goals.
    /// 
    /// Think of it like adjusting the tension on a spring:
    /// - Higher values (e.g., 10.0) create more tension, pulling the model toward simplicity. This helps
    ///   prevent overfitting (where the model memorizes the training data but performs poorly on new data),
    ///   but if set too high, the model might be too simple to capture important patterns (underfitting).
    /// - Lower values (e.g., 0.1) reduce the tension, allowing the model to fit the training data more
    ///   closely. This can help capture complex patterns, but if set too low, the model might start
    ///   fitting to noise in the data (overfitting).
    /// 
    /// Finding the right value often requires experimentation. You might start with the default of 1.0 and
    /// then try values that are 10 times larger (10.0) or smaller (0.1) to see how they affect performance.
    /// Many practitioners use techniques like cross-validation to find the optimal value for their specific
    /// dataset.</para>
    /// </remarks>
    public double LambdaKRR { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the type of matrix decomposition used to solve the kernel ridge regression equations.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to Cholesky.</value>
    /// <remarks>
    /// <para>
    /// Different matrix decomposition methods offer different trade-offs between numerical stability,
    /// computational efficiency, and applicability to specific types of kernel matrices. Cholesky
    /// decomposition is generally faster but requires the kernel matrix to be positive definite, while
    /// SVD (Singular Value Decomposition) is more robust but computationally more expensive.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical method used to solve the
    /// equations in Kernel Ridge Regression. The default value (Cholesky) works well for most cases and
    /// is computationally efficient.
    /// 
    /// Think of matrix decomposition like solving a complex puzzle. There are different strategies to
    /// solve the puzzle, each with its own advantages:
    /// 
    /// - Cholesky decomposition (the default) is like a fast, efficient strategy that works well for
    ///   most standard puzzles. It's quick but might struggle with certain unusual puzzle types.
    /// 
    /// - SVD (Singular Value Decomposition) is like a more thorough, methodical strategy that can handle
    ///   even the most challenging puzzles, but takes longer to complete.
    /// 
    /// - QR decomposition falls somewhere in between, offering a balance of robustness and efficiency.
    /// 
    /// Most beginners should stick with the default Cholesky decomposition. You might consider changing
    /// this setting if:
    /// - You encounter numerical stability issues (error messages about matrix decomposition)
    /// - You're working with a very large dataset and need to optimize performance
    /// - You're using a specialized kernel function that creates unusual kernel matrices
    /// 
    /// In these cases, trying a different decomposition type might help resolve issues or improve
    /// performance.</para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}
