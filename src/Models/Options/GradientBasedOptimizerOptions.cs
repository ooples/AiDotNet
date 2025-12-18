namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for gradient-based optimization algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Gradient-based optimizers are algorithms that find the minimum or maximum of a function
/// by following the direction of steepest descent or ascent (the gradient).
/// </para>
/// <para><b>For Beginners:</b> Imagine you're in a hilly landscape and want to find the lowest point.
/// Gradient-based optimization is like always walking downhill in the steepest direction until you can't go any lower.
/// The "gradient" is simply the direction of the steepest slope at your current position.
/// </para>
/// <para>
/// These algorithms are fundamental to training many machine learning models, including neural networks,
/// linear regression, and logistic regression.
/// </para>
/// <para>
/// This class inherits from <see cref="OptimizationAlgorithmOptions"/>, which means it includes all the
/// base configuration options for optimization algorithms plus any additional options specific to
/// gradient-based methods.
/// </para>
/// </remarks>
public class GradientBasedOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the gradient cache to use for storing and retrieving computed gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The gradient cache helps avoid redundant gradient calculations by storing previously computed gradients.
    /// This can significantly improve performance, especially when the same model is evaluated multiple times.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as a memory that stores calculations you've already done.
    /// Instead of recalculating the same gradient multiple times, the optimizer can look it up in this cache,
    /// saving computational resources.
    /// </para>
    /// </remarks>
    public IGradientCache<T> GradientCache { get; set; } = new DefaultGradientCache<T>();

    /// <summary>
    /// Gets or sets the loss function to use for evaluating model performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The loss function measures how well the model's predictions match the actual target values.
    /// Different loss functions are appropriate for different types of problems (e.g., regression vs. classification).
    /// </para>
    /// <para><b>For Beginners:</b> The loss function is like a scorecard that tells you how well your model is doing.
    /// A higher loss means worse performance, so the optimizer tries to find model parameters that minimize this loss.
    /// </para>
    /// </remarks>
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Gets or sets the regularization method to use for preventing overfitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Regularization adds a penalty for complexity to the loss function, which helps prevent the model
    /// from overfitting to the training data. Common regularization methods include L1 (Lasso) and L2 (Ridge).
    /// </para>
    /// <para><b>For Beginners:</b> Regularization is like adding a rule that says "keep it simple."
    /// It prevents your model from becoming too complex and fitting the training data too perfectly,
    /// which can actually hurt performance on new, unseen data.
    /// </para>
    /// </remarks>
    public IRegularization<T, TInput, TOutput> Regularization { get; set; } = new L2Regularization<T, TInput, TOutput>();
}