namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Multinomial Logistic Regression, a classification method that generalizes
/// logistic regression to multiclass problems with more than two possible discrete outcomes.
/// </summary>
/// <remarks>
/// <para>
/// Multinomial Logistic Regression extends binary logistic regression to handle multiple classes by using
/// the softmax function rather than the sigmoid function. It models the probabilities of each class directly,
/// learning a set of coefficients for each class (except one reference class). The model is trained using
/// maximum likelihood estimation, typically optimized through iterative methods like gradient descent or
/// Newton's method. This approach is also known as Softmax Regression or Maximum Entropy Classification.
/// </para>
/// <para><b>For Beginners:</b> Multinomial Logistic Regression is a technique for classifying data into
/// multiple categories.
/// 
/// While regular Logistic Regression can only decide between two options (like "yes" or "no"),
/// Multinomial Logistic Regression can decide between many options - for example:
/// - Classifying emails as "work," "personal," or "spam"
/// - Identifying handwritten digits (0-9)
/// - Categorizing products into different types
/// 
/// Think of it like a voting system:
/// - Each feature in your data "votes" for different categories
/// - The model learns how much weight to give each feature's vote
/// - When making a prediction, it collects all these weighted votes 
/// - It converts these votes into probabilities for each category using a "softmax" function
/// - The category with the highest probability is the prediction
/// 
/// This class lets you configure how the model learns these voting weights from your training data.
/// </para>
/// </remarks>
public class MultinomialLogisticRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations allowed for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper limit on how many iterations the optimization algorithm will perform
    /// when fitting the model. The algorithm may terminate earlier if convergence is achieved based on the
    /// tolerance value. In multinomial logistic regression, each iteration updates the coefficient estimates
    /// to better fit the training data. More iterations allow for more precise coefficient estimates but
    /// increase computational cost. The appropriate value depends on the complexity of the problem and the
    /// characteristics of the dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many attempts the algorithm gets to
    /// find the best weights for its voting system.
    /// 
    /// Imagine you're adjusting the recipe for a cake:
    /// - Each "iteration" is like making the cake, tasting it, and adjusting the ingredients
    /// - You keep making adjustments until you're satisfied with the result
    /// - This parameter is your maximum number of attempts
    /// 
    /// The default value of 100 means the algorithm will make at most 100 attempts to improve its model.
    /// 
    /// You might want to increase this value if:
    /// - You have a complex dataset with many features and classes
    /// - You notice the model hasn't fully converged (settled on a solution) when training ends
    /// - You prioritize accuracy over training speed
    /// 
    /// You might want to decrease this value if:
    /// - You need faster training times
    /// - You're doing initial exploratory analysis
    /// - Your model converges quickly anyway
    /// 
    /// The algorithm might stop before reaching this maximum if it determines that additional
    /// iterations won't significantly improve the model (based on the Tolerance setting).
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance that determines when the optimization algorithm should stop.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.0001 (1e-4).</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for determining when the optimization has converged. The algorithm
    /// will stop when the improvement in the log-likelihood or cost function between consecutive iterations
    /// falls below this tolerance value. A smaller tolerance requires more precision in the coefficient estimates,
    /// potentially leading to better model performance but requiring more iterations. A larger tolerance allows
    /// for earlier termination but might result in less optimal coefficient estimates.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much improvement is considered
    /// "good enough" to keep training.
    /// 
    /// Continuing with our cake recipe analogy:
    /// - As you adjust ingredients, the cake gets better
    /// - Eventually, the improvements become very small
    /// - This setting decides when those improvements are small enough to stop
    /// 
    /// The default value of 0.0001 means:
    /// - If an iteration improves the model by less than 0.01%
    /// - The algorithm will decide it's "good enough" and stop
    /// 
    /// You might want to decrease this value (like to 0.00001) if:
    /// - You need very precise coefficient estimates
    /// - You're working on a problem where small differences matter
    /// - You have the computational resources for longer training
    /// 
    /// You might want to increase this value (like to 0.001) if:
    /// - You need faster training times
    /// - You're doing exploratory analysis
    /// - You're working with noisy data where high precision isn't meaningful
    /// 
    /// Finding the right tolerance is about balancing precision with computational efficiency -
    /// too strict and training takes forever, too lenient and the model might be suboptimal.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the matrix decomposition type to use when solving the linear system.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to QR decomposition.</value>
    /// <remarks>
    /// <para>
    /// The decomposition type determines how the system of linear equations is solved during optimization.
    /// QR decomposition is a numerically stable choice suitable for most problems.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the mathematical method used to solve equations
    /// during model fitting. The default QR method works well for most cases - you typically don't need
    /// to change this unless you have specific performance or numerical stability requirements.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Qr;
}
