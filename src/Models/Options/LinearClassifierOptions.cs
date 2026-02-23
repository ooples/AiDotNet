namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for linear classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Linear classifiers learn a linear decision boundary to separate classes.
/// They are simple, interpretable, and often very effective.
/// </para>
/// <para><b>For Beginners:</b> Linear classifiers draw a straight line (or hyperplane in
/// higher dimensions) to separate different classes of data.
///
/// Key concepts:
/// - They learn weights for each feature
/// - Prediction is: sign(weight * features + bias)
/// - Training adjusts weights to minimize errors
///
/// Linear classifiers are great when:
/// - You have many features (high-dimensional data)
/// - Data is approximately linearly separable
/// - You need fast training and prediction
/// - You want interpretable models (feature importance = weight magnitude)
/// </para>
/// </remarks>
public class LinearClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the learning rate for gradient-based optimization.
    /// </summary>
    /// <value>
    /// A positive learning rate. Default is 0.01.
    /// </value>
    /// <remarks>
    /// <para>
    /// Controls how much to adjust weights on each update.
    /// - Too high: Training may oscillate or diverge
    /// - Too low: Training will be slow to converge
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of training iterations.
    /// </summary>
    /// <value>
    /// The maximum iterations. Default is 1000.
    /// </value>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance.
    /// </summary>
    /// <value>
    /// The tolerance for convergence check. Default is 1e-4.
    /// </value>
    /// <remarks>
    /// Training stops early if the loss improvement is below this threshold.
    /// </remarks>
    public double Tolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets whether to fit an intercept term (bias).
    /// </summary>
    /// <value>
    /// True (default) to include an intercept; false to pass through origin.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The intercept (also called bias) allows the
    /// decision boundary to not pass through the origin.
    ///
    /// Almost always set to true unless you specifically know your data
    /// should be classified through the origin.
    /// </para>
    /// </remarks>
    public bool FitIntercept { get; set; } = true;

    /// <summary>
    /// Gets or sets the regularization strength (alpha).
    /// </summary>
    /// <value>
    /// A positive regularization parameter. Default is 0.0001.
    /// </value>
    /// <remarks>
    /// Higher values mean stronger regularization, which prevents overfitting
    /// but may increase bias. Set to 0 for no regularization.
    /// </remarks>
    public double Alpha { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets whether to shuffle training data at each epoch.
    /// </summary>
    /// <value>
    /// True (default) to shuffle; false to use data in original order.
    /// </value>
    public bool Shuffle { get; set; } = true;

    /// <summary>
    /// Gets or sets the penalty type for regularization.
    /// </summary>
    /// <value>
    /// The penalty type. Default is L2.
    /// </value>
    public LinearPenalty Penalty { get; set; } = LinearPenalty.L2;

    /// <summary>
    /// Gets or sets the loss function type.
    /// </summary>
    /// <value>
    /// The loss function. Default is Hinge (for SVM-style classifier).
    /// </value>
    public LinearLoss Loss { get; set; } = LinearLoss.Hinge;
}

/// <summary>
/// Penalty types for linear classifiers.
/// </summary>
public enum LinearPenalty
{
    /// <summary>
    /// No regularization.
    /// </summary>
    None,

    /// <summary>
    /// L1 (Lasso) regularization - promotes sparsity.
    /// </summary>
    L1,

    /// <summary>
    /// L2 (Ridge) regularization - prevents large weights.
    /// </summary>
    L2,

    /// <summary>
    /// Elastic Net - combination of L1 and L2.
    /// </summary>
    ElasticNet
}

/// <summary>
/// Loss functions for linear classifiers.
/// </summary>
public enum LinearLoss
{
    /// <summary>
    /// Hinge loss - used for SVM-style max-margin classification.
    /// </summary>
    Hinge,

    /// <summary>
    /// Squared hinge loss - smoother version of hinge loss.
    /// </summary>
    SquaredHinge,

    /// <summary>
    /// Log loss - logistic regression style.
    /// </summary>
    Log,

    /// <summary>
    /// Modified Huber loss - robust to outliers.
    /// </summary>
    ModifiedHuber,

    /// <summary>
    /// Perceptron loss - classic perceptron algorithm.
    /// </summary>
    Perceptron
}
