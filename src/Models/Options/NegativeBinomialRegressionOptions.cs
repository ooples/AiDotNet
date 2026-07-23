namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Negative Binomial Regression, a statistical model used for count data
/// that exhibits overdispersion (variance exceeding the mean).
/// </summary>
/// <remarks>
/// <para>
/// Negative Binomial Regression extends Poisson regression by introducing an additional parameter
/// that allows the variance to exceed the mean, making it suitable for overdispersed count data.
/// This model is appropriate when analyzing count outcomes (like number of events, occurrences, or items)
/// that show greater variability than would be expected under a Poisson distribution. The model is
/// typically fitted using maximum likelihood estimation, optimized through iterative methods such as
/// Fisher scoring or Newton-Raphson iterations.
/// </para>
/// <para><b>For Beginners:</b> Negative Binomial Regression is a specialized technique for analyzing
/// count data - data where you're counting how many times something happens.
/// 
/// While Poisson regression is commonly used for count data, it assumes that the mean and variance are equal.
/// In real-world data, we often see more variability than Poisson allows for:
/// 
/// Think of it like predicting daily customer counts at a restaurant:
/// - Some days might have 50 customers, others 150, with an average of 100
/// - Poisson would expect most days to be fairly close to 100
/// - But real data often shows more extreme values (very busy days, very slow days)
/// - Negative Binomial can handle this extra variability
/// 
/// This model is particularly useful when:
/// - You're counting events (visits, purchases, accidents, etc.)
/// - Your data shows "clumping" or extra variation
/// - Some counts are much higher or lower than the average would suggest
/// 
/// This class lets you configure how the model learns from your data to make accurate predictions
/// despite this extra variability.
/// </para>
/// </remarks>
public class NegativeBinomialRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations allowed for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper limit on how many iterations the optimization algorithm will perform
    /// when fitting the model. The algorithm may terminate earlier if convergence is achieved based on
    /// the tolerance value. In Negative Binomial Regression, each iteration updates both the regression
    /// coefficients and the dispersion parameter to better fit the training data. The appropriate number
    /// of iterations depends on the complexity of the data and the initial values used.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many attempts the algorithm makes
    /// to fine-tune the model before giving up.
    /// 
    /// Imagine you're adjusting the seasoning in a recipe:
    /// - Each "iteration" is like tasting the dish and adding a little more of this or that
    /// - You keep making adjustments until you're satisfied with the taste
    /// - This parameter is your maximum number of taste-and-adjust cycles
    /// 
    /// The default value of 100 means the algorithm will make at most 100 attempts to improve its model.
    /// 
    /// You might want to increase this value if:
    /// - Your model is complex or has many features
    /// - You notice the model hasn't fully converged when training completes
    /// - You see in the logs that the algorithm is still making significant improvements
    ///   when it reaches iteration 100
    /// 
    /// You might want to decrease this value if:
    /// - You need faster training times
    /// - You're doing preliminary exploration
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
    /// <value>The convergence tolerance, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for determining when the optimization has converged. The algorithm
    /// will stop when the improvement in the log-likelihood between consecutive iterations falls below this
    /// tolerance value. A smaller tolerance requires more precision in the parameter estimates, potentially
    /// leading to better model performance but requiring more iterations. A larger tolerance allows for earlier
    /// termination but might result in less optimal parameter estimates.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much improvement is considered
    /// "good enough" to stop training.
    /// 
    /// Continuing with our recipe analogy:
    /// - As you make adjustments, the dish gets better
    /// - Eventually, the improvements become very small
    /// - This setting decides when those improvements are small enough to stop tasting and adjusting
    /// 
    /// The default value of 0.000001 (one millionth) means:
    /// - If an iteration improves the model by less than one millionth of its current quality
    /// - The algorithm will decide it's "good enough" and stop
    /// 
    /// This is a fairly strict setting, appropriate for many statistical modeling applications
    /// where high precision is desirable.
    /// 
    /// You might want to increase this value (make it less strict, like 1e-4) if:
    /// - Training is taking too long
    /// - You're doing preliminary analysis
    /// - You don't need extremely precise parameter estimates
    /// 
    /// You might want to decrease this value (make it more strict, like 1e-8) if:
    /// - You need very precise parameter estimates
    /// - You're working on a problem where tiny differences matter
    /// - You have the computational resources for longer training
    /// 
    /// Finding the right tolerance is about balancing precision with computational efficiency.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the matrix decomposition type to use when solving the weighted least squares problem.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to QR decomposition.</value>
    /// <remarks>
    /// <para>
    /// The decomposition type determines how the system of linear equations is solved during each iteration
    /// of the IRLS algorithm. QR decomposition is a numerically stable choice suitable for most problems.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the mathematical method used to solve equations
    /// during model fitting. The default QR method works well for most cases - you typically don't need
    /// to change this unless you have specific performance or numerical stability requirements.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Qr;
}
