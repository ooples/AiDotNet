namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Quantile Regression, a technique that enables prediction of specific
/// quantiles of the conditional distribution rather than just the conditional mean.
/// </summary>
/// <remarks>
/// <para>
/// Quantile Regression extends traditional regression methods by estimating conditional quantiles
/// of the response variable. While standard regression estimates the conditional mean E(Y|X),
/// Quantile Regression can estimate any conditional quantile Q(a|X) for a ? (0,1), including
/// medians (a = 0.5) and other percentiles. This technique provides a more comprehensive view of the
/// relationship between variables, allowing for the analysis of the full conditional distribution.
/// It is particularly valuable when the conditional distribution is non-Gaussian, skewed, or when
/// outliers are present. Quantile Regression is also robust to heteroscedasticity (non-constant variance)
/// and can reveal how different parts of the distribution respond differently to predictor variables.
/// </para>
/// <para><b>For Beginners:</b> Quantile Regression helps predict specific percentiles of possible outcomes, not just the average outcome.
/// 
/// Think about salary predictions:
/// - Regular regression might tell you "the average salary for this job is $75,000"
/// - But Quantile Regression could tell you:
///   - "10% of people in this job earn less than $50,000" (10th percentile)
///   - "Half of people in this job earn less than $70,000" (median or 50th percentile)
///   - "90% of people in this job earn less than $120,000" (90th percentile)
/// 
/// What this technique does:
/// - It focuses on specific slices of the data distribution
/// - Instead of minimizing squared errors (as in mean regression)
/// - It minimizes a different loss function that depends on which quantile you want
/// - This gives you insight into different parts of the outcome distribution
/// 
/// This is especially useful when:
/// - The outcomes aren't evenly distributed around the average
/// - You're interested in extreme cases (very high or low values)
/// - Different factors might affect different parts of the distribution differently
/// - You want to understand risk or uncertainty better
/// 
/// For example, in healthcare, knowing that a treatment reduces the risk of severe complications
/// (the high quantile) is different information than knowing it reduces the average symptom severity.
///
/// This class lets you configure how the quantile regression algorithm operates.
/// </para>
/// </remarks>
public class QuantileRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the quantile to be estimated by the regression model.
    /// </summary>
    /// <value>The quantile value between 0 and 1, defaulting to 0.5 (median regression).</value>
    /// <remarks>
    /// <para>
    /// This parameter determines which quantile of the conditional distribution the model will estimate.
    /// The quantile must be a value between 0 and 1, where 0.5 represents the median (50th percentile),
    /// 0.9 represents the 90th percentile, 0.1 represents the 10th percentile, and so on. Setting this
    /// value to 0.5 (the default) results in median regression, which is more robust to outliers than
    /// mean regression. Lower values (e.g., 0.1) focus on the lower tail of the distribution, while
    /// higher values (e.g., 0.9) focus on the upper tail. Different quantiles can reveal how the
    /// relationship between predictors and the response variable changes across the distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls which percentile of the data you want to predict.
    /// 
    /// The default value of 0.5 means:
    /// - You're trying to predict the median (middle value)
    /// - Half of outcomes will likely be above your prediction
    /// - Half of outcomes will likely be below your prediction
    /// 
    /// Think of it like height predictions:
    /// - Quantile 0.5 (median): The height where half of people are taller and half are shorter
    /// - Quantile 0.9 (90th percentile): The height where only 10% of people are taller
    /// - Quantile 0.1 (10th percentile): The height where 90% of people are taller
    /// 
    /// You might choose different quantiles for different purposes:
    /// - 0.5 for a typical or central prediction (median)
    /// - 0.9 or higher when you're concerned about upper limits or worst-case scenarios
    /// - 0.1 or lower when you're concerned about lower limits or best-case scenarios
    /// - Multiple quantiles (running the model multiple times) to get a full picture of possibilities
    /// 
    /// For example, in flood risk modeling, the 0.99 quantile might tell you the water level that
    /// has only a 1% chance of being exceeded - critical information for safety planning.
    /// </para>
    /// </remarks>
    public double Quantile { get; set; } = 0.5; // Default to median regression

    /// <summary>
    /// Gets or sets the learning rate for the optimization algorithm.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The learning rate controls the step size in the gradient-based optimization process used to
    /// estimate the quantile regression model. A smaller learning rate leads to more precise but slower
    /// convergence, while a larger learning rate can speed up learning but risks overshooting the optimal
    /// solution or causing instability. The appropriate learning rate depends on the scale and characteristics
    /// of the data. If the algorithm fails to converge or produces poor results, adjusting this parameter
    /// may help. For quantile regression, the optimization landscape can sometimes be less smooth than for
    /// mean regression, potentially requiring a smaller learning rate for stable convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm adjusts its predictions during training.
    /// 
    /// The default value of 0.01 means:
    /// - The algorithm takes moderate-sized steps when improving its model
    /// - It's a balance between learning speed and learning stability
    /// 
    /// Think of it like walking down a hill blindfolded:
    /// - A small learning rate (like 0.001) means taking tiny, cautious steps
    ///   - Safer but takes much longer to reach the bottom
    /// - A large learning rate (like 0.1) means taking big, bold steps
    ///   - Faster but you might trip or miss the lowest point entirely
    /// 
    /// You might want a smaller learning rate (like 0.001):
    /// - When your model is unstable or oscillating during training
    /// - When you need very precise results and have time for longer training
    /// - When working with data that has very large or varied scales
    /// 
    /// You might want a larger learning rate (like 0.05):
    /// - When training is taking too long
    /// - When you're doing initial exploration
    /// - When you have a lot of data and want faster convergence
    /// 
    /// Finding the right learning rate often requires experimentation - too small and training takes forever,
    /// too large and the model might never find the best solution.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter limits the number of iterations the optimization algorithm will perform when
    /// fitting the quantile regression model. It serves as a stopping criterion to prevent excessive
    /// computation time in cases where convergence is slow or not achieved. The algorithm may terminate
    /// earlier if convergence is detected before reaching this limit. For simpler datasets or higher
    /// learning rates, fewer iterations may be sufficient, while complex relationships or lower learning
    /// rates might require more iterations to reach convergence. Monitoring the convergence behavior
    /// can help in determining an appropriate value for this parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many times the algorithm will try to improve its predictions before stopping.
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will make up to 1000 attempts to refine its model
    /// - It might stop earlier if it determines the model isn't improving anymore
    /// 
    /// Think of it like a game where you're trying to guess a number:
    /// - Each iteration is one guess
    /// - After each guess, you get feedback to improve your next guess
    /// - MaxIterations is like setting a limit: "I'll stop after 1000 guesses even if I haven't found the exact answer"
    /// 
    /// You might want more iterations (like 5000 or 10000):
    /// - When your data is complex or has subtle patterns
    /// - When using a very small learning rate
    /// - When you need high precision and have the computational resources
    /// 
    /// You might want fewer iterations (like 500 or 100):
    /// - When you need faster training times
    /// - When you have simpler data relationships
    /// - When you're doing exploratory analysis and don't need perfect results
    /// - When you notice the model stops improving well before 1000 iterations
    /// 
    /// This setting helps prevent the model from training indefinitely, ensuring it eventually completes
    /// even if it hasn't reached perfect convergence.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;
}
