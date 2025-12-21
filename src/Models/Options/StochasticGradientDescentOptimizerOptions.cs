namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Stochastic Gradient Descent (SGD) optimization, a widely used
/// algorithm for training machine learning models with large datasets.
/// </summary>
/// <remarks>
/// <para>
/// Stochastic Gradient Descent (SGD) is a variation of the gradient descent optimization algorithm that 
/// updates model parameters using gradients calculated from randomly selected subsets of the training data 
/// (mini-batches) rather than the entire dataset. This approach significantly reduces computational cost 
/// per iteration, making it suitable for large-scale machine learning problems. SGD introduces randomness 
/// into the optimization process, which can help escape local minima and potentially find better solutions. 
/// However, this randomness also leads to noisier updates and potentially slower convergence compared to 
/// full-batch gradient descent. This class inherits from GradientBasedOptimizerOptions and overrides the 
/// MaxIterations property to provide a more appropriate default value for SGD optimization.
/// </para>
/// <para><b>For Beginners:</b> Stochastic Gradient Descent is a faster way to train machine learning models with large datasets.
/// 
/// When training machine learning models:
/// - We need to find the best parameters that minimize errors
/// - Traditional gradient descent uses the entire dataset for each update
/// - This becomes very slow with large datasets
/// 
/// Stochastic Gradient Descent solves this by:
/// - Using only a small random subset of data (mini-batch) for each update
/// - Making many faster, approximate updates instead of fewer exact ones
/// - Eventually converging to a good solution, often more quickly
/// 
/// This approach offers several benefits:
/// - Much faster iterations, especially with large datasets
/// - Can escape local minima due to the noise in updates
/// - Often finds good solutions faster in practice
/// - Enables training on datasets too large to fit in memory
/// 
/// This class lets you configure the SGD optimization process.
/// </para>
/// </remarks>
public class StochasticGradientDescentOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization algorithm.
    /// </summary>
    /// <value>A positive integer, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of iterations (epochs) that the SGD algorithm will perform. 
    /// Each iteration processes one mini-batch of data. The optimization will stop either when this number of 
    /// iterations is reached or when another stopping criterion (such as convergence tolerance) is met, whichever 
    /// comes first. The default value of 1000 is suitable for many applications, but may need adjustment based on 
    /// the specific problem, dataset size, and mini-batch size. For complex problems or large datasets, more 
    /// iterations might be needed to reach convergence. This property overrides the MaxIterations property 
    /// inherited from the base class to provide a more appropriate default value for SGD optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how many rounds of updates the algorithm will perform.
    /// 
    /// The maximum iterations parameter:
    /// - Sets an upper limit on how many mini-batch updates will be performed
    /// - Prevents the algorithm from running indefinitely
    /// - Serves as a safety mechanism if convergence isn't reached
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will process at most 1000 mini-batches
    /// - This is often sufficient for many problems
    /// 
    /// Think of it like this:
    /// - Each iteration is one step toward finding the optimal solution
    /// - More iterations allow more steps and potentially better results
    /// - But there are diminishing returns after a certain point
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems that need more iterations to converge
    /// - Decrease it for simpler problems or when you need faster results
    /// - Monitor validation metrics to determine if more iterations are helpful
    /// 
    /// For example, when training a deep neural network on a large dataset,
    /// you might need to increase this to 5000 or more to reach optimal performance.
    /// </para>
    /// </remarks>
    public new int MaxIterations { get; set; } = 1000;
}
