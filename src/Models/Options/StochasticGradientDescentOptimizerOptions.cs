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
    /// Initializes a new instance of the StochasticGradientDescentOptimizerOptions class with appropriate defaults.
    /// </summary>
    public StochasticGradientDescentOptimizerOptions()
    {
        MaxIterations = 1000;
    }

    /// <summary>
    /// Gets or sets the batch size for stochastic gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 1 for true stochastic behavior.</value>
    /// <remarks>
    /// <para>
    /// The batch size determines how many samples are processed before updating the model parameters.
    /// A batch size of 1 represents true Stochastic Gradient Descent, processing one sample at a time.
    /// Larger batch sizes create mini-batch gradient descent behavior with SGD's update rule.
    /// </para>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model:
    ///
    /// - BatchSize = 1: True stochastic - update after each sample (default)
    /// - BatchSize = 32: Mini-batch - update after every 32 samples
    /// - BatchSize = [entire dataset]: Batch gradient descent
    ///
    /// Smaller batch sizes:
    /// - More frequent updates (faster convergence initially)
    /// - More noise in gradients (can help escape local minima)
    /// - Less efficient use of vectorized operations
    ///
    /// Larger batch sizes:
    /// - Smoother gradient estimates
    /// - Better use of GPU/vectorization
    /// - May require adjusting learning rate
    ///
    /// The default of 1 gives true stochastic behavior. Consider using
    /// MiniBatchGradientDescentOptimizer if you want mini-batch behavior with
    /// additional features like adaptive learning rates.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 1;
}
