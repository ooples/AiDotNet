using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for optimization algorithms used in machine learning models.
/// </summary>
/// <remarks>
/// An optimizer is responsible for finding the best parameters for a machine learning model
/// by minimizing or maximizing an objective function.
/// 
/// <b>For Beginners:</b> Think of an optimizer as a "tuning expert" that adjusts your model's settings
/// to make it perform better. Just like tuning a radio to get the clearest signal, an optimizer
/// tunes your model's parameters to get the best predictions.
/// 
/// Common examples of optimizers include:
/// - Gradient Descent: Gradually moves toward better parameters by following the slope
/// - Adam: An advanced optimizer that adapts its learning rate for each parameter
/// - L-BFGS: Works well for smaller datasets and uses memory of previous steps
/// 
/// Why optimizers matter:
/// - They determine how quickly your model learns
/// - They affect whether your model finds the best solution or gets stuck
/// - Different optimizers work better for different types of problems
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IOptimizer<T, TInput, TOutput> : IModelSerializer
{
    /// <summary>
    /// Performs the optimization process to find the best parameters for a model.
    /// </summary>
    /// <remarks>
    /// This method takes input data and attempts to find the optimal parameters
    /// that minimize or maximize the objective function.
    /// 
    /// <b>For Beginners:</b> This is where the actual "learning" happens. The optimizer looks at your data
    /// and tries different parameter values to find the ones that make your model perform best.
    /// 
    /// The process typically involves:
    /// 1. Evaluating how well the current parameters perform
    /// 2. Calculating how to change the parameters to improve performance
    /// 3. Updating the parameters
    /// 4. Repeating until the model performs well enough or reaches a maximum number of attempts
    /// </remarks>
    /// <param name="inputData">The data needed for optimization, including the objective function,
    /// initial parameters, and any constraints.</param>
    /// <returns>The result of the optimization process, including the optimized parameters
    /// and performance metrics.</returns>
    OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData);

    /// <summary>
    /// Determines whether the optimization process should stop early.
    /// </summary>
    /// <remarks>
    /// Early stopping is a technique to prevent overfitting by stopping the optimization
    /// process before it completes all iterations if certain conditions are met.
    /// 
    /// <b>For Beginners:</b> This is like knowing when to stop cooking - if the model is "done" 
    /// (trained well enough), this method says "stop now" instead of continuing unnecessarily.
    /// 
    /// Common reasons for early stopping include:
    /// - The model's performance isn't improving anymore
    /// - The model's performance on validation data is getting worse (overfitting)
    /// - The changes in parameters are becoming very small (convergence)
    /// 
    /// Early stopping helps:
    /// - Save computation time
    /// - Prevent the model from becoming too specialized to the training data
    /// - Produce models that generalize better to new data
    /// </remarks>
    /// <returns>True if the optimization process should stop early; otherwise, false.</returns>
    bool ShouldEarlyStop();

    /// <summary>
    /// Gets the configuration options for the optimization algorithm.
    /// </summary>
    /// <remarks>
    /// These options control how the optimization algorithm behaves, including
    /// parameters like learning rate, maximum iterations, and convergence criteria.
    ///
    /// <b>For Beginners:</b> This provides the "settings" or "rules" that the optimizer follows.
    /// Just like a recipe has instructions (bake at 350°F for 30 minutes), an optimizer
    /// has settings (learn at rate 0.01, stop after 1000 tries).
    ///
    /// Common optimization options include:
    /// - Learning rate: How big of adjustments to make (step size)
    /// - Maximum iterations: How many attempts to make before giving up
    /// - Tolerance: How small an improvement is considered "good enough" to stop
    /// - Regularization: Settings that prevent the model from becoming too complex
    /// </remarks>
    /// <returns>The configuration options for the optimization algorithm.</returns>
    OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions();

    /// <summary>
    /// Resets the optimizer state to prepare for a fresh optimization run.
    /// </summary>
    /// <remarks>
    /// This method clears accumulated state including:
    /// - Model cache (prevents retrieving solutions from previous runs)
    /// - Fitness history (accumulated scores from previous optimizations)
    /// - Iteration history (logs from previous runs)
    /// - Adaptive parameters (learning rate, momentum reset to initial values)
    ///
    /// <b>For Beginners:</b> Think of this like "clearing the whiteboard" before starting a new problem.
    /// When you run optimization multiple times (like during cross-validation), you want each run
    /// to start fresh without being influenced by previous runs. This method ensures that.
    ///
    /// When to call Reset():
    /// - Before each cross-validation fold (ensures independent fold evaluations)
    /// - Before training the final model after cross-validation
    /// - Any time you want to reuse an optimizer for a completely new optimization task
    ///
    /// Why this matters:
    /// - Prevents state contamination between independent training runs
    /// - Ensures reproducible results regardless of how many times you've used the optimizer
    /// - Avoids memory leaks from accumulated history
    /// - Maintains correct adaptive learning rate dynamics
    /// </remarks>
    void Reset();
}
