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
    /// Resets the optimizer to its initial state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the model that this optimizer is configured to optimize.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the model that the optimizer is working with.
    /// It allows for consistency checks between the model used by the optimizer and
    /// the model used elsewhere in the pipeline.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you see which model the optimizer is currently
    /// set up to work with. It's important for maintaining consistency, ensuring that
    /// the optimizer is working with the model you expect.
    /// </para>
    /// </remarks>
    IFullModel<T, TInput, TOutput> Model { get; }

    /// <summary>
    /// Performs a single optimization step, updating the model parameters based on gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method performs one iteration of parameter updates using the current gradients.
    /// It's typically called after computing gradients through backpropagation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like taking one small step toward a better model.
    /// After calculating how wrong the model is (gradients), this method adjusts the
    /// model's parameters slightly to make it more accurate.
    ///
    /// Think of it like adjusting a recipe:
    /// 1. You taste the dish (check model performance)
    /// 2. You determine what needs changing (calculate gradients)
    /// 3. You adjust the ingredients (this Step method updates parameters)
    /// 4. Repeat until the dish tastes good (model is accurate)
    ///
    /// Most training loops call this method many times, each time making the model
    /// a little bit better.
    /// </para>
    /// </remarks>
    void Step();

    /// <summary>
    /// Calculates the parameter update based on the provided gradients.
    /// </summary>
    /// <param name="gradients">The gradients used to compute the parameter updates.</param>
    /// <returns>The calculated parameter updates as a dictionary mapping parameter names to their update vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method computes how much each parameter should change based on the gradients
    /// from backpropagation. It applies the optimizer's specific update rule (e.g., SGD, Adam, RMSProp)
    /// to transform gradients into actual parameter updates.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After your model calculates how wrong it is (gradients), this method
    /// figures out exactly how much to adjust each parameter to improve the model. Different optimizers
    /// use different strategies - some make simple adjustments, while others use momentum or adaptive
    /// learning rates to make smarter updates.
    ///
    /// Think of it like GPS navigation:
    /// 1. Gradients tell you "you're 10 miles off course"
    /// 2. This method decides "adjust your route by 2 miles north, 3 miles east"
    /// 3. The optimizer's strategy determines how aggressive the adjustment should be
    ///
    /// This is commonly used in federated learning and distributed training scenarios where
    /// updates need to be calculated separately from their application.
    /// </para>
    /// </remarks>
    Dictionary<string, Vector<T>> CalculateUpdate(Dictionary<string, Vector<T>> gradients);
}