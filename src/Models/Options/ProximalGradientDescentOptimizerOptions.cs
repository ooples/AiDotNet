namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Proximal Gradient Descent optimizer, an advanced optimization algorithm
/// that combines traditional gradient descent with proximal operators to handle regularization effectively.
/// </summary>
/// <remarks>
/// <para>
/// Proximal Gradient Descent is an extension of standard gradient descent that is particularly effective
/// for solving optimization problems with regularization terms. It alternates between standard gradient steps
/// on the smooth part of the objective function and proximal operations on the non-smooth regularization terms.
/// This approach is especially valuable for problems involving L1 regularization (which promotes sparsity) or
/// other complex regularization schemes that are difficult to optimize with standard gradient methods. The
/// proximal approach helps maintain desirable properties of the regularization while ensuring stable convergence.
/// It is widely used in machine learning for training models where specific structural properties (like sparsity,
/// group structure, or low rank) are desired in the solution.
/// </para>
/// <para><b>For Beginners:</b> Proximal Gradient Descent is a specialized optimization method that helps train machine learning models with regularization.
/// 
/// Imagine you're trying to find the lowest point in a hilly landscape while also staying within certain boundaries:
/// - Regular gradient descent is like always walking directly downhill
/// - But sometimes this approach can lead you to areas that are too complex or "overfit" to your training data
/// - Regularization adds "penalty zones" to discourage overly complex solutions
/// - Proximal gradient descent helps navigate these penalty zones effectively
/// 
/// What this optimizer does:
/// 1. Takes a step in the direction that reduces prediction error (like regular gradient descent)
/// 2. Then takes a "proximal step" that handles the regularization penalties separately
/// 3. By splitting the process this way, it can find solutions that balance accuracy and simplicity
/// 
/// Think of it like training a dog:
/// - The gradient step teaches the dog to complete a task correctly
/// - The proximal step ensures the dog doesn't develop bad habits along the way
/// - Together, they produce well-behaved, effective results
/// 
/// This approach is particularly useful when you want your model to:
/// - Use only a subset of available features (sparsity)
/// - Group related features together
/// - Avoid extreme parameter values
/// 
/// This class lets you configure how this specialized optimization process works.
/// </para>
/// </remarks>
public class ProximalGradientDescentOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for proximal gradient descent.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the strength of the regularization term in the objective function.
    /// </summary>
    /// <value>The regularization strength, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the weight of the regularization term relative to the main loss function.
    /// Higher values increase the influence of regularization, promoting simpler models (e.g., sparser
    /// weights with L1 regularization or smaller weights with L2 regularization). Lower values reduce
    /// the regularization effect, allowing the model to focus more on minimizing the loss function.
    /// The optimal value depends on the specific problem, data characteristics, and the type of regularization
    /// being used. This parameter is often one of the most important hyperparameters to tune, as it directly
    /// controls the trade-off between fitting the training data and maintaining model simplicity.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strongly the regularization penalties affect your model.
    /// 
    /// The default value of 0.01 means:
    /// - The regularization has a moderate influence on the model
    /// - There's a balance between minimizing errors and keeping the model simple
    /// 
    /// Think of regularization like a budget constraint:
    /// - Your model wants to "spend" parameter values to fit the data perfectly
    /// - Regularization sets a "budget" that limits this spending
    /// - Higher RegularizationStrength means a tighter budget (simpler model)
    /// - Lower RegularizationStrength means a looser budget (potentially more complex model)
    /// 
    /// You might want a higher value (like 0.1 or 1.0):
    /// - When you suspect your model is overfitting
    /// - When you have limited training data
    /// - When you want to encourage sparse solutions (with L1 regularization)
    /// - When you want smaller parameter values overall (with L2 regularization)
    /// 
    /// You might want a lower value (like 0.001 or 0.0001):
    /// - When you have abundant training data
    /// - When underfitting is more of a concern than overfitting
    /// - When you want the model to focus more on minimizing training error
    /// - When you have complex patterns that require more expressive models
    /// 
    /// Finding the right regularization strength often requires experimentation with different values.
    /// </para>
    /// </remarks>
    public double RegularizationStrength { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the step size for the proximal operator component of the algorithm.
    /// </summary>
    /// <value>The proximal step size, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the size of the step taken during the proximal update phase of the algorithm.
    /// The proximal step specifically handles the regularization term, separate from the gradient step that
    /// addresses the smooth part of the objective function. A larger proximal step size makes the algorithm
    /// more aggressive in enforcing regularization constraints, while a smaller value makes it more conservative.
    /// The optimal value depends on the regularization type and strength, as well as the overall optimization
    /// landscape. In many cases, this parameter interacts with the RegularizationStrength and may need to be
    /// tuned accordingly. Too large a value can cause instability, while too small a value can slow convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how aggressively the algorithm enforces regularization in each iteration.
    /// 
    /// The default value of 0.1 means:
    /// - The algorithm takes moderate-sized steps when applying regularization
    /// - This provides a balance between rapid convergence and stability
    /// 
    /// Think of the proximal step as the "correction" after the main step:
    /// - First, the algorithm takes a step to reduce prediction errors
    /// - Then, it takes a proximal step to enforce simplicity (regularization)
    /// - This setting controls the size of that second, corrective step
    /// 
    /// You might want a larger value (like 0.3 or 0.5):
    /// - When you want regularization effects to be applied more quickly
    /// - When the regularization term is well-behaved and unlikely to cause instability
    /// - When faster convergence is a priority
    /// 
    /// You might want a smaller value (like 0.05 or 0.01):
    /// - When you notice instability in the optimization process
    /// - When using strong regularization that could cause large parameter changes
    /// - When you prefer more gradual, stable convergence
    /// - When working with particularly complex or ill-conditioned problems
    /// 
    /// This parameter often needs to be adjusted in coordination with RegularizationStrength
    /// to achieve optimal results.
    /// </para>
    /// </remarks>
    public double ProximalStepSize { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of inner iterations for each main optimization iteration.
    /// </summary>
    /// <value>The number of inner iterations, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This parameter specifies the number of proximal gradient steps to perform within each outer 
    /// iteration of the optimization algorithm. In proximal methods, it's common to have an inner loop
    /// that refines the proximal update before proceeding to the next main iteration. More inner iterations
    /// can lead to more accurate proximal updates at the cost of increased computation time. The appropriate
    /// value depends on the complexity of the regularization term and the desired accuracy of the proximal
    /// mapping. For simple regularization schemes like L1 or L2, fewer inner iterations may be sufficient,
    /// while more complex regularization might benefit from additional inner refinement.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many mini-steps the algorithm takes for each main optimization step.
    /// 
    /// The default value of 10 means:
    /// - For each main iteration, the algorithm performs 10 smaller refinement steps
    /// - These refinement steps help ensure the regularization is properly applied
    /// 
    /// Think of it like polishing a surface:
    /// - The main algorithm makes large changes to get the general shape right
    /// - Then these inner iterations carefully refine and polish the result
    /// - More inner iterations mean more careful polishing
    /// 
    /// You might want more inner iterations (like 20 or 50):
    /// - When using complex regularization that requires careful handling
    /// - When high precision is important in your final model
    /// - When you notice the optimization isn't converging well with fewer iterations
    /// - When you have the computational resources to spare
    /// 
    /// You might want fewer inner iterations (like 5 or 3):
    /// - When using simple regularization schemes like basic L1 or L2
    /// - When computational efficiency is a priority
    /// - When you're in early experimental phases and need quick results
    /// - When you find that additional inner iterations don't improve results
    /// 
    /// More inner iterations typically mean better quality results but longer training times.
    /// </para>
    /// </remarks>
    public int InnerIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization process.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the maximum number of outer iterations the optimization algorithm will perform.
    /// It serves as a hard limit to prevent excessive computation time in cases where convergence is slow or
    /// not achieved. Each iteration involves a gradient step on the smooth part of the objective function 
    /// followed by a proximal operation on the regularization term, potentially with multiple inner iterations.
    /// The appropriate value depends on the complexity of the optimization problem, the desired precision,
    /// and the available computational resources. Note that this property hides (shadows) the MaxIterations 
    /// property inherited from the base GradientBasedOptimizerOptions class, potentially allowing for different
    /// default values or validation logic specific to proximal gradient descent.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the maximum number of major steps the algorithm will take before stopping.
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will take at most 1000 main optimization steps
    /// - It will stop earlier if it reaches convergence (finds a good solution)
    /// - But it won't continue beyond 1000 steps even if not fully converged
    /// 
    /// Think of it like setting a maximum travel time for a journey:
    /// - Ideally, you reach your destination before the time limit
    /// - But if the journey is taking too long, you stop when you hit the limit
    /// - This prevents the algorithm from running indefinitely on difficult problems
    /// 
    /// You might want more iterations (like 5000 or 10000):
    /// - For complex problems that need more time to converge
    /// - When you prioritize finding the best possible solution over speed
    /// - When early experiments show that 1000 iterations is insufficient
    /// - When you have the computational resources to spare
    /// 
    /// You might want fewer iterations (like 500 or 100):
    /// - When quick approximate solutions are preferred over perfect ones
    /// - For simpler problems that converge quickly
    /// - When running many experimental models where time is limited
    /// - When you find that the model converges well before reaching the limit
    /// 
    /// Note: This setting overrides the MaxIterations from the parent class to provide a default
    /// specifically calibrated for proximal gradient descent.
    /// </para>
    /// </remarks>
    public new int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the multiplicative factor for increasing the learning rate during adaptive optimization.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how aggressively the learning rate increases when the optimization is making
    /// good progress. In adaptive optimization schemes, the learning rate may be increased when successive
    /// iterations show consistent improvement. A factor of 1.05 means the learning rate increases by 5% when
    /// conditions warrant an increase. Higher values lead to more aggressive acceleration when the optimization
    /// is progressing well, potentially speeding up convergence. However, too large a value can cause instability
    /// by increasing the learning rate too quickly. The optimal value depends on the optimization landscape and
    /// the balance needed between convergence speed and stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm speeds up when it's making good progress.
    /// 
    /// The default value of 1.05 means:
    /// - When the algorithm is successfully reducing the error
    /// - It will increase its step size by 5% to move faster
    /// - This allows it to accelerate when it's on a promising path
    /// 
    /// Think of it like adjusting your walking speed:
    /// - When you're confident you're heading in the right direction, you walk a bit faster
    /// - This value determines how much faster you go when things are working well
    /// - A value of 1.05 represents a cautious acceleration (5% increase)
    /// 
    /// You might want a higher value (like 1.1 or 1.2):
    /// - When you want more aggressive acceleration
    /// - When the optimization landscape is smooth and well-behaved
    /// - When faster convergence is a priority and some instability is acceptable
    /// - When you've noticed that optimization progress is consistently positive
    /// 
    /// You might want a lower value (like 1.02 or 1.01):
    /// - When you've observed instability in the optimization process
    /// - When working with complex, ill-conditioned problems
    /// - When very stable, predictable convergence is more important than speed
    /// - When small parameter changes can dramatically affect model performance
    /// 
    /// This parameter works in tandem with LearningRateDecreaseFactor to adaptively adjust the
    /// optimization speed based on progress.
    /// </para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the multiplicative factor for decreasing the learning rate when progress stalls or reverses.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the learning rate decreases when the optimization encounters
    /// difficulties or appears to be overshooting. In adaptive optimization schemes, the learning rate is
    /// typically decreased when an iteration fails to improve the objective function or when other indicators
    /// suggest the current step size is too large. A factor of 0.95 means the learning rate decreases by 5%
    /// when conditions warrant a decrease. Lower values (further from 1.0) lead to more aggressive deceleration
    /// when problems are encountered, which can help stabilize the optimization but may slow convergence.
    /// The optimal value depends on the optimization landscape and the balance needed between convergence
    /// speed and stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm slows down when it encounters problems.
    /// 
    /// The default value of 0.95 means:
    /// - When the algorithm takes a step that doesn't improve the solution
    /// - It will reduce its step size by 5% to be more careful
    /// - This helps prevent overshooting or bouncing around without progress
    /// 
    /// Think of it like navigating a tricky path:
    /// - When you encounter obstacles or start to lose your way, you slow down
    /// - This value determines how much you slow down when things get difficult
    /// - A value of 0.95 represents a moderate slowdown (5% decrease)
    /// 
    /// You might want a lower value (like 0.8 or 0.9):
    /// - When you want more drastic slowdowns after bad steps
    /// - When the optimization landscape has sharp valleys or discontinuities
    /// - When stability is much more important than speed
    /// - When you've observed the algorithm overshooting repeatedly
    /// 
    /// You might want a higher value (like 0.98 or 0.99):
    /// - When you want only minor slowdowns after bad steps
    /// - When you're concerned about convergence becoming too slow
    /// - When occasional bad steps are expected and shouldn't trigger major adjustments
    /// - When the optimization landscape is relatively smooth
    /// 
    /// This parameter works in tandem with LearningRateIncreaseFactor to adaptively adjust the
    /// optimization speed based on progress.
    /// </para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
}
