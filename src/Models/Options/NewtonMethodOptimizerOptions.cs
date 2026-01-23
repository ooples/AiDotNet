namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Newton's Method optimizer, an advanced second-order optimization technique
/// that uses both gradient and Hessian information to accelerate convergence in optimization problems.
/// </summary>
/// <remarks>
/// <para>
/// Newton's Method is a powerful optimization algorithm that leverages second-order derivatives 
/// (Hessian matrix) in addition to first-order gradients to determine optimal step directions and sizes. 
/// This approach can achieve faster convergence than first-order methods, particularly near the optimum 
/// and for well-conditioned problems. The method approximates the objective function locally as a quadratic 
/// function and steps directly toward the minimum of this approximation. This class provides configuration 
/// options to control the learning rate dynamics of the Newton optimizer, allowing for adaptive step sizing 
/// that can improve stability and convergence speed across different optimization landscapes.
/// </para>
/// <para><b>For Beginners:</b> Newton's Method is an advanced technique for helping AI models learn faster and more efficiently.
/// 
/// Imagine you're trying to find the lowest point in a valley while blindfolded:
/// - First-order methods (like regular gradient descent) only tell you which direction is downhill
/// - Newton's Method tells you both the downhill direction AND how curved the terrain is
/// 
/// This extra information about curvature helps the optimizer:
/// - Take larger steps when the terrain is relatively flat
/// - Take smaller, more careful steps when the terrain is highly curved
/// - Often reach the lowest point in fewer steps
/// 
/// Think of it like having a more intelligent navigation system:
/// - Regular gradient descent says "go downhill"
/// - Newton's Method says "go downhill, but adjust your stride based on the terrain"
/// 
/// This method typically excels when:
/// - The optimization problem is well-behaved (smooth, not too many bumps)
/// - You need fast convergence and can afford the extra computation
/// - The problem isn't too high-dimensional
/// 
/// The settings in this class let you control how the learning rate adapts during optimization,
/// balancing between speed and stability.
/// </para>
/// </remarks>
public class NewtonMethodOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for gradient computation.
    /// </summary>
    /// <value>A positive integer, defaulting to -1 (full batch).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples are used to calculate gradients.
    /// Newton's Method uses full-batch gradients (batch size -1) because it requires computing the Hessian
    /// matrix, which depends on consistent gradient and second derivative information from the entire dataset.
    /// Using mini-batches would introduce noise that makes the Hessian approximation unreliable.</para>
    /// </remarks>
    public int BatchSize { get; set; } = -1;

    /// <summary>
    /// Gets or sets the initial learning rate used by the Newton's Method optimizer.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the initial step size for the Newton method optimization process. Despite 
    /// Newton's Method theoretically being able to determine optimal step sizes automatically, in practice, 
    /// a learning rate is often applied to the Newton step to improve stability, especially when far from 
    /// the optimum or when the Hessian approximation is imperfect. The default value of 0.1 provides a 
    /// conservative starting point that balances convergence speed with numerical stability. This property 
    /// overrides the base class implementation with a value more appropriate for Newton's Method.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how big the initial steps are
    /// when the optimizer starts searching for the best solution.
    /// 
    /// The default value of 0.1 means:
    /// - The algorithm takes modest initial steps
    /// - This provides a balance between speed and stability
    /// 
    /// Think of it like exploring an unfamiliar area:
    /// - Too small steps (like 0.01) would be cautious but very slow
    /// - Too large steps (like 1.0) might overshoot important details
    /// - The default 0.1 is like a moderate walking pace - reasonable for most situations
    /// 
    /// You might want a higher value (like 0.5) if:
    /// - You're confident the function is well-behaved
    /// - Initial progress seems too slow
    /// - You have good Hessian estimates
    /// 
    /// You might want a lower value (like 0.05) if:
    /// - The optimization is unstable or diverging
    /// - The problem is known to be challenging
    /// - You're working with a poorly conditioned problem
    /// 
    /// Unlike in regular gradient descent, Newton's Method theoretically shouldn't need
    /// a learning rate at all, but in practice, this damping factor helps prevent
    /// instability when the quadratic approximation isn't perfect.
    /// </para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is increased when the algorithm
    /// is making good progress.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how rapidly the learning rate can grow when consecutive iterations
    /// show improvements in the optimization objective. A value of 1.05 means the learning rate 
    /// can increase by 5% per successful iteration, gradually accelerating the optimization process
    /// when moving in promising directions. While Newton's Method has theoretical guarantees for 
    /// quadratic functions, adaptive learning rates help handle non-quadratic regions of the objective 
    /// function and imperfect Hessian approximations. Higher values enable more aggressive acceleration 
    /// but may reduce stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much the step size increases
    /// when the algorithm is successfully moving toward better solutions.
    /// 
    /// The default value of 1.05 means:
    /// - After each successful step, the learning rate grows by 5%
    /// - This allows the algorithm to gradually speed up when things are going well
    /// 
    /// Using our navigation analogy:
    /// - When you're confident you're on the right path, you gradually walk faster
    /// - This adaptive pace helps you reach your destination more efficiently
    /// 
    /// You might want a higher value (like 1.1) if:
    /// - The optimization seems to be progressing too slowly
    /// - You're confident in the stability of your problem
    /// - You want to reach convergence more quickly
    /// 
    /// You might want a lower value (closer to 1.0) if:
    /// - The optimization becomes unstable with larger steps
    /// - You prefer a more cautious approach
    /// - The problem has delicate features that require careful exploration
    /// 
    /// This adaptive increase helps Newton's Method balance between taking advantage
    /// of its powerful second-order information while maintaining stability in
    /// real-world optimization problems.
    /// </para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is decreased when the algorithm
    /// is not making good progress.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the learning rate is reduced when the optimization
    /// algorithm encounters difficulties or does not improve the objective function. A value
    /// of 0.95 means the learning rate decreases by 5% when progress stalls, allowing the
    /// algorithm to take more cautious steps in challenging regions of the parameter space.
    /// This adaptive behavior is particularly important for Newton's Method when encountering
    /// non-quadratic regions, ill-conditioned Hessians, or saddle points, where the standard
    /// Newton step might be too aggressive or point in unhelpful directions.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the step size decreases
    /// when the algorithm stops making progress or moves in the wrong direction.
    /// 
    /// The default value of 0.95 means:
    /// - After an unsuccessful step, the learning rate shrinks by 5%
    /// - This makes the algorithm more cautious when it encounters difficulties
    /// 
    /// Continuing our navigation analogy:
    /// - When the terrain becomes tricky or you start going the wrong way
    /// - You slow down and take smaller, more careful steps
    /// 
    /// You might want a lower value (like 0.8) if:
    /// - The optimization frequently becomes unstable
    /// - You want to quickly reduce step size after poor steps
    /// - The function has many difficult regions that require careful navigation
    /// 
    /// You might want a higher value (closer to 1.0) if:
    /// - You want to maintain faster progress even through challenging regions
    /// - Minor setbacks shouldn't dramatically slow the optimization
    /// - You're confident in the overall stability of your approach
    /// 
    /// This adaptive decrease helps Newton's Method remain stable and effective
    /// even when the theoretical assumptions of the method aren't fully met in
    /// practical optimization problems.
    /// </para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
}
