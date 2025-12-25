namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Nesterov Accelerated Gradient optimization algorithm, a momentum-based
/// technique that improves convergence speed in gradient descent optimization.
/// </summary>
/// <remarks>
/// <para>
/// Nesterov Accelerated Gradient (NAG) is an enhancement to standard gradient descent optimization that 
/// incorporates momentum with a look-ahead approach. By evaluating gradients at a position estimated 
/// by the momentum term rather than the current position, NAG provides better responsiveness to changes 
/// in the error surface. This results in faster convergence rates and improved performance, particularly 
/// in problems with high curvature or narrow valleys in the error surface. The algorithm adaptively 
/// adjusts both learning rate and momentum during training to optimize performance.
/// </para>
/// <para><b>For Beginners:</b> Nesterov Accelerated Gradient is a technique that helps AI models learn faster and better.
/// 
/// Imagine you're trying to find the lowest point in a valley by walking downhill:
/// - Regular gradient descent is like always taking a step directly downhill from where you stand
/// - Adding momentum is like rolling a ball downhill - it picks up speed and can go faster
/// - Nesterov adds a clever twist: it looks ahead in the direction the ball is rolling before deciding which way is downhill
/// 
/// This "look-ahead" approach helps the model:
/// - Learn faster in most situations
/// - Avoid overshooting the best solution
/// - Navigate tricky terrain in the learning landscape
/// - Adapt to different types of problems
/// 
/// The settings in this class let you control:
/// - How quickly the learning rate and momentum can increase when things are going well
/// - How quickly they decrease when progress slows down
/// 
/// This adaptive behavior helps the model automatically find efficient settings as it learns,
/// rather than requiring you to find the perfect fixed values upfront.
/// </para>
/// </remarks>
public class NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for momentum-based optimizers.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is increased when the algorithm
    /// is making good progress.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how rapidly the learning rate can grow when consecutive iterations
    /// show improvements in the optimization objective. A value of 1.05 means the learning rate 
    /// can increase by 5% per successful iteration, allowing the algorithm to accelerate learning
    /// when moving in a promising direction. Higher values enable more aggressive acceleration but
    /// may lead to instability, while values closer to 1.0 provide more conservative adaptation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much bigger your learning steps
    /// can get when things are going well.
    /// 
    /// Imagine you're walking down a smooth, gentle slope:
    /// - You might start with small, careful steps
    /// - As you gain confidence that you're heading in the right direction, you might take larger steps
    /// - This parameter controls how much larger those steps can become
    /// 
    /// The default value of 1.05 means:
    /// - When the model is improving with each step
    /// - The step size can grow by 5% after each successful step
    /// 
    /// You might want a higher value (like 1.1) if:
    /// - Your model seems to learn very slowly
    /// - You're confident the learning landscape is smooth
    /// - You want to speed up training
    /// 
    /// You might want a lower value (closer to 1.0) if:
    /// - Your training seems unstable
    /// - The model's performance fluctuates wildly
    /// - You want more cautious, reliable learning
    /// 
    /// This adaptive step sizing helps the model learn efficiently without requiring you
    /// to manually find the perfect fixed learning rate.
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
    /// The appropriate value depends on the smoothness of the objective function and the
    /// presence of local minima or saddle points.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much smaller your learning steps
    /// become when the model stops improving.
    /// 
    /// Imagine you're hiking downhill and suddenly encounter a tricky, rocky section:
    /// - You'd naturally take smaller, more careful steps
    /// - This parameter controls how much more cautious you become
    /// 
    /// The default value of 0.95 means:
    /// - When the model is not improving or getting worse
    /// - The step size will shrink by 5% after each unsuccessful step
    /// 
    /// You might want a lower value (like 0.8) if:
    /// - You notice the model getting stuck in "bad" areas
    /// - Training often diverges or oscillates
    /// - You want to quickly reduce step size when things go wrong
    /// 
    /// You might want a higher value (closer to 1.0) if:
    /// - Progress is generally steady
    /// - You don't want to slow down too much when encountering small bumps
    /// - You're willing to risk some instability for faster training
    /// 
    /// This adaptive caution helps the model navigate difficult learning landscapes
    /// without getting permanently stuck or wildly unstable.
    /// </para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the factor by which the momentum is increased when the algorithm
    /// is making good progress.
    /// </summary>
    /// <value>The momentum increase factor, defaulting to 1.02.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how the momentum coefficient increases during successful optimization
    /// steps. In the Nesterov Accelerated Gradient method, momentum helps the algorithm maintain
    /// velocity in consistent directions, accelerating convergence. A value of 1.02 allows the
    /// momentum to build gradually (2% increase per successful iteration), providing stability
    /// while still adapting to the optimization landscape. This parameter overrides the base
    /// class implementation with behavior specific to NAG.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm builds up
    /// momentum when it's consistently moving in a good direction.
    /// 
    /// Think of momentum like rolling a ball downhill:
    /// - As it rolls, it picks up speed
    /// - This parameter controls how quickly it gains that speed
    /// 
    /// The default value of 1.02 means:
    /// - When the model is improving with consecutive steps
    /// - The momentum effect increases by 2% after each successful step
    /// 
    /// You might want a higher value (like 1.05) if:
    /// - Your model seems to hesitate too much
    /// - The error landscape has long, shallow slopes
    /// - You want to make faster progress
    /// 
    /// You might want a lower value (closer to 1.0) if:
    /// - Training seems to overshoot optimal values
    /// - The model oscillates around good solutions
    /// - You want more careful, controlled progress
    /// 
    /// Good momentum settings help the model learn faster by remembering the general
    /// direction of improvement rather than zigzagging down the slope.
    /// </para>
    /// </remarks>
    public new double MomentumIncreaseFactor { get; set; } = 1.02;

    /// <summary>
    /// Gets or sets the factor by which the momentum is decreased when the algorithm
    /// is not making good progress.
    /// </summary>
    /// <value>The momentum decrease factor, defaulting to 0.98.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how quickly the momentum coefficient is reduced when the optimization
    /// is not improving or is oscillating. When progress stalls, reducing momentum allows the algorithm
    /// to be more responsive to local gradient information rather than continuing in potentially
    /// unproductive directions. A value of 0.98 represents a modest 2% reduction in momentum per
    /// unsuccessful iteration. This parameter overrides the base class implementation with behavior
    /// specific to NAG.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm slows down
    /// (reduces momentum) when it's no longer making good progress.
    /// 
    /// Continuing with our rolling ball analogy:
    /// - If the ball is rolling in the wrong direction or is about to go uphill
    /// - You'd want it to slow down so it can change direction
    /// - This parameter controls how quickly that happens
    /// 
    /// The default value of 0.98 means:
    /// - When the model stops improving
    /// - The momentum effect decreases by 2% after each unsuccessful step
    /// 
    /// You might want a lower value (like 0.9) if:
    /// - Your model often overshoots good solutions
    /// - Training shows oscillations or unstable behavior
    /// - You want the ability to quickly change direction
    /// 
    /// You might want a higher value (closer to 1.0) if:
    /// - The learning landscape has many small local minima to avoid
    /// - You want to maintain direction through small bumps
    /// - You're confident in the general direction of optimization
    /// 
    /// Properly reducing momentum helps the model navigate challenging parts of the
    /// learning landscape without getting stuck in cycles or overshooting good solutions.
    /// </para>
    /// </remarks>
    public new double MomentumDecreaseFactor { get; set; } = 0.98;
}
