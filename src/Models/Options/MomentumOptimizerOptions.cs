namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Momentum Optimizer, which enhances gradient descent by adding
/// a fraction of the previous update direction to the current update.
/// </summary>
/// <remarks>
/// <para>
/// The Momentum Optimizer is an extension of gradient descent that helps accelerate convergence and
/// reduce oscillation in the optimization process. It achieves this by accumulating a velocity vector
/// in the direction of persistent reduction in the objective function across iterations. This approach
/// allows the optimizer to build up "momentum" in consistent directions, helping it navigate flat regions
/// more quickly and dampening oscillations in directions with high curvature. Both the learning rate and
/// momentum coefficient can be adapted during training based on the optimization performance.
/// </para>
/// <para><b>For Beginners:</b> The Momentum Optimizer is like adding a "memory" to the learning process,
/// which helps the algorithm learn faster and more effectively.
/// 
/// Imagine you're rolling a ball down a hilly landscape to find the lowest point:
/// - Standard gradient descent is like gently nudging the ball in the downhill direction at each point
/// - Momentum is like letting the ball build up speed as it rolls
/// 
/// This has several advantages:
/// - The ball can roll through small bumps and plateaus without getting stuck
/// - It builds up speed in consistent directions, moving faster toward the solution
/// - It can dampen the "zig-zagging" that happens on steep slopes
/// 
/// This class lets you configure how the ball rolls: how fast it can go (learning rate),
/// how much momentum it builds up, and how these values adjust during training based on 
/// whether progress is being made.
/// </para>
/// </remarks>
public class MomentumOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>Initializes a new instance with the Momentum optimizer defaults.</summary>
    public MomentumOptimizerOptions()
    {
        // Momentum adapts more gently than the base default (1.05 / 0.95): +2% while the loss improves,
        // -2% while it worsens. Set on the inherited properties, which the optimizer base reads.
        MomentumIncreaseFactor = 1.02;
        MomentumDecreaseFactor = 0.98;
    }

    /// <summary>Initializes a new instance as a copy of <paramref name="other"/>, with every setting copied.</summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException"><paramref name="other"/> is null.</exception>
    public MomentumOptimizerOptions(MomentumOptimizerOptions<T, TInput, TOutput> other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        BatchSize = other.BatchSize;
        LearningRateIncreaseFactor = other.LearningRateIncreaseFactor;
        LearningRateDecreaseFactor = other.LearningRateDecreaseFactor;
    }

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
    /// Gets or sets the maximum allowed learning rate for the optimization process.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines the step size in the parameter space during each update. This parameter
    /// sets an upper limit on how large the learning rate can become, even when using adaptive techniques that
    /// might otherwise increase it further. The 'new' keyword indicates this property overrides a similar
    /// property in the base class, potentially with a different default value or behavior specific to
    /// momentum-based optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how big of a step the algorithm can take
    /// in any given direction during training.
    /// 
    /// Using our rolling ball analogy:
    /// - The learning rate is like controlling how hard you can push the ball at each point
    /// - A higher rate means stronger pushes, potentially moving faster but risking overshooting
    /// - A lower rate means gentler pushes, moving more safely but potentially very slowly
    /// 
    /// The default value of 0.1 provides a reasonable balance for many problems:
    /// - High enough to make meaningful progress
    /// - Low enough to avoid wild overshooting in most scenarios
    /// 
    /// You might want to increase this value if:
    /// - Training is progressing too slowly
    /// - The optimization landscape is relatively smooth
    /// 
    /// You might want to decrease this value if:
    /// - Training is unstable or diverging
    /// - You're getting inconsistent results
    /// - Your optimization problem is particularly complex or ill-conditioned
    /// 
    /// Note: This property overrides a similar setting in the base class, which is why it has the 'new' keyword.
    /// </para>
    /// </remarks>
    public override double MaxLearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is increased when the loss is improving.
    /// </summary>
    /// <value>The learning rate increase multiplier, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how aggressively the learning rate is increased when the optimization is
    /// making progress (i.e., when the loss is decreasing). After a successful update, the current learning
    /// rate is multiplied by this factor, allowing the algorithm to take larger steps when moving in a
    /// promising direction. This adaptive approach can speed up convergence by taking larger steps when
    /// it's safe to do so, but the rate will never exceed the MaxLearningRate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm increases its step size
    /// when things are going well.
    /// 
    /// Continuing our rolling ball analogy:
    /// - When the ball is moving consistently downhill, we might want to push it harder
    /// - This setting determines how much to increase that push with each successful step
    /// 
    /// The default value of 1.05 means:
    /// - Each time the model improves, the learning rate increases by 5%
    /// - For example, a learning rate of 0.1 would become 0.105 after a successful update
    /// 
    /// This gradual increase helps the algorithm:
    /// - Speed up when it's on the right track
    /// - Cover flat regions more efficiently
    /// - Potentially escape shallow local minima
    /// 
    /// You might want to increase this value (like to 1.1) if:
    /// - Training seems too slow
    /// - You're confident the optimization landscape is well-behaved
    /// 
    /// You might want to decrease this value (like to 1.01) if:
    /// - You want more conservative adaptation
    /// - You notice training becomes unstable after periods of progress
    /// 
    /// The learning rate will never exceed the MaxLearningRate value, regardless of how many
    /// successful updates occur.
    /// </para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is decreased when the loss is getting worse.
    /// </summary>
    /// <value>The learning rate decrease multiplier, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the learning rate is reduced when the optimization encounters
    /// difficulties (i.e., when the loss increases). After an unsuccessful update, the current learning
    /// rate is multiplied by this factor, forcing the algorithm to take smaller, more cautious steps. This
    /// adaptive approach helps the algorithm recover from overshooting and navigate complex loss landscapes
    /// by automatically adjusting the step size based on the observed performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm decreases its step size
    /// when it makes a mistake.
    /// 
    /// In our rolling ball scenario:
    /// - If the ball suddenly starts rolling uphill, we've gone too far or in the wrong direction
    /// - We want to be more careful with how hard we push it in the next step
    /// - This setting determines how much more cautious we become
    /// 
    /// The default value of 0.95 means:
    /// - Each time the model gets worse, the learning rate decreases by 5%
    /// - For example, a learning rate of 0.1 would become 0.095 after an unsuccessful update
    /// 
    /// This adjustment helps the algorithm:
    /// - Recover from overshooting the optimal values
    /// - Navigate tricky, curved areas of the loss landscape
    /// - Eventually settle into a minimum
    /// 
    /// You might want to decrease this value (like to 0.8) if:
    /// - Training seems unstable
    /// - You want the algorithm to become more cautious more quickly after mistakes
    /// 
    /// You might want to increase this value (like to 0.99) if:
    /// - You want to be more persistent with the current learning rate
    /// - You're worried about getting stuck in local minima
    /// - The loss function is noisy and you don't want to overreact to small increases
    /// </para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

}
