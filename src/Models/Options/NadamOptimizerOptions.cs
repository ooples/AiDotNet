namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Nadam optimizer, which combines Nesterov momentum with Adam's
/// adaptive learning rates for efficient training of neural networks and other gradient-based models.
/// </summary>
/// <remarks>
/// <para>
/// Nadam (Nesterov-accelerated Adaptive Moment Estimation) is an optimization algorithm that extends
/// Adam by incorporating Nesterov momentum. Like Adam, it maintains adaptive learning rates for each
/// parameter based on estimates of first and second moments of the gradients. Additionally, it applies
/// the Nesterov acceleration technique, which evaluates the gradient at a "look-ahead" position rather
/// than the current position. This combination often leads to faster convergence than standard Adam,
/// particularly for problems with complex loss landscapes or sparse gradients.
/// </para>
/// <para><b>For Beginners:</b> Nadam is an advanced optimization algorithm that helps neural networks
/// and other machine learning models learn more efficiently.
/// 
/// Imagine you're trying to navigate to the lowest point in a hilly landscape while blindfolded:
/// - Standard gradient descent is like taking steps directly downhill from where you're standing
/// - Adam adds adaptive step sizes (taking bigger steps in flat areas, smaller steps in steep areas)
/// - Nadam goes a step further by trying to predict where you'll be after your next step and looking
///   at the downhill direction from that predicted position
/// 
/// This combination of techniques helps the algorithm:
/// - Learn faster than simpler methods
/// - Avoid getting stuck in small dips that aren't the true lowest point
/// - Adapt to different parts of the learning process with appropriate step sizes
/// 
/// This class lets you fine-tune how Nadam works: how quickly it learns, how much it relies on past
/// information, and how it adapts its learning rate during training.
/// </para>
/// </remarks>
public class NadamOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for Nadam.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the initial learning rate that controls the step size in parameter updates.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 0.002.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines the magnitude of parameter updates during optimization. In Nadam,
    /// this serves as the initial step size, which is then adjusted by the adaptive moment estimation
    /// mechanism based on the historical gradient information. The default value of 0.002 is typically
    /// suitable for Nadam, slightly higher than Adam's common default of 0.001, reflecting the algorithm's
    /// often improved efficiency. The actual step sizes used during training will vary by parameter and
    /// iteration as determined by the adaptive nature of the algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how big of a step the algorithm takes
    /// when updating the model's parameters.
    /// 
    /// Think of it like adjusting the speed on a treadmill:
    /// - A higher learning rate (like 0.01) means taking bigger steps, potentially moving faster toward the solution
    /// - A lower learning rate (like 0.0001) means taking smaller, more cautious steps
    /// 
    /// The default value of 0.002 is a moderate setting that works well for many problems:
    /// - Fast enough to make good progress
    /// - Cautious enough to avoid overshooting the solution
    /// 
    /// You might want to increase this value if:
    /// - Training is progressing too slowly
    /// - You have a tight time budget
    /// - Your loss function is relatively smooth
    /// 
    /// You might want to decrease this value if:
    /// - Training is unstable (loss fluctuating wildly)
    /// - You're getting poor final results
    /// - You're fine-tuning a model that's already close to optimal
    /// 
    /// Note that Nadam will adapt this learning rate differently for each parameter during training,
    /// but this initial value still significantly influences the overall training dynamics.
    /// </para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.002;

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates (momentum).
    /// </summary>
    /// <value>The first moment decay rate, defaulting to 0.9.</value>
    /// <remarks>
    /// <para>
    /// Beta1 controls the exponential decay rate for the first moment estimates, effectively determining
    /// how much the algorithm relies on recent versus older gradients when computing momentum. Values closer
    /// to 1.0 give more weight to past gradients, creating more momentum and smoothing out updates. The value
    /// of 0.9 means that approximately the last 10 iterations have significant influence on the current update.
    /// This helps the optimizer navigate through noisy gradients and saddle points more effectively.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm relies on recent
    /// versus older gradient information when determining the direction to move.
    /// 
    /// Imagine you're calculating a running average of recent temperatures:
    /// - Beta1 determines how much weight you give to yesterday's average versus today's temperature
    /// - A higher value (closer to 1) means the average changes more slowly, incorporating more history
    /// - A lower value means the average responds more quickly to recent changes
    /// 
    /// The default value of 0.9 means:
    /// - The algorithm keeps about 90% of its previous momentum
    /// - And adds about 10% of the new information in each step
    /// - This creates a smoothing effect that helps ignore small random fluctuations
    /// 
    /// You might want to increase this value (like to 0.95) if:
    /// - Your gradients are noisy (inconsistent between batches)
    /// - You want more stable, consistent progress
    /// 
    /// You might want to decrease this value (like to 0.8) if:
    /// - You want the algorithm to respond more quickly to recent gradients
    /// - Your loss landscape has sharp turns that require quick adaptation
    /// 
    /// This parameter helps balance between making steady progress in a consistent direction and
    /// being responsive to new information.
    /// </para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for the second moment estimates (adaptive learning rates).
    /// </summary>
    /// <value>The second moment decay rate, defaulting to 0.999.</value>
    /// <remarks>
    /// <para>
    /// Beta2 controls the exponential decay rate for the second moment estimates, which track the squared
    /// magnitudes of recent gradients. These estimates help adapt the learning rate for each parameter based
    /// on the historical variability of its gradients. Values closer to 1.0 create a longer "memory" of past
    /// squared gradients. The value of 0.999 means that approximately the last 1000 iterations have significant
    /// influence on the learning rate adaptation. This allows the algorithm to take smaller steps for parameters
    /// with historically large or highly variable gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how the algorithm adapts different learning
    /// rates for each parameter based on their historical gradient patterns.
    /// 
    /// While Beta1 tracks the direction to move, Beta2 tracks how variable or uncertain that direction has been:
    /// - It keeps a running average of the squared gradient values
    /// - Parameters with consistently large gradients get smaller step sizes
    /// - Parameters with small or infrequent gradients get larger step sizes
    /// 
    /// The default value of 0.999 means:
    /// - The algorithm maintains approximately 99.9% of its previous estimate of gradient variability
    /// - And adds about 0.1% of new information in each step
    /// - This creates a very long-term memory of which parameters have been volatile
    /// 
    /// You might want to decrease this value (like to 0.99) if:
    /// - You want the algorithm to adapt more quickly to recent gradient patterns
    /// - Training is progressing through distinct phases that require different adaptation
    /// 
    /// Most users won't need to change this parameter, as the default value works well across a wide
    /// range of problems. It's typically less sensitive than Beta1 and primarily affects the algorithm's
    /// adaptive behavior for different parameters rather than its overall direction.
    /// </para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets a small constant added to the denominator to improve numerical stability.
    /// </summary>
    /// <value>The numerical stability constant, defaulting to 0.00000001 (1e-8).</value>
    /// <remarks>
    /// <para>
    /// Epsilon is a small positive value added to the denominator when computing adaptive learning rates
    /// to prevent division by zero and improve numerical stability. It ensures that even parameters with
    /// very small or zero second moment estimates still receive finite updates. The value should be small
    /// enough not to interfere with the normal adaptation process but large enough to prevent numerical
    /// underflow or instability in floating-point operations.
    /// </para>
    /// <para><b>For Beginners:</b> This setting is a small safety value that prevents numerical
    /// problems when the algorithm's calculated values get extremely small.
    /// 
    /// Think of it like adding a tiny amount of friction to a wheel:
    /// - It prevents the wheel from spinning infinitely fast if there's no resistance
    /// - In the algorithm, it prevents certain mathematical calculations from becoming unstable
    /// 
    /// The default value of 0.00000001 (1e-8) is extremely small, so it typically:
    /// - Only affects parameters that have seen very few or very small gradient updates
    /// - Prevents potential division-by-zero errors in the algorithm's calculations
    /// 
    /// Most users will never need to change this value. It's a mathematical safeguard rather than
    /// a tuning parameter that affects the algorithm's learning behavior. If you do need to adjust it:
    /// 
    /// - Decrease it (like to 1e-10) if you're using very high-precision calculations and find the
    ///   default value is interfering with proper convergence for some parameters
    /// 
    /// - Increase it (like to 1e-6) if you encounter numerical stability issues (NaN or Inf values)
    ///   during training
    /// </para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is increased when the loss is improving.
    /// </summary>
    /// <value>The learning rate increase multiplier, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how aggressively the base learning rate is increased when the optimization is
    /// making progress (i.e., when the loss is decreasing). After a successful update, the current learning
    /// rate is multiplied by this factor, allowing the algorithm to take larger steps when moving in a
    /// promising direction. This adaptive approach can speed up convergence, but the rate will never exceed
    /// the MaxLearningRate. Note that this adjustment is separate from the per-parameter adaptation that
    /// Nadam performs and affects the global learning rate scale.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm increases its
    /// global step size when things are going well.
    /// 
    /// Imagine you're walking downhill trying to reach the lowest point:
    /// - When you're making good progress, you might want to walk faster
    /// - This setting determines how much faster you go with each successful step
    /// 
    /// The default value of 1.05 means:
    /// - Each time the model improves, the base learning rate increases by 5%
    /// - For example, a learning rate of 0.002 would become 0.0021 after a successful update
    /// 
    /// This gradual increase helps the algorithm:
    /// - Speed up when it's on the right track
    /// - Cover flat regions more efficiently
    /// - Potentially escape shallow local minima
    /// 
    /// You might want to increase this value (like to 1.1) if:
    /// - Training seems too slow
    /// - Your optimization landscape has large flat regions
    /// 
    /// You might want to decrease this value (like to 1.01) if:
    /// - You want more conservative adaptation
    /// - You notice training becomes unstable after periods of progress
    /// 
    /// Note that this differs from Nadam's normal adaptation mechanism - this adjusts the global
    /// learning rate scale, while Nadam's built-in adaptation adjusts rates differently for each parameter.
    /// </para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is decreased when the loss is getting worse.
    /// </summary>
    /// <value>The learning rate decrease multiplier, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the base learning rate is reduced when the optimization encounters
    /// difficulties (i.e., when the loss increases). After an unsuccessful update, the current learning
    /// rate is multiplied by this factor, forcing the algorithm to take smaller, more cautious steps. This
    /// adaptive approach helps the algorithm recover from overshooting and navigate complex loss landscapes
    /// by automatically adjusting the global step size based on the observed performance. The rate will never
    /// fall below MinLearningRate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm decreases its
    /// global step size when it makes a mistake.
    /// 
    /// Continuing the walking downhill analogy:
    /// - If you take a step and end up higher than before, you've gone in the wrong direction
    /// - You'd want to be more careful with your next step
    /// - This setting determines how much more cautious you become
    /// 
    /// The default value of 0.95 means:
    /// - Each time the model gets worse, the learning rate decreases by 5%
    /// - For example, a learning rate of 0.002 would become 0.0019 after an unsuccessful update
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
    /// 
    /// Finding the right balance between increasing and decreasing the learning rate is important
    /// for efficient training.
    /// </para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed value for the learning rate.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 0.00001 (1e-5).</value>
    /// <remarks>
    /// <para>
    /// This parameter establishes a lower bound for the learning rate to prevent it from becoming
    /// too small after multiple decreases. If repeated unsuccessful iterations would decrease the learning
    /// rate below this value, it will be clamped to this minimum. This prevents the training from
    /// effectively stopping due to extremely small steps. The 'new' keyword indicates this property
    /// overrides a similar property in the base class, potentially with a different default value.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the learning rate from becoming
    /// too small, even after many unsuccessful steps.
    /// 
    /// Imagine adjusting the volume on a radio:
    /// - The learning rate is like the volume control
    /// - After turning it down several times, you don't want it to become inaudible
    /// - This setting establishes the minimum volume level
    /// 
    /// The default value of 0.00001 (1e-5) means:
    /// - The learning rate will never go below this value
    /// - This ensures the algorithm keeps making at least some progress
    /// - Without this limit, the learning rate could become effectively zero
    /// 
    /// You might want to increase this value (like to 1e-4) if:
    /// - You notice training slows to a crawl after encountering difficulties
    /// - You want to ensure the algorithm maintains a minimum level of exploration
    /// 
    /// You might want to decrease this value (like to 1e-6) if:
    /// - You want to allow for very fine-grained adjustments near the end of training
    /// - Your problem requires extremely precise parameter values
    /// 
    /// Note: This property overrides a similar setting in the base class, which is why it has the 'new' keyword.
    /// </para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the maximum allowed value for the learning rate.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter establishes an upper bound for the learning rate to prevent it from becoming
    /// too large after multiple increases. If repeated successful iterations would increase the learning
    /// rate above this value, it will be clamped to this maximum. This helps maintain stability by
    /// preventing excessively large steps that might cause the optimization to diverge. The 'new' keyword
    /// indicates this property overrides a similar property in the base class.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the learning rate from becoming
    /// too large, even after many successful steps.
    /// 
    /// Continuing with the volume analogy:
    /// - After turning up the volume several times, you don't want it to become painfully loud
    /// - This setting establishes the maximum volume level
    /// 
    /// The default value of 0.1 means:
    /// - The learning rate will never go above this value
    /// - This prevents the algorithm from taking wildly large steps
    /// - Without this limit, the learning rate could grow uncontrollably
    /// 
    /// You might want to increase this value (like to 0.5) if:
    /// - Your problem seems to benefit from occasionally large updates
    /// - You're confident in the stability of your loss function
    /// 
    /// You might want to decrease this value (like to 0.05) if:
    /// - Training tends to become unstable
    /// - You're working with a particularly sensitive model
    /// - Your loss function has steep cliffs or sharp curves
    /// 
    /// Finding the right maximum learning rate helps prevent the optimizer from "jumping out"
    /// of good solutions due to taking steps that are too large.
    /// 
    /// Note: This property overrides a similar setting in the base class, which is why it has the 'new' keyword.
    /// </para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 0.1;
}
