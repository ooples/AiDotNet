namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the AdaMax optimization algorithm, a variant of Adam that uses the infinity norm.
/// </summary>
/// <remarks>
/// <para>
/// AdaMax is a variant of the Adam optimizer that uses the infinity norm instead of the L2 norm in the update rule.
/// This can make it more stable for certain types of problems, especially those with sparse gradients.
/// </para>
/// <para><b>For Beginners:</b> AdaMax is like a specialized version of a popular learning algorithm (Adam) that's
/// particularly good at handling situations where most values are zero with occasional large values.
/// Think of it as a specialized tool that works better than general-purpose tools for certain specific tasks.
/// </para>
/// </remarks>
public class AdaMaxOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the learning rate, which controls how quickly the model adapts to the problem.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.002.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate is like the size of steps you take when searching for something.
    /// A larger value (like 0.1) means taking bigger steps, which can help you find the solution faster but might cause you to step over it.
    /// A smaller value (like 0.001) means taking smaller steps, which is slower but more precise.
    /// The default of 0.002 is a good balance for most problems when using AdaMax.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.002;

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates.
    /// </summary>
    /// <value>The beta1 value, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta1 controls how much the algorithm remembers about the direction it was moving in previous steps.
    /// A value of 0.9 means it gives about 90% importance to past directions and 10% to the new direction.
    /// Think of it like steering a boat - you don't want to change direction completely with every small wave (which would make for a zigzag path),
    /// but rather maintain your general course while making small adjustments.</para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for the infinity norm of past gradients.
    /// </summary>
    /// <value>The beta2 value, defaulting to 0.999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta2 controls how much the algorithm remembers about the size of past steps.
    /// A value of 0.999 means it has a very long memory for step sizes.
    /// This helps the algorithm adapt to different parts of the learning process - taking appropriately sized steps
    /// whether it's making big initial adjustments or fine-tuning at the end.
    /// Think of it like remembering how difficult different parts of a hiking trail were, so you can pace yourself appropriately.</para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets a small constant added to denominators to prevent division by zero.
    /// </summary>
    /// <value>The epsilon value, defaulting to 0.00000001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon is a tiny safety value that prevents the algorithm from crashing
    /// when it would otherwise divide by zero. It's like having training wheels that only activate when needed.
    /// You typically don't need to change this unless you're experiencing numerical stability issues.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the factor by which the learning rate increases when performance improves.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is improving, the learning rate will be increased by this factor.
    /// A value of 1.05 means the learning rate becomes 105% of its previous value, allowing the model to learn faster
    /// when it's on the right track. This is like increasing your pace when you're heading in the right direction.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which the learning rate decreases when performance worsens.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is getting worse, the learning rate will be decreased by this factor.
    /// A value of 0.95 means the learning rate becomes 95% of its previous value, causing the model to take smaller steps
    /// when it might be heading in the wrong direction. This is like slowing down when you're unsure of your path.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate, overriding the base class value with a value optimized for AdaMax.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 0.00001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets a floor for how slow the learning can get. Even if the algorithm
    /// wants to reduce the learning rate further, it won't go below this value. For AdaMax, we use a smaller
    /// minimum value (0.00001) than the base optimizer because AdaMax can benefit from very fine adjustments
    /// in certain situations. Think of it as ensuring the algorithm never slows down too much to make progress.</para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate, overriding the base class value with a value optimized for AdaMax.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets a ceiling for how fast the learning can get. Even if the algorithm
    /// wants to increase the learning rate further, it won't go above this value. For AdaMax, we cap it at 0.1
    /// to prevent the algorithm from taking steps that are too large, which could cause instability.
    /// Think of it as putting a speed limit to prevent the algorithm from "overshooting" the optimal solution.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 0.1;
}
