namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Root Mean Square Propagation (RMSProp) optimizer, an adaptive learning
/// rate optimization algorithm commonly used in training neural networks.
/// </summary>
/// <remarks>
/// <para>
/// RMSProp (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm designed to 
/// address the diminishing learning rates problem of AdaGrad. Proposed by Geoffrey Hinton, RMSProp divides 
/// the learning rate for each parameter by a running average of the magnitudes of recent gradients for that 
/// parameter. Unlike AdaGrad, which accumulates all past squared gradients, RMSProp uses an exponentially 
/// decaying average, which prevents the learning rate from becoming infinitesimally small over time. This 
/// makes RMSProp particularly well-suited for non-stationary objectives and problems with noisy gradients. 
/// This class extends GradientBasedOptimizerOptions to provide specific configuration parameters for the 
/// RMSProp algorithm, including the decay rate for the moving average and a small epsilon value to prevent 
/// division by zero.
/// </para>
/// <para><b>For Beginners:</b> RMSProp is an optimization algorithm that helps neural networks learn more efficiently.
/// 
/// When training a neural network or other machine learning model:
/// - We need to adjust the model's parameters to minimize errors
/// - Different parameters may need different adjustment rates
/// - Some directions in the parameter space may need larger or smaller steps
/// 
/// RMSProp solves these problems by:
/// - Tracking the recent history of gradients (how parameters should change)
/// - Automatically adjusting the learning rate for each parameter
/// - Making larger updates for parameters with small or infrequent gradients
/// - Making smaller updates for parameters with large or frequent gradients
/// 
/// This adaptive behavior helps the model:
/// - Learn faster overall
/// - Avoid getting stuck in poor solutions
/// - Handle different types of features more effectively
/// 
/// RMSProp is particularly good for:
/// - Deep neural networks
/// - Recurrent neural networks
/// - Problems where different parameters need different learning rates
/// 
/// This class lets you configure the specific behavior of the RMSProp optimizer.
/// </para>
/// </remarks>
public class RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for RMSprop.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the decay rate for the moving average of squared gradients.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.9.</value>
    /// <remarks>
    /// <para>
    /// This property controls how quickly the moving average of squared gradients decays over time. It determines 
    /// the weight given to past squared gradients when computing the moving average. A higher value (closer to 1) 
    /// gives more weight to past gradients, resulting in a smoother but slower adaptation to changes in the gradient. 
    /// A lower value gives more weight to recent gradients, resulting in faster adaptation but potentially more 
    /// oscillation. The default value of 0.9 is commonly used in practice and provides a good balance between 
    /// stability and adaptability for most applications. Values typically range from 0.9 to 0.999, with 0.9 being 
    /// a standard choice recommended by Geoffrey Hinton, the algorithm's creator.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the optimizer remembers about past gradients.
    /// 
    /// The decay value determines:
    /// - How quickly the optimizer "forgets" older gradients
    /// - How much it focuses on recent gradient information
    /// 
    /// The default value of 0.9 means:
    /// - About 90% of the previous average is retained each update
    /// - About 10% of the new information is incorporated
    /// - This creates a weighted average that emphasizes recent history but doesn't ignore the past
    /// 
    /// Think of it like this:
    /// - Higher values (like 0.95 or 0.99): Longer memory, more stable learning, but slower to adapt to changes
    /// - Lower values (like 0.8 or 0.7): Shorter memory, quicker adaptation, but potentially more unstable
    /// 
    /// When to adjust this value:
    /// - Increase it when training is unstable or oscillating
    /// - Decrease it when the optimizer seems to learn too slowly or gets stuck
    /// 
    /// For most applications, the default value of 0.9 works well and was specifically recommended
    /// by Geoffrey Hinton, who developed the RMSProp algorithm.
    /// </para>
    /// </remarks>
    public double Decay { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets a small constant added to the denominator to improve numerical stability.
    /// </summary>
    /// <value>A small positive double value, defaulting to 1e-8 (0.00000001).</value>
    /// <remarks>
    /// <para>
    /// This property specifies a small constant value added to the denominator when scaling the learning rate 
    /// by the root mean square of recent gradients. Its primary purpose is to prevent division by zero when the 
    /// accumulated squared gradients are very small or zero. It also improves numerical stability in general by 
    /// preventing the effective learning rate from becoming excessively large when gradients are very small. 
    /// The default value of 1e-8 is small enough to have minimal impact on the optimization process while still 
    /// providing the necessary numerical stability. In most cases, this value does not need to be adjusted, but 
    /// for problems with very different gradient scales or when using very low precision arithmetic, a different 
    /// value might be appropriate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents mathematical problems when gradients become very small.
    /// 
    /// During optimization, RMSProp divides by the square root of the average squared gradient:
    /// - If this value becomes zero or very small, division could cause numerical problems
    /// - The epsilon value is added to prevent this division by zero
    /// - It's a safety measure to ensure mathematical stability
    /// 
    /// The default value of 1e-8 (0.00000001) is:
    /// - Small enough not to interfere with normal optimization
    /// - Large enough to prevent numerical instability
    /// 
    /// This is similar to how you might add a tiny amount of water to paint to prevent it from
    /// becoming completely dry and unusable - just enough to maintain workability without
    /// changing the paint's properties.
    /// 
    /// When to adjust this value:
    /// - Increase it (e.g., to 1e-7 or 1e-6) if you encounter "NaN" (Not a Number) errors during training
    /// - Decrease it (e.g., to 1e-10) if you're using very high precision and want to minimize its effect
    /// 
    /// For most users, this is an advanced setting that can be left at its default value.
    /// </para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;
}
