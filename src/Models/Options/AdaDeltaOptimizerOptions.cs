namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the AdaDelta optimization algorithm, which is an extension of AdaGrad that adapts learning rates based on a moving window of gradient updates.
/// </summary>
/// <remarks>
/// <para>
/// AdaDelta is an optimization algorithm that dynamically adapts the learning rate for each parameter based on historical gradient information.
/// It addresses some limitations of earlier algorithms by using a moving average of squared gradients.
/// </para>
/// <para><b>For Beginners:</b> AdaDelta is like a smart learning system that automatically adjusts how quickly it learns based on past experience.
/// Instead of using a fixed learning speed, it slows down for parameters that change a lot and speeds up for those that change little.
/// This helps the model learn more efficiently without requiring manual tuning of the learning rate.
/// </para>
/// </remarks>
public class AdaDeltaOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for AdaDelta.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the decay rate for the moving average of squared gradients.
    /// </summary>
    /// <value>The decay rate, defaulting to 0.95.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Rho controls how much the algorithm "remembers" about past gradients.
    /// A value of 0.95 means it gives high importance (95%) to past information and only 5% to new information.
    /// Higher values (closer to 1) make learning more stable but slower to adapt to changes.
    /// Think of it like averaging your test scores, but giving more weight to older scores than newer ones.</para>
    /// </remarks>
    public double Rho { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets a small constant added to denominators to prevent division by zero.
    /// </summary>
    /// <value>The epsilon value, defaulting to 0.000001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon is a tiny safety value that prevents the algorithm from crashing
    /// when it would otherwise divide by zero. It's like having a small backup plan that kicks in only when needed.
    /// You typically don't need to change this unless you're experiencing numerical stability issues.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets whether to automatically adjust the Rho parameter during training.
    /// </summary>
    /// <value>True to use adaptive Rho (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the algorithm will automatically adjust how much it relies on past information
    /// based on how well it's performing. If the model is improving, it will trust its memory more.
    /// If performance worsens, it will pay more attention to new information. This helps the algorithm adapt to different phases of learning.</para>
    /// </remarks>
    public bool UseAdaptiveRho { get; set; } = true;

    /// <summary>
    /// Gets or sets the factor by which Rho increases when performance improves.
    /// </summary>
    /// <value>The Rho increase factor, defaulting to 1.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is improving, Rho will be increased by this factor.
    /// A value of 1.01 means Rho becomes 101% of its previous value, making the algorithm rely slightly more
    /// on past information when things are going well. This helps stabilize learning when on the right track.</para>
    /// </remarks>
    public double RhoIncreaseFactor { get; set; } = 1.01;

    /// <summary>
    /// Gets or sets the factor by which Rho decreases when performance worsens.
    /// </summary>
    /// <value>The Rho decrease factor, defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is getting worse, Rho will be decreased by this factor.
    /// A value of 0.99 means Rho becomes 99% of its previous value, making the algorithm pay more attention
    /// to new information when things aren't going well. This helps the model adapt more quickly when it needs to change course.</para>
    /// </remarks>
    public double RhoDecreaseFactor { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the minimum allowed value for Rho.
    /// </summary>
    /// <value>The minimum Rho value, defaulting to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents Rho from becoming too small, which would make the algorithm
    /// ignore past information too much. Even if Rho keeps decreasing, it won't go below this value.
    /// A minimum of 0.5 ensures the algorithm always considers at least some historical information.</para>
    /// </remarks>
    public double MinRho { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum allowed value for Rho.
    /// </summary>
    /// <value>The maximum Rho value, defaulting to 0.9999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents Rho from becoming too large, which would make the algorithm
    /// rely too heavily on past information and adapt too slowly. Even if Rho keeps increasing, it won't go above this value.
    /// A maximum of 0.9999 ensures the algorithm always incorporates at least some new information.</para>
    /// </remarks>
    public double MaxRho { get; set; } = 0.9999;

    /// <summary>
    /// Gets or sets the initial learning rate for the AdaDelta optimizer, overriding the base class value.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// While AdaDelta is designed to eliminate the need for manually setting a learning rate,
    /// this parameter serves as a scaling factor for the updates. The default value of 1.0 works well
    /// in most cases since AdaDelta automatically adapts the effective learning rate during training.
    /// </para>
    /// <para><b>For Beginners:</b> Unlike other optimization algorithms where the learning rate directly controls
    /// how big each learning step is, in AdaDelta this value is more like an initial scaling factor. 
    /// Think of it as setting the overall speed limit rather than controlling each individual step.
    /// The default value of 1.0 is higher than in other algorithms because AdaDelta has built-in mechanisms
    /// that automatically adjust how it learns, making it less sensitive to this initial setting.
    /// In most cases, you won't need to change this value.</para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 1.0;
}
