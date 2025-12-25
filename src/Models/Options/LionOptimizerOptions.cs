namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Lion (Evolved Sign Momentum) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Lion is a modern optimization algorithm discovered through symbolic program search that offers
/// significant advantages over Adam, including 50% memory reduction and superior performance on large models.
/// Unlike Adam which maintains both momentum and variance, Lion uses only a single momentum state and
/// relies on sign-based updates for improved efficiency and generalization.
/// </para>
/// <para><b>For Beginners:</b> Think of Lion as a streamlined version of Adam that focuses on the direction
/// of learning (not the magnitude). It's like a compass that only tells you which way to go, making decisions
/// faster and using less memory. Lion is particularly effective for training large neural networks and transformers,
/// where it can achieve better results than Adam while using half the memory.</para>
/// </remarks>
public class LionOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for Lion.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the initial learning rate for the Lion optimizer.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big each step is during training.
    /// Lion typically uses a smaller learning rate (0.0001) compared to Adam (0.001) because sign-based
    /// updates provide more consistent step sizes. Think of it like setting the speed of your car -
    /// Lion moves more carefully but more reliably. You may need to tune this based on your problem,
    /// but 0.0001 (or 1e-4) is a good starting point for most applications.</para>
    /// </remarks>
    public new double InitialLearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the exponential decay rate for the momentum interpolation (used for computing the update).
    /// </summary>
    /// <value>The beta1 value, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta1 controls how much Lion blends the current gradient with past momentum
    /// when deciding which direction to move. A value of 0.9 means it gives 90% weight to the past momentum
    /// and 10% to the new gradient. This is like having inertia - you don't change direction immediately when
    /// you get new information. Higher values (closer to 1) create smoother updates but slower adaptation,
    /// while lower values respond more quickly to new gradients.</para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for updating the momentum state.
    /// </summary>
    /// <value>The beta2 value, defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta2 controls how much Lion remembers from its momentum history when
    /// updating the momentum state for the next iteration. A value of 0.99 means it retains 99% of the old
    /// momentum and incorporates 1% from the new gradient. This creates a long memory of past gradients,
    /// helping smooth out noisy updates. Think of it like a heavy flywheel that doesn't change speed quickly -
    /// it provides stability during training.</para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the weight decay (L2 regularization) coefficient.
    /// </summary>
    /// <value>The weight decay value, defaulting to 0.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Weight decay helps prevent overfitting by penalizing large parameter values.
    /// A value of 0.0 means no weight decay. When set to a small positive value (e.g., 0.01 or 0.1), it encourages
    /// the model to keep weights small, which often improves generalization to new data. Think of it like a tax
    /// on complexity - it encourages the model to be as simple as possible while still solving the problem.
    /// Lion applies weight decay in a decoupled manner, similar to AdamW.</para>
    /// </remarks>
    public double WeightDecay { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to automatically adjust Beta1 during training.
    /// </summary>
    /// <value>False by default, as Lion typically uses fixed betas.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, Beta1 will be automatically adjusted based on training progress.
    /// However, Lion was designed to work well with fixed beta values, so this is disabled by default.
    /// Unlike Adam, Lion is less sensitive to beta parameter choices due to its sign-based updates.
    /// You typically don't need to enable this unless you're doing advanced experimentation.</para>
    /// </remarks>
    public bool UseAdaptiveBeta1 { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to automatically adjust Beta2 during training.
    /// </summary>
    /// <value>False by default, as Lion typically uses fixed betas.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, Beta2 will be automatically adjusted based on training progress.
    /// However, Lion was designed to work well with fixed beta values, so this is disabled by default.
    /// The sign-based nature of Lion makes it robust to beta parameter variations.</para>
    /// </remarks>
    public bool UseAdaptiveBeta2 { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum allowed value for Beta1.
    /// </summary>
    /// <value>The minimum Beta1 value, defaulting to 0.85.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If adaptive Beta1 is enabled, this prevents it from dropping too low.
    /// A minimum of 0.85 ensures some momentum is always maintained.</para>
    /// </remarks>
    public double MinBeta1 { get; set; } = 0.85;

    /// <summary>
    /// Gets or sets the maximum allowed value for Beta1.
    /// </summary>
    /// <value>The maximum Beta1 value, defaulting to 0.95.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If adaptive Beta1 is enabled, this prevents it from becoming too high.
    /// A maximum of 0.95 ensures some responsiveness to new gradients.</para>
    /// </remarks>
    public double MaxBeta1 { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed value for Beta2.
    /// </summary>
    /// <value>The minimum Beta2 value, defaulting to 0.95.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If adaptive Beta2 is enabled, this prevents it from dropping too low.
    /// A minimum of 0.95 ensures momentum state retains sufficient history.</para>
    /// </remarks>
    public double MinBeta2 { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the maximum allowed value for Beta2.
    /// </summary>
    /// <value>The maximum Beta2 value, defaulting to 0.999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If adaptive Beta2 is enabled, this prevents it from becoming too high.
    /// A maximum of 0.999 ensures the momentum state can still adapt to changes.</para>
    /// </remarks>
    public double MaxBeta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets the factor by which Beta1 is increased when fitness improves.
    /// </summary>
    /// <value>The Beta1 increase factor, defaulting to 1.02.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When adaptive Beta1 is enabled and the optimizer is improving,
    /// Beta1 is multiplied by this factor. A value of 1.02 means Beta1 increases by 2% each time
    /// fitness improves. Higher Beta1 values create smoother, more stable updates.</para>
    /// </remarks>
    public double Beta1IncreaseFactor { get; set; } = 1.02;

    /// <summary>
    /// Gets or sets the factor by which Beta1 is decreased when fitness does not improve.
    /// </summary>
    /// <value>The Beta1 decrease factor, defaulting to 0.98.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When adaptive Beta1 is enabled and the optimizer is not improving,
    /// Beta1 is multiplied by this factor. A value of 0.98 means Beta1 decreases by 2% each time
    /// fitness doesn't improve. Lower Beta1 values make the optimizer more responsive to new gradients.</para>
    /// </remarks>
    public double Beta1DecreaseFactor { get; set; } = 0.98;

    /// <summary>
    /// Gets or sets the factor by which Beta2 is increased when fitness improves.
    /// </summary>
    /// <value>The Beta2 increase factor, defaulting to 1.02.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When adaptive Beta2 is enabled and the optimizer is improving,
    /// Beta2 is multiplied by this factor. A value of 1.02 means Beta2 increases by 2% each time
    /// fitness improves. Higher Beta2 values create longer memory of past gradients for more stability.</para>
    /// </remarks>
    public double Beta2IncreaseFactor { get; set; } = 1.02;

    /// <summary>
    /// Gets or sets the factor by which Beta2 is decreased when fitness does not improve.
    /// </summary>
    /// <value>The Beta2 decrease factor, defaulting to 0.98.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When adaptive Beta2 is enabled and the optimizer is not improving,
    /// Beta2 is multiplied by this factor. A value of 0.98 means Beta2 decreases by 2% each time
    /// fitness doesn't improve. Lower Beta2 values make the momentum state more responsive to recent changes.</para>
    /// </remarks>
    public double Beta2DecreaseFactor { get; set; } = 0.98;
}
