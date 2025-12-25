namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Adagrad optimization algorithm, which adapts the learning rate for each parameter based on historical gradient information.
/// </summary>
/// <remarks>
/// <para>
/// Adagrad is an optimization algorithm that automatically adjusts the learning rate for each parameter based on how frequently it is updated.
/// Parameters that are updated more frequently receive smaller learning rates, while parameters that are updated less frequently receive larger learning rates.
/// </para>
/// <para><b>For Beginners:</b> Adagrad is like a smart teacher that gives more attention to students who need it most.
/// If a parameter (think of it as a knob that needs adjusting) hasn't changed much in the past, Adagrad will make bigger adjustments to it.
/// If another parameter has already been adjusted a lot, Adagrad will make smaller changes to it.
/// This helps the model learn more efficiently, especially when some parameters need more tuning than others.
/// </para>
/// </remarks>
public class AdagradOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for Adagrad.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets a small constant added to the denominator to prevent division by zero.
    /// </summary>
    /// <value>The epsilon value, defaulting to 0.00000001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon is a tiny safety value that prevents the algorithm from crashing
    /// when it would otherwise divide by zero. It's like having training wheels that only activate when needed.
    /// You typically don't need to change this unless you're experiencing numerical stability issues.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the initial learning rate for the Adagrad optimizer, overriding the base class value.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The initial learning rate controls how large the first steps are in the optimization process.
    /// For Adagrad, a value of 0.01 is typically effective as the algorithm will automatically adjust
    /// the learning rate for each parameter during training.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the initial learning rate as setting how big your first steps are
    /// when starting a journey. With Adagrad, you can start with slightly larger steps (0.01) than some other
    /// algorithms because Adagrad has a built-in mechanism that will automatically slow down the steps for
    /// parameters that are updated frequently. It's like having a smart walking assistant that helps you
    /// take appropriate steps based on the terrain - bigger steps on flat ground, smaller steps on rocky terrain.</para>
    /// </remarks>
    public new double InitialLearningRate { get; set; } = 0.01;

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
}
