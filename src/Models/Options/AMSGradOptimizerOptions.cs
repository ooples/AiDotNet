namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the AMSGrad optimization algorithm, which is an improved variant of the Adam optimizer
/// that addresses potential convergence issues by maintaining the maximum of past squared gradients.
/// </summary>
/// <remarks>
/// <para>
/// AMSGrad is an adaptive learning rate optimization algorithm that combines the benefits of AdaGrad and RMSProp
/// while ensuring convergence by using a non-decreasing learning rate adjustment. It's particularly effective for
/// deep learning models and non-convex optimization problems.
/// </para>
/// <para><b>For Beginners:</b> AMSGrad is like a smart running coach that adjusts your training pace based on your
/// past performance. It remembers how difficult different parts of your training have been and adjusts accordingly,
/// making sure you don't slow down too much on challenging sections. This helps your AI model learn more efficiently
/// by giving more attention to important patterns and less to noise in the data. Unlike some other methods, AMSGrad
/// ensures that your learning progress never goes backward, which helps it reach better solutions.</para>
/// </remarks>
public class AMSGradOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the initial step size used for parameter updates during optimization.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// The learning rate controls how large each optimization step should be. Higher values can lead to faster convergence
    /// but may cause overshooting or instability, while lower values provide more stable but slower learning.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the learning rate as how big of steps your AI takes when learning.
    /// A small value (like the default 0.001) means taking small, cautious steps - the model learns slowly but steadily.
    /// A larger value means taking bigger steps - learning might be faster, but the model might step too far and miss
    /// the best solution. The default value is generally a good starting point for most problems.</para>
    /// </remarks>
    public new double InitialLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates (momentum).
    /// </summary>
    /// <value>The first moment decay rate, defaulting to 0.9.</value>
    /// <remarks>
    /// <para>
    /// Beta1 controls how much the algorithm relies on the gradient from the current iteration versus gradients from previous iterations.
    /// Values closer to 1.0 give more weight to past gradients, creating a stronger momentum effect.
    /// </para>
    /// <para><b>For Beginners:</b> Beta1 is like momentum when you're running - it determines how much your
    /// previous direction influences your current one. The default value of 0.9 means the algorithm considers
    /// about 90% of its previous direction and 10% of the new information when deciding which way to go.
    /// This helps the model move smoothly past small bumps in the learning landscape rather than zigzagging.</para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for the second moment estimates (adaptive learning rates).
    /// </summary>
    /// <value>The second moment decay rate, defaulting to 0.999.</value>
    /// <remarks>
    /// <para>
    /// Beta2 controls how quickly the algorithm adapts the learning rate for each parameter based on historical gradient magnitudes.
    /// Values closer to 1.0 result in slower adaptation but more stable learning rates.
    /// </para>
    /// <para><b>For Beginners:</b> Beta2 determines how quickly the algorithm adjusts to the difficulty of different parts
    /// of the learning process. The default value of 0.999 means it takes a long-term view, considering almost all past
    /// experience when deciding how to adjust the learning rate for each parameter. This creates stability and prevents
    /// overreacting to temporary difficulties in the learning process.</para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets a small constant added to denominators to improve numerical stability.
    /// </summary>
    /// <value>The epsilon value, defaulting to 1e-8 (0.00000001).</value>
    /// <remarks>
    /// <para>
    /// Epsilon prevents division by zero and reduces the impact of very small gradients, which could otherwise cause
    /// excessive parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> Epsilon is like a safety net that prevents mathematical errors when calculations
    /// get very small. It's a tiny value (0.00000001) that gets added to certain calculations to make sure the algorithm
    /// doesn't try to divide by zero or make other mathematical mistakes when working with very small numbers.
    /// You typically don't need to change this value unless you're experiencing numerical stability issues.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the factor by which to increase the learning rate when the loss is consistently decreasing.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// When the optimization is making consistent progress, the learning rate can be increased by this factor
    /// to speed up convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This is like speeding up when you're on a straight, clear path.
    /// The default value of 1.05 means the algorithm will increase the learning rate by 5% when things are going well.
    /// This helps the model learn faster during periods when the path to the solution is clear and direct.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which to decrease the learning rate when the loss is increasing or oscillating.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// When the optimization encounters difficulties or starts to diverge, the learning rate can be decreased by this factor
    /// to stabilize the process.
    /// </para>
    /// <para><b>For Beginners:</b> This is like slowing down when the path becomes tricky or unclear.
    /// The default value of 0.95 means the algorithm will reduce the learning rate by 5% when progress becomes difficult.
    /// This helps the model navigate challenging parts of the learning landscape without overshooting or getting stuck.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate during adaptive adjustments.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 1e-5 (0.00001).</value>
    /// <remarks>
    /// <para>
    /// This prevents the learning rate from becoming too small during adaptive adjustments, which could cause
    /// the optimization to stall.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a floor for how slow the learning can go.
    /// Even if the algorithm wants to be extra cautious, it won't reduce the step size below this value (0.00001 by default).
    /// This ensures that your model keeps making meaningful progress and doesn't get stuck taking infinitesimally small steps.</para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate during adaptive adjustments.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This prevents the learning rate from becoming too large during adaptive adjustments, which could cause
    /// the optimization to become unstable or diverge.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a ceiling for how fast the learning can go.
    /// Even when progress is smooth and the algorithm wants to speed up, it won't increase the step size above this value (0.1 by default).
    /// This prevents your model from taking such large steps that it overshoots the optimal solution or becomes unstable.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 0.1;
}
