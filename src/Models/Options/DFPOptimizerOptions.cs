namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Davidon-Fletcher-Powell (DFP) optimization algorithm, which is a quasi-Newton method
/// used for finding local minima of functions.
/// </summary>
/// <remarks>
/// <para>
/// The DFP algorithm is a second-order optimization method that approximates the inverse Hessian matrix
/// to accelerate convergence compared to first-order methods like gradient descent. It's particularly
/// effective for optimizing functions with complex curvature.
/// </para>
/// <para><b>For Beginners:</b> Think of the DFP optimizer as a smart navigation system for your AI model.
/// While basic optimizers (like gradient descent) only look at the current slope to decide where to go next,
/// DFP remembers information about previous steps to make better decisions. It's like the difference between
/// a hiker who only looks at the steepness right in front of them versus one who uses a map and compass
/// to plan a more efficient route to the top of the mountain. This typically means your model can learn
/// faster and more accurately, especially for complex problems.</para>
/// </remarks>
public class DFPOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for gradient computation.
    /// </summary>
    /// <value>A positive integer for mini-batch size, or -1 for full batch (default).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples are used to calculate gradients.
    /// DFP traditionally uses full-batch gradients (batch size -1) because it builds an approximation of
    /// the inverse Hessian matrix that requires consistent gradients between iterations.
    /// Using mini-batches would introduce noise that disrupts the Hessian approximation.</para>
    /// </remarks>
    public int BatchSize { get; set; } = -1;

    /// <summary>
    /// Gets or sets the initial learning rate for the optimization process.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The learning rate controls the step size during optimization. A higher value means larger steps,
    /// which can speed up convergence but may cause overshooting. This value overrides the base class setting.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate is like the size of steps your AI takes when learning.
    /// With the default value of 1.0, your model takes normal-sized steps. If learning is too slow, you might
    /// increase this value to take bigger steps (but risk overshooting the best solution). If learning is unstable,
    /// you might decrease it to take smaller, more careful steps. For DFP specifically, you can often use a larger
    /// initial learning rate than with simpler methods because it's better at adjusting its step size automatically.</para>
    /// </remarks>
    public new double InitialLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate during optimization.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// This value sets a lower bound on how small the learning rate can become during adaptive adjustments.
    /// It prevents the optimization from stalling due to extremely small step sizes. This value overrides the base class setting.
    /// </para>
    /// <para><b>For Beginners:</b> This is the smallest step size your model is allowed to take. With the default
    /// value of 0.000001, we're ensuring that even if the algorithm wants to take extremely tiny steps (which might
    /// make learning too slow or get stuck), it won't go below this minimum. Think of it as setting a minimum walking
    /// speed - even when being careful, you still need to keep moving forward at some pace.</para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate during optimization.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This value sets an upper bound on how large the learning rate can become during adaptive adjustments.
    /// It prevents the optimization from becoming unstable due to extremely large step sizes. This value overrides the base class setting.
    /// </para>
    /// <para><b>For Beginners:</b> This is the largest step size your model is allowed to take. With the default
    /// value of 10.0, we're ensuring that even if the algorithm thinks it should take huge steps (which might
    /// cause it to miss the best solution), it won't exceed this maximum. Think of it as setting a speed limit -
    /// even when the path seems clear, going too fast might cause you to miss your destination.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the factor by which the learning rate increases when progress is being made.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// When the optimization is making good progress, the learning rate is multiplied by this factor
    /// to accelerate convergence. A value greater than 1.0 ensures the learning rate increases.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much your model speeds up when it's making good progress.
    /// With the default value of 1.05, the step size increases by 5% whenever the model is moving in a good direction.
    /// It's like gradually speeding up when you're on a clear path - you can move faster to reach your destination sooner.
    /// A higher value makes your model speed up more aggressively when things are going well.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which the learning rate decreases when progress stalls or reverses.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// When the optimization encounters difficulties or starts to diverge, the learning rate is multiplied
    /// by this factor to take more careful steps. A value less than 1.0 ensures the learning rate decreases.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much your model slows down when it's struggling to make progress.
    /// With the default value of 0.95, the step size decreases by 5% whenever the model starts moving in circles or
    /// going in the wrong direction. It's like slowing down when the path gets tricky - you take smaller steps to
    /// avoid mistakes. A lower value makes your model slow down more quickly when it encounters difficulties.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the maximum number of iterations the optimizer will perform.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This value limits how many optimization steps will be taken before stopping, even if convergence
    /// hasn't been achieved. It prevents excessive computation time for difficult problems. This value overrides the base class setting.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum number of learning steps your model will take before
    /// stopping. With the default value of 1000, the model will try up to 1000 times to find the best solution
    /// and then stop, even if it hasn't fully converged. Think of it as setting a time limit for solving a puzzle -
    /// after a certain point, you decide to work with the best solution you've found so far rather than continuing
    /// indefinitely.</para>
    /// </remarks>
    public new int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the line search procedure within each optimization step.
    /// </summary>
    /// <value>The maximum number of line search iterations, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// Line search is a sub-procedure that finds the optimal step size in the chosen direction.
    /// This parameter limits how many attempts the line search will make to find the best step size.
    /// </para>
    /// <para><b>For Beginners:</b> During each learning step, the DFP algorithm needs to decide exactly how far
    /// to move in the chosen direction. This is called "line search" - it's like deciding whether to take a small,
    /// medium, or large step once you know which way to go. This parameter limits how many attempts the algorithm
    /// will make to find the perfect step size before moving on. With the default value of 20, it will try up to
    /// 20 different step sizes for each main iteration. Think of it as limiting how much time you spend deciding
    /// exactly how big each step should be.</para>
    /// </remarks>
    public int MaxLineSearchIterations { get; set; } = 20;

    /// <summary>
    /// Gets or sets the rate at which the algorithm adapts its approximation of the inverse Hessian matrix.
    /// </summary>
    /// <value>The adaptation rate, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the DFP algorithm updates its internal model of the function's curvature.
    /// A higher value means faster adaptation but potentially less stability, while a lower value means
    /// more stable but slower adaptation.
    /// </para>
    /// <para><b>For Beginners:</b> The DFP algorithm builds a mental map of the "shape" of the problem as it goes.
    /// This parameter controls how quickly it updates that map with new information. With the default value of 0.1,
    /// it blends 10% of new information with 90% of its existing map after each step. Think of it like learning
    /// from experience - if this value is too high, you might overreact to each new piece of information; if it's
    /// too low, you might be too slow to adapt to changing circumstances. The default provides a good balance between
    /// learning from new information and maintaining stability.</para>
    /// </remarks>
    public double AdaptationRate { get; set; } = 0.1;
}
