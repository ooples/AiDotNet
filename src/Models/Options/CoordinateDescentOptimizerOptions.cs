namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Coordinate Descent optimization algorithm, which optimizes a function
/// by solving for one variable at a time while holding others constant.
/// </summary>
/// <remarks>
/// <para>
/// Coordinate Descent is an optimization technique that minimizes a function by updating one coordinate
/// (or variable) at a time, while keeping all other coordinates fixed. This approach can be effective
/// for problems where optimizing along individual dimensions is easier than optimizing all dimensions
/// simultaneously.
/// </para>
/// <para><b>For Beginners:</b> Imagine you're trying to find the lowest point in a valley. Coordinate
/// Descent is like first walking only north/south until you can't go any lower, then switching to only
/// east/west, then back to north/south, and so on. By taking turns moving in different directions, you
/// can eventually reach the bottom of the valley. This approach can be simpler than trying to move in
/// all directions at once.</para>
/// </remarks>
public class CoordinateDescentOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient estimation.
    /// </summary>
    /// <value>A positive integer, defaulting to -1 (full batch).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// when estimating partial derivatives. Coordinate Descent traditionally uses the full dataset
    /// (batch size -1) for derivative estimation, but you can use mini-batches for faster but
    /// noisier updates on large datasets.</para>
    /// </remarks>
    public int BatchSize { get; set; } = -1;

    /// <summary>
    /// Gets or sets the rate at which the learning rate increases when performance improves.
    /// </summary>
    /// <value>The learning rate increase rate, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm is making good progress, the learning rate is increased by this factor
    /// to potentially speed up convergence.
    /// </para>
    /// <para><b>For Beginners:</b> If the algorithm is doing well (finding better solutions),
    /// it will increase its step size by this percentage. With the default value of 0.05, the step size
    /// increases by 5% when things are going well. This is like walking faster when you're confident
    /// you're heading in the right direction.</para>
    /// </remarks>
    public double LearningRateIncreaseRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the rate at which the learning rate decreases when performance worsens.
    /// </summary>
    /// <value>The learning rate decrease rate, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm's performance deteriorates, the learning rate is decreased by this factor
    /// to take more careful steps.
    /// </para>
    /// <para><b>For Beginners:</b> If the algorithm starts finding worse solutions instead of better ones,
    /// it will decrease its step size by this percentage. With the default value of 0.05, the step size
    /// decreases by 5% when things aren't going well. This is like slowing down and taking smaller steps
    /// when you realize you might be going in the wrong direction.</para>
    /// </remarks>
    public double LearningRateDecreaseRate { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the rate at which momentum increases when performance improves.
    /// </summary>
    /// <value>The momentum increase rate, defaulting to 0.01 (1%).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm is making good progress, the momentum is increased by this factor
    /// to help it move faster through flat regions and avoid local minima.
    /// </para>
    /// <para><b>For Beginners:</b> Momentum helps the algorithm maintain direction and push through
    /// difficult areas. When things are going well, momentum increases by this percentage (1% by default).
    /// It's like gaining confidence and building up speed when you're on the right track. Higher momentum
    /// helps the algorithm push through small bumps and plateaus in the solution space.</para>
    /// </remarks>
    public double MomentumIncreaseRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the rate at which momentum decreases when performance worsens.
    /// </summary>
    /// <value>The momentum decrease rate, defaulting to 0.01 (1%).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm's performance deteriorates, the momentum is decreased by this factor
    /// to prevent it from overshooting good solutions.
    /// </para>
    /// <para><b>For Beginners:</b> When the algorithm starts finding worse solutions, it reduces its
    /// momentum by this percentage (1% by default). This is like slowing down when you realize you might
    /// be rushing past the best solution. Lower momentum helps the algorithm make more careful, deliberate
    /// movements when it's not performing well.</para>
    /// </remarks>
    public double MomentumDecreaseRate { get; set; } = 0.01;
}
