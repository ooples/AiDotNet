namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Levenberg-Marquardt optimization algorithm, which is used
/// for non-linear least squares optimization in machine learning and AI models.
/// </summary>
/// <remarks>
/// <para>
/// The Levenberg-Marquardt algorithm combines the Gauss-Newton method and gradient descent
/// to efficiently solve non-linear least squares problems. It adaptively switches between the two
/// approaches depending on how well the optimization is progressing, offering both stability and speed.
/// The damping factor controls this adaptation by determining whether the algorithm behaves more like
/// gradient descent (higher damping) or more like Gauss-Newton (lower damping).
/// </para>
/// <para><b>For Beginners:</b> The Levenberg-Marquardt algorithm is a powerful technique for training
/// AI models that need to make accurate predictions. It works by repeatedly adjusting the model's
/// internal settings (parameters) to reduce prediction errors.
/// 
/// Think of it like tuning a musical instrument:
/// - You listen to how "off" the sound is (the error)
/// - You make small adjustments to the tuning pegs
/// - You check if the sound improved or got worse
/// - You keep adjusting until the instrument sounds right
/// 
/// The "damping factor" controls how boldly or cautiously the algorithm makes adjustments:
/// - Higher damping = smaller, more careful adjustments (slower but more stable)
/// - Lower damping = larger, more aggressive adjustments (faster but potentially unstable)
/// 
/// The algorithm automatically adjusts this damping factor as it progresses, becoming more
/// aggressive when things are going well and more cautious when improvements are hard to find.
/// This class allows you to configure how this damping behaves during training.
/// </para>
/// </remarks>
public class LevenbergMarquardtOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for gradient computation.
    /// </summary>
    /// <value>A positive integer, defaulting to -1 (full batch).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples are used to calculate gradients.
    /// Levenberg-Marquardt uses full-batch gradients (batch size -1) because it requires computing the
    /// Jacobian matrix across the entire dataset to properly construct the normal equations.
    /// Using mini-batches would introduce noise that makes the Jacobian approximation unreliable
    /// and would compromise the algorithm's ability to balance between Gauss-Newton and gradient descent.</para>
    /// </remarks>
    public int BatchSize { get; set; } = -1;

    /// <summary>
    /// Gets or sets the starting value for the damping factor used in the algorithm.
    /// </summary>
    /// <value>The initial damping factor, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The damping factor (µ) controls the balance between the Gauss-Newton method and gradient descent.
    /// Higher values make the algorithm behave more like gradient descent (more stable but slower),
    /// while lower values make it behave more like Gauss-Newton (faster but potentially unstable).
    /// This parameter sets the initial value used when optimization begins.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how cautiously the algorithm starts.
    /// 
    /// Think of it like learning to drive:
    /// - A higher value (like 1.0) means starting very carefully, making small adjustments
    /// - A lower value (like 0.01) means starting more confidently, making larger adjustments
    /// 
    /// The default value of 0.1 provides a good balance for most problems. If your training seems
    /// unstable at the beginning, try increasing this value. If it's progressing too slowly, try
    /// decreasing it.
    /// </para>
    /// </remarks>
    public double InitialDampingFactor { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the factor by which the damping factor is increased when an iteration fails to improve the solution.
    /// </summary>
    /// <value>The damping factor increase multiplier, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// When an optimization step fails to reduce the error, the algorithm increases the damping factor
    /// by multiplying it by this value. This makes subsequent steps more conservative, helping the
    /// algorithm recover from unsuccessful attempts. Higher values cause more dramatic shifts toward
    /// gradient descent behavior when progress stalls.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much more cautious the algorithm becomes
    /// when it makes a mistake.
    /// 
    /// When the algorithm tries an adjustment that makes things worse instead of better:
    /// - It increases the damping factor to take more careful steps
    /// - This increase is determined by multiplying the current damping by this value
    /// 
    /// The default value of 10.0 means:
    /// - If the current damping is 0.1 and an adjustment fails
    /// - The new damping becomes 0.1 × 10.0 = 1.0
    /// - The next adjustment will be about 10 times more cautious
    /// 
    /// This helps the algorithm recover quickly from poor steps without getting completely stuck.
    /// </para>
    /// </remarks>
    public double DampingFactorIncreaseFactor { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the factor by which the damping factor is decreased when an iteration successfully improves the solution.
    /// </summary>
    /// <value>The damping factor decrease multiplier, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// When an optimization step successfully reduces the error, the algorithm decreases the damping factor
    /// by multiplying it by this value. This makes subsequent steps more aggressive, allowing the
    /// algorithm to converge faster when progress is being made. Lower values cause more dramatic shifts
    /// toward Gauss-Newton behavior when the optimization is going well.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much more confident the algorithm becomes
    /// when it makes successful adjustments.
    /// 
    /// When the algorithm makes an adjustment that improves the model:
    /// - It decreases the damping factor to take larger, bolder steps
    /// - This decrease is determined by multiplying the current damping by this value
    /// 
    /// The default value of 0.1 means:
    /// - If the current damping is 1.0 and an adjustment succeeds
    /// - The new damping becomes 1.0 × 0.1 = 0.1
    /// - The next adjustment will be about 10 times more aggressive
    /// 
    /// This helps the algorithm learn faster when it's on the right track, speeding up training.
    /// </para>
    /// </remarks>
    public double DampingFactorDecreaseFactor { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum allowed value for the damping factor.
    /// </summary>
    /// <value>The minimum damping factor, defaulting to 1e-8 (0.00000001).</value>
    /// <remarks>
    /// <para>
    /// This parameter establishes a lower bound for the damping factor to prevent numerical instability.
    /// If repeated successful iterations would decrease the damping factor below this value, it will be
    /// clamped to this minimum. Very low damping factors can lead to the algorithm behaving too aggressively,
    /// potentially causing divergence or numerical issues in the matrix operations.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the algorithm from becoming too aggressive,
    /// even after many successful steps.
    /// 
    /// As the algorithm makes more and more successful adjustments, it might keep reducing the
    /// damping factor to take bigger steps. This setting puts a limit on how low the damping can go,
    /// preventing the algorithm from becoming unstable.
    /// 
    /// The default value of 0.00000001 (1e-8) is very small, allowing the algorithm to become quite
    /// aggressive when things are going well, but still providing some stability protection.
    /// 
    /// Most users won't need to change this setting. If you experience numerical errors during training,
    /// you might try increasing this value slightly.
    /// </para>
    /// </remarks>
    public double MinDampingFactor { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the maximum allowed value for the damping factor.
    /// </summary>
    /// <value>The maximum damping factor, defaulting to 1e8 (100,000,000).</value>
    /// <remarks>
    /// <para>
    /// This parameter establishes an upper bound for the damping factor to prevent the algorithm from becoming
    /// too conservative. If repeated failed iterations would increase the damping factor above this value,
    /// it will be clamped to this maximum. Very high damping factors can cause the algorithm to take
    /// extremely small steps, essentially stalling progress.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the algorithm from becoming too cautious,
    /// even after many failed steps.
    /// 
    /// If the algorithm repeatedly makes adjustments that don't improve the model, it might keep
    /// increasing the damping factor to take smaller steps. This setting puts a limit on how high
    /// the damping can go, preventing the algorithm from practically freezing.
    /// 
    /// The default value of 100,000,000 (1e8) is very large, allowing the algorithm to become quite
    /// cautious when necessary, but still ensuring it will make some progress.
    /// 
    /// Most users won't need to change this setting. If your training seems to get stuck making almost
    /// no progress, you might try decreasing this value.
    /// </para>
    /// </remarks>
    public double MaxDampingFactor { get; set; } = 1e8;

    /// <summary>
    /// Gets or sets whether the damping factor should be adaptively updated based on the success or failure of each iteration.
    /// </summary>
    /// <value>Flag indicating whether to use adaptive damping, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// When set to true, the algorithm will automatically adjust the damping factor after each iteration
    /// based on whether the error increased or decreased. When set to false, the damping factor remains
    /// fixed at the initial value throughout the optimization process. Adaptive damping is one of the key
    /// features of the Levenberg-Marquardt algorithm that makes it efficient for many problems.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm automatically
    /// adjusts how cautiously it makes changes.
    /// 
    /// When enabled (the default setting):
    /// - The algorithm becomes more aggressive (lower damping) after successful steps
    /// - The algorithm becomes more cautious (higher damping) after unsuccessful steps
    /// - This adaptive behavior helps it find the right balance between speed and stability
    /// 
    /// When disabled:
    /// - The algorithm always uses the same damping factor (set by InitialDampingFactor)
    /// - It doesn't become more aggressive or cautious based on its progress
    /// 
    /// For most problems, you should leave this enabled (true). Disabling it is rarely needed and
    /// mainly useful for certain specialized problems or for educational purposes to understand
    /// how adaptive behavior affects the optimization.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveDampingFactor { get; set; } = true;

    /// <summary>
    /// Gets or sets a custom matrix decomposition method for solving the linear system in each Levenberg-Marquardt iteration.
    /// </summary>
    /// <value>The custom matrix decomposition to use, or null to use the default method.</value>
    /// <remarks>
    /// <para>
    /// Each iteration of the Levenberg-Marquardt algorithm requires solving a linear system of equations
    /// involving the Jacobian matrix. By default, the algorithm chooses an appropriate decomposition method
    /// based on the problem characteristics, but this property allows you to specify a custom decomposition
    /// method if desired. This is an advanced option that can improve performance or numerical stability
    /// for specific types of problems.
    /// </para>
    /// <para><b>For Beginners:</b> This is an advanced setting that most users don't need to change.
    /// 
    /// The Levenberg-Marquardt algorithm needs to solve complex math equations at each step. This
    /// setting allows experts to specify exactly how those equations should be solved.
    /// 
    /// The default value (null) means the algorithm will automatically choose an appropriate method
    /// for solving these equations. This automatic choice works well for the vast majority of problems.
    /// 
    /// You would only need to change this if you have a specialized problem with unique numerical
    /// characteristics and understand the mathematical details of different matrix decomposition methods.
    /// If you're just getting started with AI and optimization, you can safely ignore this setting.
    /// </para>
    /// </remarks>
    public IMatrixDecomposition<T>? CustomDecomposition { get; set; }
}
