namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer,
/// which is an efficient optimization algorithm for training machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// L-BFGS is a quasi-Newton optimization method that approximates the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
/// algorithm using limited memory. It's particularly effective for optimizing parameters in models with
/// many parameters, as it doesn't need to store the full Hessian matrix. This makes it more memory-efficient
/// than full BFGS while still providing good convergence properties.
/// </para>
/// <para><b>For Beginners:</b> L-BFGS is an advanced optimization algorithm that helps train machine learning
/// models more efficiently than simpler methods like gradient descent.
/// 
/// Think of training a machine learning model as finding the lowest point in a hilly landscape, where the
/// lowest point represents the best model parameters. While basic algorithms like gradient descent simply
/// follow the steepest downhill path, L-BFGS is smarter:
/// 
/// - It remembers information about previous steps to make better decisions about where to go next
/// - It can take larger steps when appropriate, potentially finding the lowest point faster
/// - It requires less memory than some other advanced methods, making it practical for larger models
/// 
/// L-BFGS is particularly useful when:
/// - You have many parameters to optimize (complex models)
/// - You need faster convergence than gradient descent provides
/// - You have limited memory resources compared to what full second-order methods would require
/// 
/// This class lets you configure how L-BFGS behaves during training, including how much history it
/// remembers and how it adjusts its learning rate.</para>
/// </remarks>
public class LBFGSOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    private double _armijoConstant = 1e-4;
    private double _lineSearchContractionFactor = 0.5;
    private double _powellDampingFactor = 0.2;

    /// <summary>
    /// Initializes a new instance of the LBFGSOptimizerOptions class with appropriate defaults.
    /// </summary>
    public LBFGSOptimizerOptions()
    {
        MaxIterations = 1000;
    }

    /// <summary>Creates a complete copy of another L-BFGS configuration.</summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public LBFGSOptimizerOptions(LBFGSOptimizerOptions<T, TInput, TOutput> other)
        : base(other)
    {
        BatchSize = other.BatchSize;
        MemorySize = other.MemorySize;
        LineSearchMaxSteps = other.LineSearchMaxSteps;
        ArmijoConstant = other.ArmijoConstant;
        LineSearchContractionFactor = other.LineSearchContractionFactor;
        LineSearchFallbackStep = other.LineSearchFallbackStep;
        MinimumCurvature = other.MinimumCurvature;
        PowellDampingFactor = other.PowellDampingFactor;
        InitialLearningRate = other.InitialLearningRate;
        MinLearningRate = other.MinLearningRate;
        MaxLearningRate = other.MaxLearningRate;
        LearningRateIncreaseFactor = other.LearningRateIncreaseFactor;
        LearningRateDecreaseFactor = other.LearningRateDecreaseFactor;
    }

    /// <summary>
    /// Gets or sets the batch size for gradient computation.
    /// </summary>
    /// <value>A positive integer, defaulting to -1 (full batch).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples are used to calculate gradients.
    /// L-BFGS traditionally uses full-batch gradients (batch size -1) because it maintains a history of
    /// gradient and position differences that require consistent gradients between iterations.
    /// Using mini-batches would introduce noise that disrupts the two-loop recursion.</para>
    /// </remarks>
    public int BatchSize { get; set; } = -1;

    /// <summary>
    /// Gets or sets the memory size, which determines how many previous iterations' information
    /// the L-BFGS algorithm stores to approximate the Hessian matrix.
    /// </summary>
    /// <value>The memory size, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// The memory size parameter controls how many previous iterations' gradient information is stored
    /// to approximate the inverse Hessian matrix. A larger memory size can lead to better approximations
    /// but requires more memory and computational resources per iteration.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much "history" the L-BFGS algorithm remembers
    /// when deciding where to go next.
    /// 
    /// Imagine you're hiking down a mountain to find the lowest point:
    /// - With a small memory size (like 3-5), you only remember your most recent few steps to decide
    ///   where to go next
    /// - With a larger memory size (like 10-20), you remember more of your journey, which might help
    ///   you make better decisions, but requires more mental effort
    /// 
    /// The default value of 10 works well for many problems. Consider:
    /// - Increasing it (15-20) if you have plenty of memory and the algorithm seems to be converging slowly
    /// - Decreasing it (5-8) if you're working with very large models and memory is a concern
    /// 
    /// Generally, values between 3 and 20 are common, with diminishing returns as you increase beyond that.</para>
    /// </remarks>
    public int MemorySize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum number of backtracking steps in the line search.
    /// </summary>
    /// <value>The maximum line search steps, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// L-BFGS proposes a search direction and then asks "how far along it should I actually go?".
    /// The line search starts at a full step and repeatedly shrinks it until the Armijo
    /// sufficient-decrease condition holds. This caps how many shrinks are attempted before the
    /// search gives up and takes <see cref="LineSearchFallbackStep"/> instead. With the default
    /// contraction factor of 0.5, twenty steps reduce the trial step to about one millionth of
    /// the original.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer works out which way is downhill, then has to
    /// decide how big a stride to take. It tries a big stride first; if that overshoots and lands
    /// somewhere worse, it halves the stride and tries again. This is how many times it is allowed
    /// to halve before giving up on that direction.
    /// </para>
    /// </remarks>
    public int LineSearchMaxSteps { get; set; } = 20;

    /// <summary>
    /// Gets or sets the Armijo sufficient-decrease constant used by the line search.
    /// </summary>
    /// <value>The Armijo constant, defaulting to 1e-4.</value>
    /// <remarks>
    /// <para>
    /// A trial step is accepted when
    /// <c>f(x + step·d) &lt;= f(x) + ArmijoConstant · step · ∇f(x)ᵀd</c>. This is the standard
    /// Armijo condition (Nocedal and Wright, "Numerical Optimization", Algorithm 3.1); 1e-4 is the
    /// conventional value. It must lie strictly between 0 and 1.
    /// </para>
    /// <para><b>For Beginners:</b> A step is only worth taking if it actually improves things by a
    /// meaningful amount, not just by a hair. This setting says how much improvement counts as
    /// "meaningful", measured against how much the slope promised. The small default means the
    /// optimizer is easy to satisfy, which is usually what you want — being fussy here just makes
    /// it shrink the step more often for little gain.
    /// </para>
    /// </remarks>
    public double ArmijoConstant
    {
        get => _armijoConstant;
        set => _armijoConstant = value > 0 && value < 1 && !double.IsNaN(value)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "ArmijoConstant must be strictly between 0 and 1.");
    }

    /// <summary>
    /// Gets or sets the factor by which the trial step is shrunk on each line search backtrack.
    /// </summary>
    /// <value>The contraction factor, defaulting to 0.5 (halving).</value>
    /// <remarks>
    /// <para>
    /// Must lie strictly between 0 and 1. Smaller values shrink the step aggressively and reach an
    /// acceptable step in fewer evaluations at the cost of possibly overshooting past a good one;
    /// values closer to 1 search the step length more finely but need more objective evaluations.
    /// </para>
    /// <para><b>For Beginners:</b> When a stride turns out to be too big, this is the fraction it
    /// gets multiplied by to produce the next attempt. The default halves it each time.
    /// </para>
    /// </remarks>
    public double LineSearchContractionFactor
    {
        get => _lineSearchContractionFactor;
        set => _lineSearchContractionFactor = value > 0 && value < 1 && !double.IsNaN(value)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "LineSearchContractionFactor must be strictly between 0 and 1.");
    }

    /// <summary>
    /// Gets or sets the step length used when the line search fails to find an acceptable step.
    /// </summary>
    /// <value>The fallback step, defaulting to 1e-4.</value>
    /// <remarks>
    /// <para>
    /// When no trial step satisfies the Armijo condition within
    /// <see cref="LineSearchMaxSteps"/> backtracks, the optimizer takes this small step along the
    /// search direction rather than stalling. Because the direction is guaranteed to be a descent
    /// direction, a sufficiently small step still makes progress, and continuing lets the curvature
    /// memory refresh on the next iteration.
    /// </para>
    /// <para><b>For Beginners:</b> If every stride length the optimizer tried was unsatisfying, it
    /// takes one deliberately tiny step rather than stopping dead. Standing still would leave it
    /// stuck forever; a tiny step at least changes the situation.
    /// </para>
    /// </remarks>
    public double LineSearchFallbackStep { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the smallest curvature value for which a correction pair is stored in memory.
    /// </summary>
    /// <value>The minimum curvature, defaulting to 1e-10.</value>
    /// <remarks>
    /// <para>
    /// L-BFGS stores pairs <c>(s, y)</c> where <c>s</c> is the change in position and <c>y</c> the
    /// change in gradient. The approximation is only valid while the curvature condition
    /// <c>sᵀy &gt; 0</c> holds; a pair violating it makes the implied inverse-Hessian indefinite,
    /// so the two-loop recursion can return an ascent direction, and a near-zero <c>sᵀy</c> makes
    /// its divisions blow up. Pairs whose curvature does not exceed this threshold are discarded
    /// (Nocedal and Wright, "Numerical Optimization", section 6.1).
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer speeds itself up by remembering how the slope
    /// changed as it moved. Some of those observations are useless or actively misleading — the
    /// ones where the slope barely changed, or changed the wrong way. This setting is the bar an
    /// observation has to clear before it is worth remembering.
    /// </para>
    /// </remarks>
    public double MinimumCurvature { get; set; } = 1e-10;

    /// <summary>
    /// Gets or sets the fraction of the model curvature that a damped correction pair must retain.
    /// </summary>
    /// <value>The Powell damping factor, defaulting to 0.2.</value>
    /// <remarks>
    /// <para>
    /// On a nonconvex objective the curvature condition <c>sᵀy &gt; 0</c> is frequently violated, and
    /// simply discarding those pairs starves the memory until L-BFGS degenerates into steepest
    /// descent. Powell's damped update repairs the pair instead: it replaces <c>y</c> with a blend
    /// of <c>y</c> and <c>Bs</c> chosen so that <c>sᵀy ≥ factor · sᵀBs</c> holds by construction
    /// (Powell, 1978; Nocedal and Wright, "Numerical Optimization", Procedure 18.2). The classical
    /// value is 0.2.
    /// </para>
    /// <para>
    /// Setting this to zero disables damping, restoring the plain skip-the-pair behaviour governed
    /// by <see cref="MinimumCurvature"/> alone.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer learns the shape of the surface from how the slope
    /// changed between steps. On a bumpy, non-bowl-shaped surface some of those observations imply
    /// an impossible shape. Throwing them all away leaves the optimizer knowing nothing; this
    /// setting instead nudges each bad observation to the nearest sensible one and keeps it.
    /// </para>
    /// </remarks>
    public double PowellDampingFactor
    {
        get => _powellDampingFactor;
        set => _powellDampingFactor = value >= 0 && value <= 1 && !double.IsNaN(value)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "PowellDampingFactor must be between 0 and 1 inclusive.");
    }

    /// <summary>
    /// Gets or sets the initial learning rate for the L-BFGS algorithm, which controls the initial
    /// step size during optimization.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The initial learning rate determines the step size at the beginning of the optimization process.
    /// Unlike standard gradient descent, L-BFGS can adjust this rate during optimization based on the
    /// curvature information it approximates. This property overrides the base class implementation to
    /// provide a more suitable default for L-BFGS.
    /// </para>
    /// <para><b>For Beginners:</b> The initial learning rate determines how big your first steps are
    /// when searching for the best model parameters.
    /// 
    /// Think of it like adjusting your initial step size when walking downhill:
    /// - Too small (e.g., 0.01): You'll move very cautiously but might take a long time to reach the bottom
    /// - Too large (e.g., 10.0): You might move quickly but risk overshooting the lowest point
    /// 
    /// L-BFGS is special because it can adjust this step size automatically as it goes, but the initial
    /// value still matters. The default of 1.0 is generally a good starting point for L-BFGS, which is
    /// higher than typical values for simpler algorithms like gradient descent (which might use 0.01-0.1).
    /// 
    /// Note: This property uses the "new" keyword because it overrides the base class property with a
    /// different default value that's more appropriate for L-BFGS.</para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the minimum learning rate allowed during optimization, preventing the learning
    /// rate from becoming too small.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 1e-6 (0.000001).</value>
    /// <remarks>
    /// <para>
    /// This parameter sets a lower bound on the learning rate during optimization. If the adaptive
    /// learning rate mechanism attempts to reduce the learning rate below this value, it will be
    /// clamped to this minimum. This helps prevent the algorithm from taking steps that are too small
    /// to make meaningful progress. This property overrides the base class implementation to provide
    /// a more suitable default for L-BFGS.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the algorithm from taking steps that are too
    /// tiny to make meaningful progress.
    /// 
    /// As L-BFGS adjusts its step size during training, it might sometimes decide to take very small
    /// steps. This parameter sets a minimum size - if the algorithm wants to take an even smaller step,
    /// it will use this minimum value instead.
    /// 
    /// The default value of 0.000001 (written as 1e-6 in scientific notation) is very small, allowing
    /// the algorithm to take tiny steps when appropriate, but not so small that they become ineffective.
    /// 
    /// You typically won't need to change this unless:
    /// - You notice the algorithm is making extremely slow progress in the later stages of training
    ///   (might want to increase it)
    /// - You're working with a function that requires extremely precise optimization
    ///   (might want to decrease it)
    /// 
    /// Note: This property uses the "new" keyword because it overrides the base class property with a
    /// different default value that's more appropriate for L-BFGS.</para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum learning rate allowed during optimization, preventing the learning
    /// rate from becoming too large.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper bound on the learning rate during optimization. If the adaptive
    /// learning rate mechanism attempts to increase the learning rate above this value, it will be
    /// clamped to this maximum. This helps prevent the algorithm from taking steps that are too large,
    /// which could cause instability or divergence. This property overrides the base class implementation
    /// to provide a more suitable default for L-BFGS.
    /// </para>
    /// <para><b>For Beginners:</b> This setting prevents the algorithm from taking steps that are too
    /// large, which could cause it to miss the optimal solution.
    /// 
    /// As L-BFGS adjusts its step size during training, it might sometimes decide to take very large
    /// steps. This parameter sets a maximum size - if the algorithm wants to take an even larger step,
    /// it will use this maximum value instead.
    /// 
    /// The default value of 10.0 allows the algorithm to take fairly large steps when appropriate, but
    /// not so large that it risks completely overshooting good solutions.
    /// 
    /// You typically won't need to change this unless:
    /// - You notice the algorithm is making wild jumps and not converging well
    ///   (might want to decrease it)
    /// - You're working with a function that has very flat regions that require large steps
    ///   (might want to increase it)
    /// 
    /// Note: This property uses the "new" keyword because it overrides the base class property with a
    /// different default value that's more appropriate for L-BFGS.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is increased when the algorithm determines
    /// that larger steps would be beneficial.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// During optimization, if the algorithm determines that progress is being made consistently, it may
    /// increase the learning rate to accelerate convergence. This parameter controls how aggressively
    /// the learning rate is increased, with values greater than 1.0 representing the multiplicative
    /// factor applied to the current learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm increases its step
    /// size when it's making good progress.
    /// 
    /// When L-BFGS is moving in a promising direction and making consistent progress, it might decide
    /// to increase its step size to get to the solution faster. This parameter determines how much it
    /// increases the step size each time:
    /// 
    /// - With the default value of 1.05, the step size increases by 5% each time
    /// - A value of 1.1 would increase the step size by 10% each time
    /// - A value of 1.01 would increase the step size by just 1% each time
    /// 
    /// Higher values make the algorithm more aggressive in speeding up when things are going well, but
    /// might also make it more likely to overshoot. The default value of 1.05 provides a moderate
    /// increase that works well in most situations.</para>
    /// </remarks>

    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    /// <summary>
    /// Gets or sets the factor by which the learning rate is decreased when the algorithm encounters
    /// difficulties or needs to take more careful steps.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// During optimization, if the algorithm encounters challenges such as increasing error or difficult
    /// terrain in the optimization landscape, it may decrease the learning rate to take more careful steps.
    /// This parameter controls how quickly the learning rate is reduced, with values less than 1.0
    /// representing the multiplicative factor applied to the current learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm reduces its step
    /// size when it encounters difficulties.
    /// 
    /// When L-BFGS isn't making good progress or finds itself in a tricky part of the optimization
    /// landscape, it might decide to take smaller, more careful steps. This parameter determines how
    /// much it decreases the step size each time:
    /// 
    /// - With the default value of 0.95, the step size decreases by 5% each time
    /// - A value of 0.9 would decrease the step size by 10% each time
    /// - A value of 0.99 would decrease the step size by just 1% each time
    /// 
    /// Lower values make the algorithm more cautious when it encounters problems, quickly reducing
    /// step size to navigate difficult areas. The default value of 0.95 provides a moderate decrease
    /// that works well in most situations, allowing the algorithm to adapt without becoming too timid.
    /// 
    /// This parameter works together with LearningRateIncreaseFactor to help the algorithm adapt its
    /// step size throughout the optimization process.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the maximum number of iterations the L-BFGS algorithm will perform before stopping.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper limit on the number of optimization steps the algorithm will take.
    /// Even if other stopping criteria (such as convergence thresholds) have not been met, the algorithm
    /// will terminate after this many iterations. This property overrides the base class implementation
    /// to provide a more suitable default for L-BFGS.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the maximum number of steps the algorithm
    /// will take before giving up, even if it hasn't found the optimal solution yet.
    /// 
    /// Think of this as a safety limit to prevent the algorithm from running forever. The L-BFGS algorithm
    /// will stop when either:
    /// - It finds a solution that's good enough (based on other stopping criteria), or
    /// - It reaches this maximum number of iterations
    /// 
    /// The default value of 1000 is typically sufficient for many problems. Consider:
    /// - Increasing it (e.g., to 2000 or 5000) for complex problems where the algorithm might need
    ///   more steps to converge to a good solution
    /// - Decreasing it (e.g., to 500 or less) if you need faster results and can accept a less
    ///   optimal solution, or if you're just testing the algorithm
    /// 
    /// L-BFGS typically converges faster than simpler methods like gradient descent, so it often
    /// needs fewer iterations to reach a good solution. However, the exact number needed depends
    /// greatly on your specific problem.
    /// 
    /// Note: The default value for MaxIterations is set to 1000 via the constructor, which is more
    /// appropriate for L-BFGS than the base class default.</para>
    /// </remarks>
}
