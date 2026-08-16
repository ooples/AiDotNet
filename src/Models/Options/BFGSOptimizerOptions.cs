namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// The BFGS algorithm is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
/// It approximates the Hessian matrix (which contains second derivatives) using gradient information,
/// making it more efficient than methods that require explicit computation of the Hessian.
/// </para>
/// <para><b>For Beginners:</b> BFGS is an advanced optimization algorithm that helps find the best solution 
/// to a problem efficiently. Think of it like finding the lowest point in a hilly landscape when you can only 
/// see the steepness at your current position. Unlike simpler methods that just go downhill, BFGS builds a 
/// "mental map" of the landscape as it explores, helping it make smarter decisions about where to go next. 
/// This makes it faster and more reliable for complex problems.</para>
/// </remarks>
public class BFGSOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the BFGSOptimizerOptions class with appropriate defaults.
    /// </summary>
    public BFGSOptimizerOptions()
    {
        MaxIterations = 1000;
    }

    /// <summary>Creates a complete copy of an existing BFGS options instance.</summary>
    /// <param name="other">The options instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    /// <remarks>
    /// Reference-typed collaborators are shared rather than duplicated — see
    /// <see cref="GradientBasedOptimizerOptions{T, TInput, TOutput}"/>'s copy helper for what that means.
    /// </remarks>
    public BFGSOptimizerOptions(BFGSOptimizerOptions<T, TInput, TOutput> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        CopyInheritedPropertiesFrom(other);
        BatchSize = other.BatchSize;
        UseLineSearch = other.UseLineSearch;
        MaxLineSearchIterations = other.MaxLineSearchIterations;
        InitialLearningRate = other.InitialLearningRate;
        // MinLearningRate and MaxLearningRate are `new` shadows of the base properties, so
        // CopyInheritedPropertiesFrom copies the BASE ones and leaves these at their defaults. They have
        // to be named here or a copy silently reverts the caller's bounds.
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
    /// BFGS traditionally uses full-batch gradients (batch size -1) because it builds an approximation of
    /// the inverse Hessian matrix that requires consistent gradients between iterations.
    /// Using mini-batches would introduce noise that disrupts the Hessian approximation.</para>
    /// </remarks>
    public int BatchSize { get; set; } = -1;

    /// <summary>
    /// Gets or sets whether each step is checked against the Armijo sufficient-decrease condition and
    /// shortened (or rejected) when it fails.
    /// </summary>
    /// <value><c>true</c> to line-search each step; <c>false</c> to take the full step. Default: <c>true</c>.</value>
    /// <remarks>
    /// <para>
    /// BFGS produces a DIRECTION; Nocedal &amp; Wright pair it with a line search (Algorithm 3.1) because
    /// the direction is only guaranteed to point downhill — the full step along it is not guaranteed to
    /// improve anything. Turning this off makes every step exactly <c>lr</c> along the direction.
    /// </para>
    /// <para><b>For Beginners:</b> A line search is "take the step, and if it made things worse, take a
    /// smaller one instead". It costs an extra forward pass per attempt and usually pays for itself.
    /// Leave it on unless you are matching another framework's fixed-step behaviour.
    /// </para>
    /// </remarks>
    public bool UseLineSearch { get; set; } = true;

    /// <summary>
    /// Gets or sets how many times a failing step is halved before it is rejected outright.
    /// </summary>
    /// <value>A positive integer, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// Backtracking terminates on its own for a genuine descent direction, so this bound only matters when
    /// the direction is not one — which happens when the inverse-Hessian approximation has gone stale.
    /// The default of 20 takes the step below 1e-6 of its original length (0.5^20), well past the point
    /// where continuing is useful.
    /// </para>
    /// <para><b>For Beginners:</b> How many times to try a smaller step before giving up on this one and
    /// leaving the parameters where they were. You rarely need to change it.
    /// </para>
    /// </remarks>
    public int MaxLineSearchIterations { get; set; } = 20;

    /// <summary>
    /// Gets or sets the initial learning rate for the optimization process.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The learning rate controls the step size during optimization. A higher value can lead to faster convergence
    /// but may cause overshooting, while a lower value provides more precise optimization but may require more iterations.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate is like the size of steps you take when searching for the lowest 
    /// point in a valley. This setting determines your initial step size. A larger value (like the default 1.0) means 
    /// taking bigger steps at first, which can help you cover ground quickly but might cause you to step over the 
    /// lowest point. The algorithm will adjust this value as it runs, but setting a good starting point can help it 
    /// find the solution faster.</para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate during optimization.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// This value sets a lower bound on how small the learning rate can become during the optimization process.
    /// It prevents the algorithm from taking excessively small steps that would slow down convergence.
    /// </para>
    /// <para><b>For Beginners:</b> As the algorithm gets closer to the best solution, it typically takes smaller 
    /// steps for precision. This setting prevents the steps from becoming too tiny, which could make progress 
    /// extremely slow. The default value (0.000001) is very small and rarely needs adjustment. You might increase 
    /// this if the algorithm seems to be making very slow progress in later iterations.</para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate during optimization.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This value sets an upper bound on how large the learning rate can become during the optimization process.
    /// It prevents the algorithm from taking excessively large steps that could cause instability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how large the algorithm's steps can become. Even if the 
    /// algorithm thinks a very large step would be beneficial, this cap prevents it from taking steps that are 
    /// too aggressive, which could cause it to overshoot the optimal solution dramatically. The default value (10.0) 
    /// works well for most problems, but you might decrease it if you notice the algorithm becoming unstable.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the factor by which to increase the learning rate when progress is good.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// When the optimization is making good progress, the learning rate is multiplied by this factor
    /// to potentially speed up convergence by taking larger steps.
    /// </para>
    /// <para><b>For Beginners:</b> When the algorithm is making good progress (moving consistently downhill), 
    /// it can try taking slightly larger steps to speed things up. This value determines how much larger those 
    /// steps become. The default value (1.05) means each successful step increases the step size by 5%. A larger 
    /// value would make the algorithm more aggressive, while a smaller value would make it more cautious.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which to decrease the learning rate when progress is poor.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// When the optimization encounters difficulties (such as overshooting or increasing error),
    /// the learning rate is multiplied by this factor to take smaller, more careful steps.
    /// </para>
    /// <para><b>For Beginners:</b> When the algorithm takes a step that doesn't improve the solution 
    /// (like stepping uphill instead of downhill), it needs to be more careful. This value determines 
    /// how much smaller the next step will be. The default value (0.95) reduces the step size by 5% 
    /// when this happens. A smaller value (like 0.5) would make the algorithm much more cautious after 
    /// a bad step, while a value closer to 1.0 would keep it more aggressive.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization process.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter limits how many iterations the algorithm will perform when trying to find
    /// the optimal solution. The algorithm will stop either when it reaches this limit
    /// or when the convergence tolerance is met, whichever comes first.
    /// </para>
    /// <para><b>For Beginners:</b> This is a safety limit that prevents the algorithm from running forever
    /// if it can't find a satisfactory solution. The default (1000 iterations) is sufficient for most problems.
    /// If your problem is particularly complex, you might need to increase this. If the algorithm consistently
    /// hits this limit without converging, it usually indicates either a very difficult problem or that other
    /// settings need adjustment.</para>
    /// </remarks>
}
