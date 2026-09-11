namespace AiDotNet.Models.Options;

/// <summary>
/// Settings for <see cref="AiDotNet.Optimizers.ScheduleFreeAdamWOptimizer{T, TInput, TOutput}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Schedule-Free AdamW (Defazio et al. 2024) removes the learning-rate schedule rather than
/// replacing it. Instead of annealing a rate towards the end of a run whose length must be known in
/// advance, it maintains a running average of the iterates and evaluates that. The averaging does
/// the work a decay schedule normally does, and nothing has to know how long training will last.
/// </para>
/// <para><b>For Beginners:</b> Most training lowers the learning rate as it goes, which means you
/// have to decide up front how many steps you will run — stop early and the rate never came down,
/// run longer and it came down too soon. This optimizer averages its recent positions instead, so
/// it behaves well whenever you choose to stop.
/// </para>
/// </remarks>
public class ScheduleFreeAdamWOptimizerOptions<T, TInput, TOutput>
    : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>Samples per optimizer step.</summary>
    public int BatchSize { get; set; } = 32;

    /// <inheritdoc />
    public override double InitialLearningRate { get; set; } = 0.0025;

    /// <summary>
    /// Where gradients are evaluated between the fast iterate and the running average.
    /// </summary>
    /// <remarks>
    /// The paper's beta, interpolating two classical methods: 0 is Polyak-Ruppert averaging, where
    /// gradients are taken at the fast iterate, and 1 is primal averaging, where they are taken at
    /// the average. The paper notes the interpolation gets the benefits of both, and uses 0.9.
    /// </remarks>
    public double Interpolation { get; set; } = 0.9;

    /// <summary>Second-moment decay for the AdamW step applied to the fast iterate.</summary>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>Numerical-stability epsilon.</summary>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>Decoupled weight decay.</summary>
    public double WeightDecay { get; set; } = 0.0;

    /// <summary>Steps over which the rate ramps up before the schedule-free behaviour takes over.</summary>
    /// <remarks>
    /// The one schedule this optimizer keeps. The paper still warms up -- Moonshine ramps to 1.4e-3
    /// over 8192 steps -- because the averaging cannot stabilise the very first updates, when there
    /// is almost nothing to average.
    /// </remarks>
    public int WarmupSteps { get; set; }
}
