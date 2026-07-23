namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for multi-fidelity/ASHA AutoML search.
/// </summary>
/// <remarks>
/// <para>
/// Multi-fidelity search tries many configurations quickly with a smaller "budget" (for example, a subset of the
/// training data) and then promotes only the most promising trials to larger budgets.
/// </para>
/// <para>
/// ASHA (Asynchronous Successive Halving Algorithm) extends this with parallel trial execution and
/// early stopping of underperforming trials, providing 5-10x speedup over grid/random search.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of fully training every trial (slow), multi-fidelity does:
/// <list type="number">
/// <item><description>Train many candidates on a small amount of data.</description></item>
/// <item><description>Keep only the best candidates.</description></item>
/// <item><description>Train those candidates on more data.</description></item>
/// <item><description>Repeat until you reach full training.</description></item>
/// </list>
/// This usually finds strong models faster than running full training for every attempt.
/// </para>
/// </remarks>
public sealed class AutoMLMultiFidelityOptions
{
    /// <summary>
    /// Gets or sets the ordered list of training-data fractions to use as fidelity levels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Values must be in (0, 1]. The final level should be 1.0 to represent full-fidelity training.
    /// If the list does not include 1.0, multi-fidelity will append it automatically.
    /// </para>
    /// <para><b>For Beginners:</b> <c>0.25</c> means "train on 25% of the training data".</para>
    /// </remarks>
    public double[] TrainingFractions { get; set; } = { 0.25, 0.5, 1.0 };

    /// <summary>
    /// Gets or sets the reduction factor used when promoting trials between fidelity levels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 3 means "keep about 1/3 of the trials" when moving to the next fidelity level.
    /// </para>
    /// <para><b>For Beginners:</b> Higher values are more aggressive (fewer promotions).</para>
    /// </remarks>
    public double ReductionFactor { get; set; } = 3.0;

    // ============================================================
    // ASHA (Asynchronous Successive Halving) options
    // ============================================================

    /// <summary>
    /// Gets or sets whether to enable ASHA-style async parallel trial execution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, trials at each rung are executed in parallel up to <see cref="MaxParallelism"/>,
    /// and underperforming trials are stopped early based on <see cref="EarlyStoppingPatience"/>.
    /// </para>
    /// <para><b>For Beginners:</b> Enable this for faster search on multi-core systems.</para>
    /// </remarks>
    public bool EnableAsyncExecution { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum number of trials to run in parallel at each fidelity rung.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Only applies when <see cref="EnableAsyncExecution"/> is true.
    /// A value of 0 or negative means use <c>Environment.ProcessorCount</c>.
    /// </para>
    /// <para><b>For Beginners:</b> Set to the number of CPU cores you want to use.</para>
    /// </remarks>
    public int MaxParallelism { get; set; } = 0;

    /// <summary>
    /// Gets or sets the early stopping patience for individual trials within a rung.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If a trial's score doesn't improve for this many checkpoints, it is stopped early.
    /// A value of 0 or negative disables per-trial early stopping.
    /// </para>
    /// <para><b>For Beginners:</b> Higher values give trials more time but slower search.</para>
    /// </remarks>
    public int EarlyStoppingPatience { get; set; } = 3;

    /// <summary>
    /// Gets or sets the minimum improvement threshold for early stopping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A trial must improve by at least this amount to reset the patience counter.
    /// </para>
    /// <para><b>For Beginners:</b> Smaller values are more lenient; 0.001 is a good default.</para>
    /// </remarks>
    public double EarlyStoppingMinDelta { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the grace period (minimum checkpoints) before early stopping can trigger.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Trials are allowed at least this many checkpoints before being considered for early stopping.
    /// This prevents killing trials too early before they have a chance to converge.
    /// </para>
    /// <para><b>For Beginners:</b> Higher values give all trials more time to "warm up".</para>
    /// </remarks>
    public int GracePeriod { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to use aggressive bracket halving (HyperBand-style).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, multiple brackets with different starting fidelities are explored in parallel.
    /// This trades off exploration (many trials at low fidelity) vs exploitation (fewer trials at high fidelity).
    /// </para>
    /// <para><b>For Beginners:</b> Enable for broader hyperparameter exploration.</para>
    /// </remarks>
    public bool EnableHyperBandBrackets { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of HyperBand brackets to use when <see cref="EnableHyperBandBrackets"/> is true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each bracket has a different starting fidelity and reduction schedule.
    /// More brackets increase exploration diversity but also increase compute cost.
    /// </para>
    /// </remarks>
    public int HyperBandBrackets { get; set; } = 3;
}
