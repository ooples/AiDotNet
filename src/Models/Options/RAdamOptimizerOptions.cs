namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the RAdam (Rectified Adam) optimization algorithm, which adds a variance
/// rectification term to Adam so that the adaptive learning rate is only trusted once enough gradient
/// history exists to estimate it reliably.
/// </summary>
/// <remarks>
/// <para>
/// RAdam was introduced in "On the Variance of the Adaptive Learning Rate and Beyond" (Liu et al., ICLR 2020,
/// arXiv:1908.03265). The paper shows that Adam's adaptive term has an undesirably large variance during the
/// first few hundred steps — which is exactly why Adam so often needs a hand-tuned warmup schedule — and
/// derives a closed-form rectification that removes the need for one.
/// </para>
/// <para><b>For Beginners:</b> Adam adapts its step size for every parameter by dividing by an estimate of how
/// much each gradient has been bouncing around. Very early in training there are only a handful of gradients to
/// estimate that from, so the estimate itself is unreliable, and dividing by an unreliable number can throw the
/// model off course. The usual workaround is "warmup": start with a tiny learning rate and ramp it up by hand.
/// RAdam replaces that guesswork with a formula. It works out how trustworthy the estimate currently is, scales
/// the step down accordingly, and — for the first handful of steps, when the estimate is not usable at all —
/// skips the adaptive part entirely and takes a plain momentum step. You get Adam's behaviour without having to
/// tune a warmup schedule.</para>
/// </remarks>
public class RAdamOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>Initializes RAdam options with the paper-compatible defaults.</summary>
    public RAdamOptimizerOptions()
    {
    }

    /// <summary>Creates a complete copy of an existing RAdam options instance.</summary>
    /// <param name="other">The options instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public RAdamOptimizerOptions(RAdamOptimizerOptions<T, TInput, TOutput> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        CopyInheritedPropertiesFrom(other);
        BatchSize = other.BatchSize;
        InitialLearningRate = other.InitialLearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        LearningRateIncreaseFactor = other.LearningRateIncreaseFactor;
        LearningRateDecreaseFactor = other.LearningRateDecreaseFactor;
        MinLearningRate = other.MinLearningRate;
        MaxLearningRate = other.MaxLearningRate;
    }

    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. The default of 32 is a good balance for RAdam.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the initial step size used for parameter updates during optimization.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// This is the alpha_t of Algorithm 2 in the paper. Because RAdam rectifies the adaptive term itself, this
    /// learning rate does not need an accompanying warmup schedule — that is the whole point of the algorithm.
    /// </para>
    /// <para><b>For Beginners:</b> Think of the learning rate as how big a step your model takes when learning.
    /// The default of 0.001 is the same value Adam uses and is a good starting point for most problems. With
    /// plain Adam you would often also have to set up a warmup schedule to protect the first few hundred steps;
    /// with RAdam you do not, because it handles that part for you.</para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates (momentum).
    /// </summary>
    /// <value>The first moment decay rate, defaulting to 0.9.</value>
    /// <remarks>
    /// <para>
    /// Beta1 is the decay rate of the running average of gradients, and is used unchanged from Adam. Note that
    /// this term is used in BOTH branches of the RAdam update: when rectification is not yet available, the
    /// bias-corrected first moment is the entire step.
    /// </para>
    /// <para><b>For Beginners:</b> Beta1 is momentum — how much the direction you were already moving in
    /// influences the direction you move next. The default of 0.9 means roughly 90% of the previous direction
    /// carries over. This smooths out the path and stops the model zigzagging.</para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for the second moment estimates (adaptive learning rates).
    /// </summary>
    /// <value>The second moment decay rate, defaulting to 0.999.</value>
    /// <remarks>
    /// <para>
    /// Beta2 governs the adaptive term, and in RAdam it additionally sets how long the un-rectified phase lasts:
    /// the maximum length of the approximated simple moving average is rho_infinity = 2/(1 - Beta2) - 1, and the
    /// adaptive term is only used once the current estimate rho_t exceeds 4. At the default 0.999 that means
    /// roughly the first four steps take plain momentum steps.
    /// </para>
    /// <para><b>For Beginners:</b> Beta2 controls how much gradient history the per-parameter step sizes are
    /// based on. The default of 0.999 takes a long-term view. It also decides how many steps RAdam waits before
    /// it starts trusting those step sizes at all — with the default, about four.</para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets a small constant added to the denominator to improve numerical stability.
    /// </summary>
    /// <value>The epsilon value, defaulting to 1e-8 (0.00000001).</value>
    /// <remarks>
    /// <para>
    /// Algorithm 2 of the paper writes the adaptive term as sqrt((1 - Beta2^t) / v_t) with no epsilon. Epsilon is
    /// the standard numerical guard carried over from Adam and present in the reference implementation; it only
    /// matters when the second moment is essentially zero, and it applies to the rectified branch only — the
    /// un-rectified branch has no denominator to guard.
    /// </para>
    /// <para><b>For Beginners:</b> Epsilon is a safety net that stops the algorithm dividing by zero when a
    /// gradient has been flat for a long time. You rarely need to change it.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the factor by which to increase the learning rate when the loss is consistently decreasing.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// Only consulted when <see cref="GradientBasedOptimizerOptions{T, TInput, TOutput}.UseAdaptiveLearningRate"/>
    /// is enabled. This outer adaptation is an AiDotNet convenience shared by every gradient optimizer, not part
    /// of the RAdam paper.
    /// </para>
    /// <para><b>For Beginners:</b> This is like speeding up when the path ahead is straight and clear. The
    /// default of 1.05 raises the learning rate by 5% after an epoch that went well.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which to decrease the learning rate when the loss is increasing or oscillating.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para>
    /// Only consulted when <see cref="GradientBasedOptimizerOptions{T, TInput, TOutput}.UseAdaptiveLearningRate"/>
    /// is enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This is like slowing down when the path gets tricky. The default of 0.95 cuts
    /// the learning rate by 5% after an epoch that went badly.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate during adaptive adjustments.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 1e-5 (0.00001).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A floor on how slow learning can get, so the model never grinds to a halt
    /// taking infinitesimally small steps.</para>
    /// </remarks>
    public override double MinLearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate during adaptive adjustments.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A ceiling on how fast learning can get, so the model never takes such a large
    /// step that it overshoots the solution.</para>
    /// </remarks>
    public override double MaxLearningRate { get; set; } = 0.1;
}
