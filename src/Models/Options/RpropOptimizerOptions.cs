namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Rprop (resilient backpropagation), which adapts a separate step size for every
/// weight from the SIGN of its gradient and discards the magnitude entirely.
/// </summary>
/// <remarks>
/// <para>
/// From Riedmiller and Braun, "A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP
/// Algorithm" (Proc. IEEE International Conference on Neural Networks, 1993, pp. 586-591,
/// doi:10.1109/ICNN.1993.298623). The defaults below are the paper's: eta+ = 1.2, eta- = 0.5, Delta_0 = 0.1,
/// Delta_min = 1e-6, Delta_max = 50.
/// </para>
/// <para>
/// There is deliberately no learning rate here. In Rprop the per-weight step size IS the state — it grows while
/// the gradient keeps pointing the same way and shrinks the moment it reverses — so a global learning rate would
/// have nothing to multiply. The fused Rprop kernel takes no learning-rate argument either.
/// </para>
/// <para><b>Important:</b> Rprop requires FULL-BATCH gradients. The paper is explicit that it is a batch-learning
/// method, and the reason is structural rather than incidental: the algorithm reads meaning into a gradient sign
/// flip ("I overshot, halve the step"), and on mini-batches the sign flips constantly for reasons that are just
/// sampling noise, which drives every step size down to Delta_min and stalls training. AiDotNet's Rprop
/// optimizer therefore computes the gradient over the entire training set on each iteration and exposes no batch
/// size.</para>
/// <para><b>For Beginners:</b> Most optimizers move further when the gradient is large. Rprop ignores how large
/// the gradient is and looks only at which direction it points. Each weight remembers its own step size: keep
/// going the same way and the step grows by 20%; reverse direction and you must have gone too far, so the step is
/// halved. This makes it completely immune to gradients that are vanishingly small or explosively large, which is
/// why it was so effective on the deep-ish networks of its era.</para>
/// </remarks>
public class RpropOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>Initializes Rprop options with the values from the original paper.</summary>
    public RpropOptimizerOptions()
    {
    }

    /// <summary>Creates a complete copy of an existing Rprop options instance.</summary>
    /// <param name="other">The options instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public RpropOptimizerOptions(RpropOptimizerOptions<T, TInput, TOutput> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        CopyInheritedPropertiesFrom(other);
        InitialStepSize = other.InitialStepSize;
        EtaPlus = other.EtaPlus;
        EtaMinus = other.EtaMinus;
        MinStepSize = other.MinStepSize;
        MaxStepSize = other.MaxStepSize;
        InitialLearningRate = other.InitialLearningRate;
    }

    /// <summary>
    /// Gets or sets the initial per-weight update value Delta_0.
    /// </summary>
    /// <value>The initial step size, defaulting to 0.1 (the paper's value).</value>
    /// <remarks>
    /// <para>
    /// Must lie within [<see cref="MinStepSize"/>, <see cref="MaxStepSize"/>]; the optimizer rejects values
    /// outside that range at construction rather than silently clamping on the first step.
    /// </para>
    /// <para>
    /// Note that this is 0.1 rather than the 0.01 some ports use as their default. The paper presents 0.1 as the
    /// standard choice, and the algorithm is famously insensitive to it — a poor Delta_0 is corrected within a
    /// few steps by the same doubling-and-halving that drives everything else.
    /// </para>
    /// <para><b>For Beginners:</b> How big each weight's first step is. Because the step sizes adapt so quickly,
    /// this matters far less than a learning rate would in other optimizers.</para>
    /// </remarks>
    public double InitialStepSize { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the factor by which a step size grows when the gradient keeps its sign.
    /// </summary>
    /// <value>eta+, defaulting to 1.2.</value>
    /// <remarks>
    /// <para>
    /// The paper's value. It must be greater than 1 for the step to grow at all.
    /// </para>
    /// <para><b>For Beginners:</b> When a weight keeps being pushed the same way, its step grows by 20% so it can
    /// cover ground faster. Growth is deliberately gentler than the shrinkage below, so the algorithm
    /// accelerates cautiously but brakes hard.</para>
    /// </remarks>
    public double EtaPlus { get; set; } = 1.2;

    /// <summary>
    /// Gets or sets the factor by which a step size shrinks when the gradient reverses sign.
    /// </summary>
    /// <value>eta-, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// The paper's value. It must lie strictly between 0 and 1.
    /// </para>
    /// <para><b>For Beginners:</b> A reversed gradient means the last step jumped over the bottom of the valley,
    /// so the step is halved. Note that no move is made on the step where this happens — the algorithm shrinks
    /// and waits rather than shrinking and moving.</para>
    /// </remarks>
    public double EtaMinus { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the smallest permitted step size Delta_min.
    /// </summary>
    /// <value>The minimum step size, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A floor, so repeated halving cannot shrink a weight's step to nothing and
    /// freeze it permanently.</para>
    /// </remarks>
    public double MinStepSize { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the largest permitted step size Delta_max.
    /// </summary>
    /// <value>The maximum step size, defaulting to 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A ceiling, so a long run of consistent gradients cannot compound the step size
    /// into a wild jump.</para>
    /// </remarks>
    public double MaxStepSize { get; set; } = 50.0;

    /// <summary>
    /// Gets or sets the compatibility alias for <see cref="InitialStepSize"/>.
    /// </summary>
    /// <value>The current <see cref="InitialStepSize"/>.</value>
    /// <remarks>
    /// <para>
    /// Rprop has no learning rate: the per-weight step size plays that role and is adapted by the algorithm
    /// itself. This inherited property is therefore a synchronized alias for <see cref="InitialStepSize"/> so
    /// generic configuration and logging surfaces cannot display or set an unrelated number.
    /// </para>
    /// <para><b>For Beginners:</b> This and <see cref="InitialStepSize"/> are two names for the same setting.</para>
    /// </remarks>
    public override double InitialLearningRate
    {
        get => InitialStepSize;
        set => InitialStepSize = value;
    }
}
