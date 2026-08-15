namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for ASGD (Averaged Stochastic Gradient Descent), which runs ordinary decayed SGD
/// while maintaining a running average of the iterates and uses that average as the answer.
/// </summary>
/// <remarks>
/// <para>
/// The averaging idea is Ruppert (1988) and Polyak and Juditsky, "Acceleration of Stochastic Approximation by
/// Averaging" (SIAM Journal on Control and Optimization 30(4), 1992, doi:10.1137/0330046): averaging the tail of
/// an SGD trajectory attains the optimal asymptotic convergence rate even though each individual step does not.
/// </para>
/// <para>
/// The specific parameterization here — the learning-rate schedule gamma_t = gamma_0 (1 + a*gamma_0*t)^-c and the
/// decay term (1 - lambda*gamma_t) folded into the weight update — follows Xu, "Towards Optimal One Pass Large
/// Scale Learning with Averaged Stochastic Gradient Descent" (arXiv:1107.2490, 2011), which is also the form
/// implemented by the fused ASGD kernel and by PyTorch's <c>torch.optim.ASGD</c>.
/// </para>
/// <para><b>For Beginners:</b> Plain SGD bounces around the answer rather than settling on it, because every step
/// reacts to one noisy batch. ASGD does not try to stop the bouncing — it just keeps a running average of where
/// the model has been, and the average lands much closer to the true answer than any individual position does.
/// Think of estimating the centre of a dartboard: one dart tells you little, but the average of many darts is
/// very accurate even if no single dart was good.</para>
/// <para><b>Important:</b> With the default <see cref="T0"/> of 1,000,000 the averaging never actually starts in
/// a normal-length run, and ASGD behaves as decayed SGD. That matches PyTorch's default. To get the averaging
/// benefit, set <see cref="T0"/> to roughly the step at which you expect training to have settled — commonly
/// halfway through the run.</para>
/// </remarks>
public class ASGDOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training examples the optimizer looks at before each update.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the initial step size gamma_0 used for parameter updates.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// This is gamma_0 in gamma_t = gamma_0 (1 + Lambda*gamma_0*t)^-Alpha. Note that ASGD's default is 0.01,
    /// ten times larger than Adam's 0.001, because ASGD is plain SGD underneath and has no per-parameter
    /// adaptive scaling to shrink its steps.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step to take. ASGD tolerates — and benefits from — a larger value
    /// than Adam-style optimizers, because the averaging cleans up the resulting noise.</para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the decay coefficient lambda.
    /// </summary>
    /// <value>The decay term, defaulting to 1e-4.</value>
    /// <remarks>
    /// <para>
    /// Lambda does double duty, exactly as in the reference formulation. It is the L2 decay applied
    /// multiplicatively to the weights each step — theta &lt;- (1 - Lambda*gamma_t)*theta - gamma_t*g — and it is
    /// also the <c>a</c> of the learning-rate schedule gamma_t = gamma_0 (1 + Lambda*gamma_0*t)^-Alpha. The paper
    /// notes it should be a constant factor times the smallest eigenvalue of the Hessian, and that it must not
    /// be zero.
    /// </para>
    /// <para><b>For Beginners:</b> This gently shrinks the weights toward zero on every step (which discourages
    /// overfitting) and simultaneously controls how fast the step size decays. Larger values mean stronger
    /// shrinkage and faster decay.</para>
    /// </remarks>
    public double Lambda { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the decay exponent for the learning-rate schedule.
    /// </summary>
    /// <value>The exponent, defaulting to 0.75.</value>
    /// <remarks>
    /// <para>
    /// The <c>c</c> of gamma_t = gamma_0 (1 + Lambda*gamma_0*t)^-c. The paper recommends 2/3 for quadratic losses
    /// and 3/4 for non-quadratic ones; 0.75 is the general-purpose default and the one PyTorch uses.
    /// </para>
    /// <para><b>For Beginners:</b> How quickly the step size shrinks as training proceeds. Higher means faster
    /// shrinking. The default is a good general choice.</para>
    /// </remarks>
    public double Alpha { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the step at which averaging begins.
    /// </summary>
    /// <value>The averaging start step, defaulting to 1e6.</value>
    /// <remarks>
    /// <para>
    /// Controls the averaging weight mu_t = 1 / max(1, t - T0). While t is at or below T0, mu_t is 1 and the
    /// running average simply tracks the current iterate; past T0 it becomes a true running mean of the tail of
    /// the trajectory. Averaging only helps once the iterates are fluctuating around the solution rather than
    /// still travelling toward it, which is why it is started late rather than from step 1.
    /// </para>
    /// <para><b>For Beginners:</b> When to start keeping the running average. Averaging in the early steps would
    /// drag the answer back toward where training started, so the default waits — in fact the default (a million
    /// steps) effectively never starts, which means you get plain decayed SGD unless you lower this. Set it to
    /// around half your total number of steps to actually use the averaging.</para>
    /// </remarks>
    public double T0 { get; set; } = 1e6;

    /// <summary>
    /// Gets or sets the L2 weight decay added to the gradient.
    /// </summary>
    /// <value>The weight decay, defaulting to 0.</value>
    /// <remarks>
    /// <para>
    /// This is the conventional coupled weight decay g &lt;- g + WeightDecay*theta, applied before the step. It is
    /// separate from <see cref="Lambda"/>, which enters multiplicatively and also drives the learning-rate
    /// schedule; both are present because the reference implementation exposes both.
    /// </para>
    /// <para><b>For Beginners:</b> An extra pull toward zero for every weight, used to reduce overfitting. Zero
    /// (the default) turns it off; typical non-zero values are around 1e-4 to 1e-2.</para>
    /// </remarks>
    public double WeightDecay { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the factor by which to increase the learning rate when the loss is consistently decreasing.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para>
    /// Only consulted when <see cref="GradientBasedOptimizerOptions{T, TInput, TOutput}.UseAdaptiveLearningRate"/>
    /// is enabled. This outer adaptation is an AiDotNet convenience shared by every gradient optimizer, and it
    /// operates on gamma_0, on top of ASGD's own schedule.
    /// </para>
    /// <para><b>For Beginners:</b> Speeds learning up by 5% after an epoch that went well. Off by default.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which to decrease the learning rate when the loss is increasing or oscillating.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Slows learning by 5% after an epoch that went badly. Off by default.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate during adaptive adjustments.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 1e-5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A floor on how slow learning can get.</para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate during adaptive adjustments.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A ceiling on how fast learning can get.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 0.1;
}
