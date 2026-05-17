using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Adam optimization algorithm, which combines the benefits of AdaGrad and RMSProp.
/// </summary>
/// <remarks>
/// <para>
/// Adam (Adaptive Moment Estimation) is a popular optimization algorithm that computes adaptive learning rates for each parameter.
/// It stores both an exponentially decaying average of past gradients (first moment) and past squared gradients (second moment).
/// </para>
/// <para><b>For Beginners:</b> Adam is like a smart learning assistant that remembers both the direction (momentum) and the
/// size of previous steps. It automatically adjusts how big each step should be for each parameter, making it easier to train
/// models without having to manually tune the learning rate. Adam is often a good default choice for many machine learning problems.</para>
/// </remarks>
public class AdamOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Default ctor — overrides the base <see cref="GradientBasedOptimizerOptions{T,TInput,TOutput}.EnableGradientClipping"/>
    /// default from <c>false</c> to <c>true</c> with <see cref="GradientBasedOptimizerOptions{T,TInput,TOutput}.MaxGradientNorm"/>
    /// at the canonical PyTorch transformer-training value of <c>1.0</c>.
    /// Without clipping, Adam's first-step bias correction (biasC1 ≈ 0.1,
    /// biasC2 ≈ 0.001) creates huge updates on randomly-initialised large
    /// models — Hawk's 135M-parameter LM diverges from loss 0.43 to 6.97
    /// over 10 iterations on default LR=1e-3 (issue #1275 acceptance
    /// criterion 3). The PyTorch convention for transformer training is
    /// <c>torch.nn.utils.clip_grad_norm_(params, 1.0)</c> after every
    /// backward; Adam matches that convention by default. Callers who need
    /// the unclipped behaviour can explicitly set
    /// <c>EnableGradientClipping = false</c>.
    /// </summary>
    public AdamOptimizerOptions()
    {
        EnableGradientClipping = true;
        MaxGradientNorm = 1.0;
    }

    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// The batch size determines how many samples are processed before updating the model parameters.
    /// Larger batch sizes provide more stable gradient estimates but use more memory.
    /// </para>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model:
    ///
    /// - BatchSize = 1: Update after each sample (true stochastic)
    /// - BatchSize = 32: Update after every 32 samples (typical mini-batch)
    /// - BatchSize = [entire dataset]: Batch gradient descent
    ///
    /// The default of 32 is a good balance between speed and stability for Adam.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the initial learning rate for the Adam optimizer.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big each step is during training.
    /// A value of 0.001 means taking small, careful steps. If this value is too large, the model might miss the optimal solution
    /// by stepping too far. If it's too small, training will take a very long time. The default of 0.001 works well for most problems,
    /// which is why Adam is popular - it doesn't require much tuning of this value.</para>
    /// </remarks>
    public override double InitialLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the exponential decay rate for the first moment estimates.
    /// </summary>
    /// <value>The beta1 value, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta1 controls how much the algorithm remembers about the direction it was moving in previous steps.
    /// A value of 0.9 means it gives 90% importance to past directions and 10% to the new direction.
    /// Think of it like steering a boat - this parameter determines how much you consider your previous steering direction
    /// versus the new direction you want to go. Higher values (closer to 1) make for smoother but potentially slower learning.</para>
    /// </remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the exponential decay rate for the second moment estimates.
    /// </summary>
    /// <value>The beta2 value, defaulting to 0.999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta2 controls how much the algorithm remembers about the size of previous steps for each parameter.
    /// A value of 0.999 means it gives 99.9% importance to past step sizes and only 0.1% to new information.
    /// This helps stabilize learning by preventing wild changes in step size. Think of it like remembering how bumpy the road has been
    /// for each wheel of your car, allowing you to adjust the suspension accordingly for a smoother ride.</para>
    /// </remarks>
    public double Beta2 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets a small constant added to denominators to prevent division by zero.
    /// </summary>
    /// <value>The epsilon value, defaulting to 0.00000001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Epsilon is a tiny safety value that prevents the algorithm from crashing
    /// when it would otherwise divide by zero. It's like having training wheels that only activate when needed.
    /// You typically don't need to change this unless you're experiencing numerical stability issues.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the policy for the PyTorch GradScaler-style anomaly guard
    /// that skips an Adam step when any gradient contains NaN or Inf.
    /// </summary>
    /// <value>Defaults to <see cref="AdamAnomalyGuardMode.Auto"/>.</value>
    /// <remarks>
    /// <para>
    /// The guard scans every gradient element once per <c>Step</c>. Without it,
    /// a single NaN/Inf gradient permanently poisons Adam's <c>m</c> / <c>v</c>
    /// moment accumulators and every subsequent step produces NaN weights.
    /// With it, the offending step is a no-op (parameters, moments, and the
    /// step index all stay put) and training resumes once gradients become
    /// finite again. Verified on HopeNetwork memorization where step ~10
    /// produces a NaN gradient from <c>sqrt(v_hat) ≈ 0</c>.
    /// </para>
    /// <para>
    /// The scan is O(total-gradient-elements) per step. On large models
    /// where gradients are already the dominant cost, this is measurable
    /// overhead. The <c>Never</c> mode lets callers skip the scan when
    /// upstream NaN/Inf is impossible (e.g. fully-deterministic fp64
    /// regression tests). <c>Auto</c> currently behaves like
    /// <c>Always</c>; reserved for a future heuristic that gates on the
    /// numeric type.
    /// </para>
    /// </remarks>
    public AdamAnomalyGuardMode AnomalyGuardMode { get; set; } = AdamAnomalyGuardMode.Auto;

    /// <summary>
    /// Gets or sets whether to automatically adjust the Beta parameters during training.
    /// </summary>
    /// <value>True to use adaptive betas (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the algorithm will automatically adjust how much it relies on past information
    /// based on how well it's performing. If the model is improving, it will trust its memory more.
    /// If performance worsens, it will pay more attention to new information. This helps the algorithm adapt to different phases of learning,
    /// like slowing down when approaching the destination and speeding up when far away.</para>
    /// </remarks>
    public bool UseAdaptiveBetas { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum allowed value for Beta1.
    /// </summary>
    /// <value>The minimum Beta1 value, defaulting to 0.8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents Beta1 from becoming too small, which would make the algorithm
    /// ignore past directions too much. Even if Beta1 keeps decreasing, it won't go below this value.
    /// A minimum of 0.8 ensures the algorithm always considers at least some of its previous momentum,
    /// preventing it from changing direction too abruptly.</para>
    /// </remarks>
    public double MinBeta1 { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the maximum allowed value for Beta1.
    /// </summary>
    /// <value>The maximum Beta1 value, defaulting to 0.999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents Beta1 from becoming too large, which would make the algorithm
    /// rely too heavily on past directions and adapt too slowly to new information. Even if Beta1 keeps increasing,
    /// it won't go above this value. A maximum of 0.999 ensures the algorithm always incorporates at least some new directional information.</para>
    /// </remarks>
    public double MaxBeta1 { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets the minimum allowed value for Beta2.
    /// </summary>
    /// <value>The minimum Beta2 value, defaulting to 0.8.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents Beta2 from becoming too small, which would make the algorithm
    /// ignore past step sizes too much. Even if Beta2 keeps decreasing, it won't go below this value.
    /// A minimum of 0.8 ensures the algorithm always considers at least some of its previous step size information,
    /// maintaining some stability in the learning process.</para>
    /// </remarks>
    public double MinBeta2 { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the maximum allowed value for Beta2.
    /// </summary>
    /// <value>The maximum Beta2 value, defaulting to 0.9999.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents Beta2 from becoming too large, which would make the algorithm
    /// rely too heavily on past step sizes and adapt too slowly. Even if Beta2 keeps increasing, it won't go above this value.
    /// A maximum of 0.9999 ensures the algorithm always incorporates at least some new information about step sizes,
    /// allowing it to adapt to changing conditions during training.</para>
    /// </remarks>
    public double MaxBeta2 { get; set; } = 0.9999;

    /// <summary>
    /// Gets or sets whether to use the AMSGrad variant of Adam.
    /// </summary>
    /// <value><c>true</c> to use AMSGrad; otherwise, <c>false</c>. Defaults to <c>false</c>.</value>
    /// <remarks>
    /// <para>
    /// AMSGrad (Reddi, Kale, Kumar 2018, "On the Convergence of Adam and Beyond") replaces
    /// the bias-corrected second-moment v̂_t with v̂_max_t = max(v̂_max_{t-1}, v̂_t) before
    /// dividing m̂_t. The Adam paper's convergence proof relies on v_t being non-decreasing
    /// on average; in practice v_t can shrink rapidly when gradients drop after convergence,
    /// at which point m_t / sqrt(v_t) drifts the parameters away from a tight optimum. The
    /// MoreData_ShouldNotDegrade invariant in the model-family test suite catches this as
    /// 200-iter loss greater than 50-iter loss on fixed-input regression (NTM, GRU, DBM,
    /// and similar recurrent / stateful models — see #1332 cluster 6 + cluster 1.1).
    /// AMSGrad provably prevents this drift at the cost of slightly slower convergence on
    /// well-conditioned objectives.
    /// </para>
    /// <para><b>For Beginners:</b> Standard Adam can drift its parameters away from a
    /// good solution after the model has already converged on simple problems. AMSGrad
    /// is a small math change to Adam that prevents this drift. Enable it for stateful
    /// or recurrent models where you've observed loss climbing back up during long
    /// training runs.</para>
    /// </remarks>
    public bool UseAMSGrad { get; set; } = false;
}
