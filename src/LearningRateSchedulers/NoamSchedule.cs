namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Implements the Noam learning rate schedule from "Attention Is All You Need"
/// (Vaswani et al. 2017, §5.3): inverse-square-root decay with linear warmup.
/// </summary>
/// <remarks>
/// <para>
/// Formula:
/// <code>
///   lr(t) = factor · d_model^(-0.5) · min(t^(-0.5), t · warmup^(-1.5))
/// </code>
/// where t is the 1-indexed training step from the paper. Peaks at t = warmup
/// with value <c>factor · d_model^(-0.5) · warmup^(-0.5)</c>, then decays as
/// t^(-0.5).
/// </para>
/// <para>
/// Step-counter convention (matches PyTorch / HuggingFace transformer
/// schedulers): the library's <see cref="LearningRateSchedulerBase._currentStep"/>
/// is incremented at end-of-batch and represents "batches completed so far"
/// (0-based). The Noam paper's t is 1-based, so this scheduler maps
/// <c>t = step + 1</c> internally:
/// <list type="bullet">
///   <item>Before any Step() call, <c>_currentStep = 0</c> ⇒ <c>t = 1</c> ⇒ warmup-start LR.</item>
///   <item>Batch N reads the LR that was set by the (N-1)th Step() call ⇒ lr(t=N) ⇒ <c>t = step + 1</c> with <c>step = N-1</c>.</item>
///   <item>Reset restores the warmup-start LR (NOT <c>_baseLearningRate</c>, which we use as a peak-LR sentinel for the base ctor's positive guard).</item>
/// </list>
/// </para>
/// <para>
/// This schedule pairs with Adam β₁=0.9, β₂=0.98, ε=1e-9 (the Vaswani 2017
/// hyperparameters): the small β₂ tracks rapidly-changing attention/embedding
/// gradients, and the warmup phase keeps the initial updates small while the
/// second-moment estimates have not yet stabilized. Without warmup, β₂=0.98
/// is too aggressive for early-training stability — which is why these values
/// must be applied together, not piecewise.
/// </para>
/// <para><b>For Beginners:</b> Transformers are sensitive to learning-rate
/// choice early in training because attention weights are softmax-normalized
/// and gradients can explode when the network has to figure out which tokens
/// to attend to. The Noam schedule ramps up the learning rate slowly for the
/// first few thousand steps (the "warmup"), then decreases it like the
/// inverse square root of the step number. This is how every Transformer in
/// the original 2017 paper was trained.
/// </para>
/// </remarks>
public class NoamSchedule : LearningRateSchedulerBase
{
    private readonly int _modelDimension;
    private readonly int _warmupSteps;
    private readonly double _factor;

    // Pre-computed step-invariant factors so ComputeLearningRate's per-batch
    // cost is two Math.Sqrt calls + a couple of multiplies, instead of three
    // Math.Pow calls. Math.Pow with non-integer exponents is materially
    // slower than Math.Sqrt on every modern .NET runtime (Pow goes through
    // exp(y · ln(x)); Sqrt has a dedicated SSE/AVX intrinsic). The schedule
    // fires once per batch, so even a few hundred ns saved per step adds up
    // across long training runs. Closes review-comment #1270.zzwx.
    //
    // _modelDimensionInvSqrt = d_model^(-0.5)            — paper's lr scale.
    // _warmupInvPow15        = warmup^(-1.5)             — paper's t · w^-1.5
    //                                                       term coefficient.
    // _scaledModelInvSqrt    = factor · d_model^(-0.5)   — multiplied into
    //                                                       the final return.
    // _scaledWarmupTerm      = _scaledModelInvSqrt · warmup^(-1.5)
    //                                                     — coefficient on
    //                                                       the linear (t · ...)
    //                                                       branch of the min.
    private readonly double _scaledModelInvSqrt;
    private readonly double _scaledWarmupTerm;

    /// <summary>
    /// Initializes a new Noam schedule (Vaswani 2017 inverse-sqrt with linear warmup).
    /// </summary>
    /// <param name="modelDimension">The Transformer's model dimension (d_model).</param>
    /// <param name="warmupSteps">Number of warmup steps. Default: 4000 (Vaswani 2017 §5.3).
    /// For small training budgets (less than a few thousand steps) consider lowering this
    /// to ~10% of the total step count so the schedule actually exits the warmup phase.</param>
    /// <param name="factor">Multiplicative scale on the schedule. Default: 1.0 (paper-faithful).
    /// Set higher to amplify the peak LR for tasks that benefit from a larger step size.</param>
    /// <exception cref="ArgumentException">Thrown when modelDimension or warmupSteps is non-positive.</exception>
    public NoamSchedule(int modelDimension, int warmupSteps = 4000, double factor = 1.0)
        : base(baseLearningRate: ComputePeakLr(modelDimension, warmupSteps, factor))
    {
        if (modelDimension <= 0)
            throw new ArgumentException("modelDimension must be positive.", nameof(modelDimension));
        if (warmupSteps <= 0)
            throw new ArgumentException("warmupSteps must be positive.", nameof(warmupSteps));
        if (factor <= 0)
            throw new ArgumentException("factor must be positive.", nameof(factor));

        _modelDimension = modelDimension;
        _warmupSteps = warmupSteps;
        _factor = factor;

        // Pre-compute the step-invariant coefficients. d_model^(-0.5) is
        // 1/sqrt(d_model); warmup^(-1.5) is 1/(warmup · sqrt(warmup)).
        double modelInvSqrt = 1.0 / Math.Sqrt(modelDimension);
        double warmupInvPow15 = 1.0 / ((double)warmupSteps * Math.Sqrt(warmupSteps));
        _scaledModelInvSqrt = factor * modelInvSqrt;
        _scaledWarmupTerm = _scaledModelInvSqrt * warmupInvPow15;

        // Step 0 (before any Step() call) maps to t=1 (warmup-start) under
        // the t=step+1 convention (industry-standard, matching tensor2tensor
        // and Hugging Face / PyTorch warmup schedulers driven from a
        // 0-indexed step counter). Without this override, the base ctor
        // leaves _currentLearningRate at _baseLearningRate (which we set to
        // the peak LR via ComputePeakLr to satisfy the base's positive-LR
        // guard) — causing the very first batch to use the peak LR instead
        // of the tiny warmup-start value.
        _currentLearningRate = ComputeLearningRate(0);
    }

    /// <summary>Number of warmup steps configured.</summary>
    public int WarmupSteps => _warmupSteps;

    /// <summary>Model dimension this schedule was configured for.</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Multiplicative scale on the schedule (1.0 = paper-faithful). Exposed so
    /// the fused-training path can reconstruct an equivalent
    /// <c>AiDotNet.Tensors.Engines.Compilation.LrSchedule.Noam(...)</c> and run
    /// Adam+Noam on the fused fast path with an identical per-step LR ramp.
    /// </summary>
    public double Factor => _factor;

    /// <inheritdoc />
    /// <remarks>
    /// <c>step</c> here is the library's "batches completed so far" counter
    /// (0-based), matching the value that <see cref="LearningRateSchedulerBase.Step"/>
    /// passes in after its post-batch increment. Internally we map to the
    /// paper's 1-indexed t via <c>t = step + 1</c>:
    /// <list type="bullet">
    ///   <item><c>step = 0</c> (no batches yet, ctor) → <c>t = 1</c> (warmup-start)</item>
    ///   <item><c>step = warmup_steps - 1</c> → <c>t = warmup_steps</c> (peak)</item>
    ///   <item><c>step = N</c> → <c>t = N + 1</c> (LR for the (N+1)th batch)</item>
    /// </list>
    /// Negative <c>step</c> is clamped to <c>t = 1</c> (the warmup-start
    /// value), keeping the formula away from a non-positive t that would
    /// produce <c>0^(-0.5) = ∞</c> or a negative <c>arg2</c> multiplier.
    /// The base class never passes a negative step at runtime; this guard
    /// is purely defensive against deserialized state.
    /// </remarks>
    protected override double ComputeLearningRate(int step)
    {
        // Industry-standard mapping (tensor2tensor / Hugging Face /
        // PyTorch): the framework's 0-indexed step → paper's 1-indexed t
        // via t = step + 1. The negative-step clamp is defensive — base
        // class never passes a negative step, but if a deserialized state
        // ever did, 0^(-0.5) = ∞ would otherwise leak through. Closes
        // review-comment #1269.yuXt (PR #1269 had the older
        // `step <= 0 ? 1 : step` mapping that collapsed steps 0 and 1
        // onto t=1; this branch always had t=step+1 so the cherry-pick
        // preserves the existing fix).
        int t = step < 0 ? 1 : step + 1;
        // Paper formula: lr(t) = factor · d_model^(-0.5) · min(t^(-0.5), t · warmup^(-1.5))
        // Pre-multiply factor · d_model^(-0.5) into both branches so the per-
        // step cost is one Math.Sqrt + a couple of multiplies, vs. three
        // Math.Pow calls in the original. arg1Branch = scaledModelInvSqrt /
        // sqrt(t); arg2Branch = (scaledModelInvSqrt · warmupInvPow15) · t.
        // The min and the t-branch crossover happen at t = warmup, exactly
        // matching the paper. Closes review-comment #1270.zzwx.
        double arg1Branch = _scaledModelInvSqrt / Math.Sqrt(t);
        double arg2Branch = _scaledWarmupTerm * t;
        return Math.Min(arg1Branch, arg2Branch);
    }

    /// <summary>
    /// Restores the scheduler to its initial state. The base implementation
    /// would set <c>_currentLearningRate = _baseLearningRate</c>, but Noam
    /// uses <c>_baseLearningRate</c> as a peak-LR sentinel (to satisfy the
    /// base ctor's positive-LR guard), so the default Reset would skip
    /// warmup on resume. Override to restore the warmup-start LR (t=1)
    /// instead — matching the post-ctor state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        // base.Reset() sets _currentStep = 0; mirror the ctor's
        // "warmup-start LR for batch 0" by passing step=0
        // (ComputeLearningRate maps step=0 → paper's t=1).
        _currentLearningRate = ComputeLearningRate(0);
    }

    private static double ComputePeakLr(int modelDimension, int warmupSteps, double factor)
    {
        if (modelDimension <= 0 || warmupSteps <= 0 || factor <= 0)
        {
            // Defer the validation error to the constructor so the message is
            // attributed to the right parameter; return a positive sentinel
            // so the base ctor's own positive-LR guard doesn't fire first.
            return 1.0;
        }
        return factor * Math.Pow(modelDimension, -0.5) * Math.Pow(warmupSteps, -0.5);
    }

    /// <inheritdoc />
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["model_dimension"] = _modelDimension;
        state["warmup_steps"] = _warmupSteps;
        state["factor"] = _factor;
        return state;
    }
}
