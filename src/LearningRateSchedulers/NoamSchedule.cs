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
/// where t is the 1-indexed training step. The schedule rises linearly during
/// the warmup phase (steps 1 .. warmup), peaks at step = warmup with value
/// <c>factor · d_model^(-0.5) · warmup^(-0.5)</c>, then decays as t^(-0.5).
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

        // Step 0 (before any Step() call) — start at the t=1 value, not the
        // base/peak LR. Otherwise the first Train() call before the scheduler
        // is stepped would use the peak LR, contradicting the "warmup from
        // tiny" semantic.
        _currentLearningRate = ComputeLearningRate(1);
    }

    /// <summary>Number of warmup steps configured.</summary>
    public int WarmupSteps => _warmupSteps;

    /// <summary>Model dimension this schedule was configured for.</summary>
    public int ModelDimension => _modelDimension;

    /// <inheritdoc />
    protected override double ComputeLearningRate(int step)
    {
        // Vaswani uses 1-indexed steps. Guard against step=0 producing 0^-0.5 = ∞.
        int t = step <= 0 ? 1 : step;
        double arg1 = Math.Pow(t, -0.5);
        double arg2 = t * Math.Pow(_warmupSteps, -1.5);
        return _factor * Math.Pow(_modelDimension, -0.5) * Math.Min(arg1, arg2);
    }

    /// <summary>
    /// Restores the scheduler to its initial state. The base implementation
    /// would set <c>_currentLearningRate = _baseLearningRate</c>, but Noam
    /// uses <c>_baseLearningRate</c> as a peak-LR sentinel (to satisfy the
    /// base ctor's positive-LR guard at <c>ComputePeakLr</c>), so the default
    /// <c>Reset()</c> would skip warmup and jump straight to the peak LR on
    /// any subsequent training run. Override to restore the warmup-start LR
    /// (t=1) instead, matching the post-ctor state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _currentLearningRate = ComputeLearningRate(1);
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
