namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Warmup-Stable-Decay (WSD) learning-rate schedule: a linear warmup, a long constant ("stable") phase, and a
/// final decay to <c>endLr</c>.
/// </summary>
/// <remarks>
/// <para>
/// Unlike cosine annealing, nothing in the warmup or stable phase depends on the total run length, so a run that
/// is checkpointed during the stable phase can be <b>extended</b> by resuming it with a larger
/// <c>decayStartStep</c>. Language-model pretraining uses this to read quality at several token budgets from one
/// stable trajectory: each budget branches a short decay from a stable checkpoint.
/// </para>
/// <para><b>For Beginners:</b> The learning rate ramps up, holds steady for most of training, then drops to
/// (almost) zero at the end. Because the "hold steady" part doesn't know when training will end, you can decide
/// later to train longer without restarting from scratch.</para>
/// <para><b>Reference:</b> Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable
/// Training Strategies", 2024 (https://arxiv.org/abs/2404.06395).</para>
/// </remarks>
/// <example>
/// <code>
/// // 500 warmup steps, stable until step 8000, linear decay to 0 by step 10000.
/// var wsd = new WarmupStableDecayScheduler(3e-3, warmupSteps: 500, decayStartStep: 8000, decaySteps: 2000);
/// </code>
/// </example>
public class WarmupStableDecayScheduler : LearningRateSchedulerBase
{
    private readonly int _warmupSteps;
    private readonly int _decayStartStep;
    private readonly int _decaySteps;
    private readonly DecayShape _decayShape;
    private readonly double _endLr;

    /// <summary>
    /// Shape of the final decay phase.
    /// </summary>
    public enum DecayShape
    {
        /// <summary>Linear decay from the base rate to <c>endLr</c>.</summary>
        Linear,

        /// <summary>Half-cosine decay from the base rate to <c>endLr</c>.</summary>
        Cosine,

        /// <summary>1 - sqrt(progress) decay (the MiniCPM / "1-sqrt" cooldown).</summary>
        OneMinusSqrt
    }

    /// <summary>
    /// Creates a WSD schedule.
    /// </summary>
    /// <param name="baseLearningRate">Peak (stable-phase) learning rate.</param>
    /// <param name="warmupSteps">Steps of linear warmup from 0 to <paramref name="baseLearningRate"/>.</param>
    /// <param name="decayStartStep">
    /// Step at which the decay begins. Use <see cref="int.MaxValue"/> for a stable-only run that will be
    /// decayed later by resuming with a finite value.
    /// </param>
    /// <param name="decaySteps">Length of the decay phase.</param>
    /// <param name="decayShape">Shape of the decay (default linear).</param>
    /// <param name="endLr">Learning rate at the end of the decay (default 0).</param>
    public WarmupStableDecayScheduler(
        double baseLearningRate,
        int warmupSteps,
        int decayStartStep,
        int decaySteps,
        DecayShape decayShape = DecayShape.Linear,
        double endLr = 0.0)
        : base(baseLearningRate, 0.0)
    {
        if (warmupSteps < 0)
            throw new ArgumentException("Warmup steps cannot be negative.", nameof(warmupSteps));
        if (decayStartStep < warmupSteps)
            throw new ArgumentException("Decay cannot start before warmup ends.", nameof(decayStartStep));
        if (decaySteps < 0)
            throw new ArgumentException("Decay steps cannot be negative.", nameof(decaySteps));
        if (endLr < 0 || endLr > baseLearningRate)
            throw new ArgumentException("End learning rate must lie in [0, baseLearningRate].", nameof(endLr));

        _warmupSteps = warmupSteps;
        _decayStartStep = decayStartStep;
        _decaySteps = decaySteps;
        _decayShape = decayShape;
        _endLr = endLr;
        _currentLearningRate = ComputeLearningRate(0);
    }

    /// <summary>Gets the number of warmup steps.</summary>
    public int WarmupSteps => _warmupSteps;

    /// <summary>Gets the step at which the decay begins.</summary>
    public int DecayStartStep => _decayStartStep;

    /// <summary>Gets the length of the decay phase.</summary>
    public int DecaySteps => _decaySteps;

    /// <summary>Gets the decay shape.</summary>
    public DecayShape Shape => _decayShape;

    /// <summary>Gets the final learning rate.</summary>
    public double EndLr => _endLr;

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _currentLearningRate = ComputeLearningRate(0);
    }

    /// <inheritdoc />
    protected override double ComputeLearningRate(int step)
    {
        if (step < _warmupSteps)
        {
            return _baseLearningRate * step / _warmupSteps;
        }

        if (step < _decayStartStep)
        {
            return _baseLearningRate;
        }

        if (_decaySteps == 0)
        {
            return _endLr;
        }

        double progress = Math.Min(1.0, (double)(step - _decayStartStep) / _decaySteps);
        double factor = _decayShape switch
        {
            DecayShape.Cosine => (1 + Math.Cos(Math.PI * progress)) / 2,
            DecayShape.OneMinusSqrt => 1 - Math.Sqrt(progress),
            _ => 1 - progress
        };
        return _endLr + (_baseLearningRate - _endLr) * factor;
    }

    /// <inheritdoc />
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["warmup_steps"] = _warmupSteps;
        state["decay_start_step"] = _decayStartStep;
        state["decay_steps"] = _decaySteps;
        state["decay_shape"] = _decayShape.ToString();
        state["end_lr"] = _endLr;
        return state;
    }
}
