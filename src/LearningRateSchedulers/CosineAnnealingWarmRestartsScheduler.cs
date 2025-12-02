namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Sets the learning rate using cosine annealing with warm restarts.
/// </summary>
/// <remarks>
/// <para>
/// This scheduler implements the SGDR (Stochastic Gradient Descent with Warm Restarts) algorithm.
/// It uses cosine annealing but periodically restarts the learning rate to the initial value,
/// optionally increasing the period between restarts.
/// </para>
/// <para><b>For Beginners:</b> Imagine running a race in sprints instead of one continuous run.
/// After each sprint (cycle), you rest (restart learning rate) and then sprint again. This "warm restart"
/// approach helps the model escape local minima and often finds better solutions. The sprints can
/// optionally get longer each time (controlled by T_mult), allowing for more fine-tuning in later cycles.
/// </para>
/// <para>
/// Based on the paper "SGDR: Stochastic Gradient Descent with Warm Restarts" by Loshchilov &amp; Hutter.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Warm restarts with initial period of 10, doubling each cycle
/// var scheduler = new CosineAnnealingWarmRestartsScheduler(
///     baseLearningRate: 0.1,
///     t0: 10,
///     tMult: 2,
///     etaMin: 0.001
/// );
/// </code>
/// </example>
public class CosineAnnealingWarmRestartsScheduler : LearningRateSchedulerBase
{
    private readonly int _t0;
    private readonly int _tMult;
    private readonly double _etaMin;

    private int _currentCycle;
    private int _cycleStep;
    private int _currentT;

    /// <summary>
    /// Initializes a new instance of the CosineAnnealingWarmRestartsScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">The initial (maximum) learning rate.</param>
    /// <param name="t0">Number of steps for the first restart.</param>
    /// <param name="tMult">Factor to increase T after each restart. Default: 1 (constant period)</param>
    /// <param name="etaMin">Minimum learning rate. Default: 0</param>
    /// <exception cref="ArgumentException">Thrown when t0 is not positive or tMult is less than 1.</exception>
    public CosineAnnealingWarmRestartsScheduler(
        double baseLearningRate,
        int t0,
        int tMult = 1,
        double etaMin = 0.0)
        : base(baseLearningRate, etaMin)
    {
        if (t0 <= 0)
            throw new ArgumentException("T_0 must be positive.", nameof(t0));
        if (tMult < 1)
            throw new ArgumentException("T_mult must be >= 1.", nameof(tMult));

        _t0 = t0;
        _tMult = tMult;
        _etaMin = etaMin;
        _currentT = t0;
        _currentCycle = 0;
        _cycleStep = 0;
    }

    /// <summary>
    /// Gets the initial period.
    /// </summary>
    public int T0 => _t0;

    /// <summary>
    /// Gets the period multiplier.
    /// </summary>
    public int TMult => _tMult;

    /// <summary>
    /// Gets the minimum learning rate.
    /// </summary>
    public double EtaMin => _etaMin;

    /// <summary>
    /// Gets the current cycle number.
    /// </summary>
    public int CurrentCycle => _currentCycle;

    /// <inheritdoc/>
    public override double Step()
    {
        _currentStep++;
        _cycleStep++;

        // Check if we need to restart
        if (_cycleStep >= _currentT)
        {
            _currentCycle++;
            _cycleStep = 0;
            _currentT = _t0 * (int)Math.Pow(_tMult, _currentCycle);
        }

        _currentLearningRate = ComputeLearningRate(_currentStep);
        return _currentLearningRate;
    }

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        // Compute cycle and position within cycle
        int cycle = 0;
        int t = _t0;
        int accumulated = 0;

        while (accumulated + t <= step)
        {
            accumulated += t;
            cycle++;
            t = _t0 * (int)Math.Pow(_tMult, cycle);
        }

        int cyclePosition = step - accumulated;
        int currentPeriod = _t0 * (int)Math.Pow(_tMult, cycle);

        double cosineValue = Math.Cos(Math.PI * cyclePosition / currentPeriod);
        return _etaMin + 0.5 * (_baseLearningRate - _etaMin) * (1 + cosineValue);
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _currentCycle = 0;
        _cycleStep = 0;
        _currentT = _t0;
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["t0"] = _t0;
        state["t_mult"] = _tMult;
        state["eta_min"] = _etaMin;
        state["current_cycle"] = _currentCycle;
        state["cycle_step"] = _cycleStep;
        state["current_t"] = _currentT;
        return state;
    }

    /// <inheritdoc/>
    public override void LoadState(Dictionary<string, object> state)
    {
        base.LoadState(state);
        if (state.TryGetValue("current_cycle", out var cycle))
            _currentCycle = Convert.ToInt32(cycle);
        if (state.TryGetValue("cycle_step", out var cycleStep))
            _cycleStep = Convert.ToInt32(cycleStep);
        if (state.TryGetValue("current_t", out var currentT))
            _currentT = Convert.ToInt32(currentT);
    }
}
