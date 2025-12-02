namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Implements cyclical learning rate policy.
/// </summary>
/// <remarks>
/// <para>
/// CyclicLR cycles the learning rate between two boundaries with a constant frequency.
/// This approach can help escape local minima and find better solutions by periodically
/// increasing the learning rate.
/// </para>
/// <para><b>For Beginners:</b> Instead of always decreasing the learning rate, cyclic learning
/// rates go up and down in cycles. The idea is that periodically increasing the learning rate
/// can help the model escape local minima (suboptimal solutions) and explore better solutions.
/// Think of it like occasionally taking bigger jumps while hiking to avoid getting stuck in small valleys.
/// </para>
/// <para>
/// Based on the paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Triangular mode cycling between 0.001 and 0.1
/// var scheduler = new CyclicLRScheduler(
///     baseLearningRate: 0.001,
///     maxLearningRate: 0.1,
///     stepSizeUp: 2000,
///     mode: CyclicLRScheduler.CyclicMode.Triangular
/// );
/// </code>
/// </example>
public class CyclicLRScheduler : LearningRateSchedulerBase
{
    private readonly double _maxLearningRate;
    private readonly int _stepSizeUp;
    private readonly int _stepSizeDown;
    private readonly CyclicMode _mode;
    private readonly double _gamma;

    private int _cycleCount;

    /// <summary>
    /// Mode for cyclic learning rate.
    /// </summary>
    public enum CyclicMode
    {
        /// <summary>Basic triangular cycle</summary>
        Triangular,
        /// <summary>Triangular cycle with amplitude halved each cycle</summary>
        Triangular2,
        /// <summary>Scales amplitude by gamma^cycle</summary>
        ExponentialRange
    }

    /// <summary>
    /// Initializes a new instance of the CyclicLRScheduler class.
    /// </summary>
    /// <param name="baseLearningRate">Minimum learning rate.</param>
    /// <param name="maxLearningRate">Maximum learning rate.</param>
    /// <param name="stepSizeUp">Number of training iterations in the increasing half of a cycle. Default: 2000</param>
    /// <param name="stepSizeDown">Number of training iterations in the decreasing half. Default: same as stepSizeUp</param>
    /// <param name="mode">Cycling mode. Default: Triangular</param>
    /// <param name="gamma">Constant for 'exp_range' mode, scales amplitude by gamma^cycle. Default: 1.0</param>
    public CyclicLRScheduler(
        double baseLearningRate,
        double maxLearningRate,
        int stepSizeUp = 2000,
        int? stepSizeDown = null,
        CyclicMode mode = CyclicMode.Triangular,
        double gamma = 1.0)
        : base(baseLearningRate)
    {
        if (maxLearningRate <= baseLearningRate)
            throw new ArgumentException("Max learning rate must be greater than base learning rate.", nameof(maxLearningRate));
        if (stepSizeUp <= 0)
            throw new ArgumentException("Step size up must be positive.", nameof(stepSizeUp));
        if (gamma <= 0 || gamma > 1)
            throw new ArgumentException("Gamma must be in (0, 1].", nameof(gamma));

        _maxLearningRate = maxLearningRate;
        _stepSizeUp = stepSizeUp;
        _stepSizeDown = stepSizeDown ?? stepSizeUp;
        _mode = mode;
        _gamma = gamma;
        _cycleCount = 0;
    }

    /// <summary>
    /// Gets the maximum learning rate.
    /// </summary>
    public double MaxLearningRate => _maxLearningRate;

    /// <summary>
    /// Gets the step size for increasing phase.
    /// </summary>
    public int StepSizeUp => _stepSizeUp;

    /// <summary>
    /// Gets the step size for decreasing phase.
    /// </summary>
    public int StepSizeDown => _stepSizeDown;

    /// <summary>
    /// Gets the current cycle count.
    /// </summary>
    public int CycleCount => _cycleCount;

    /// <inheritdoc/>
    protected override double ComputeLearningRate(int step)
    {
        int cycleLength = _stepSizeUp + _stepSizeDown;
        int cycle = step / cycleLength;
        int cyclePosition = step % cycleLength;

        double scale;
        if (cyclePosition < _stepSizeUp)
        {
            // Increasing phase
            scale = (double)cyclePosition / _stepSizeUp;
        }
        else
        {
            // Decreasing phase
            scale = 1.0 - (double)(cyclePosition - _stepSizeUp) / _stepSizeDown;
        }

        double amplitude = _maxLearningRate - _baseLearningRate;

        switch (_mode)
        {
            case CyclicMode.Triangular:
                return _baseLearningRate + amplitude * scale;

            case CyclicMode.Triangular2:
                amplitude = amplitude / Math.Pow(2, cycle);
                return _baseLearningRate + amplitude * scale;

            case CyclicMode.ExponentialRange:
                amplitude = amplitude * Math.Pow(_gamma, step);
                return _baseLearningRate + amplitude * scale;

            default:
                return _baseLearningRate + amplitude * scale;
        }
    }

    /// <inheritdoc/>
    public override Dictionary<string, object> GetState()
    {
        var state = base.GetState();
        state["max_learning_rate"] = _maxLearningRate;
        state["step_size_up"] = _stepSizeUp;
        state["step_size_down"] = _stepSizeDown;
        state["mode"] = _mode.ToString();
        state["gamma"] = _gamma;
        state["cycle_count"] = _cycleCount;
        return state;
    }
}
