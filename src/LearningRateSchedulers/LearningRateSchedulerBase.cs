namespace AiDotNet.LearningRateSchedulers;

/// <summary>
/// Base class for learning rate schedulers providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract base class implements the common behavior for all learning rate schedulers,
/// including state management, step tracking, and serialization support.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all learning rate schedulers build upon.
/// It handles the common tasks like keeping track of what step we're on and saving/loading state
/// so that training can be resumed from a checkpoint.
/// </para>
/// </remarks>
public abstract class LearningRateSchedulerBase : ILearningRateScheduler
{
    /// <summary>
    /// The base (initial) learning rate.
    /// </summary>
    protected double _baseLearningRate;

    /// <summary>
    /// The current learning rate.
    /// </summary>
    protected double _currentLearningRate;

    /// <summary>
    /// The current step count.
    /// </summary>
    protected int _currentStep;

    /// <summary>
    /// The minimum learning rate (floor).
    /// </summary>
    protected double _minLearningRate;

    /// <summary>
    /// Initializes a new instance of the LearningRateSchedulerBase class.
    /// </summary>
    /// <param name="baseLearningRate">The initial learning rate.</param>
    /// <param name="minLearningRate">The minimum learning rate (floor). Default is 0.</param>
    protected LearningRateSchedulerBase(double baseLearningRate, double minLearningRate = 0.0)
    {
        if (baseLearningRate <= 0)
            throw new ArgumentException("Base learning rate must be positive.", nameof(baseLearningRate));
        if (minLearningRate < 0)
            throw new ArgumentException("Minimum learning rate cannot be negative.", nameof(minLearningRate));

        _baseLearningRate = baseLearningRate;
        _currentLearningRate = baseLearningRate;
        _minLearningRate = minLearningRate;
        _currentStep = 0;
    }

    /// <inheritdoc/>
    public double CurrentLearningRate => _currentLearningRate;

    /// <inheritdoc/>
    public double BaseLearningRate => _baseLearningRate;

    /// <inheritdoc/>
    public int CurrentStep => _currentStep;

    /// <inheritdoc/>
    public virtual double Step()
    {
        _currentStep++;
        _currentLearningRate = Math.Max(_minLearningRate, ComputeLearningRate(_currentStep));
        return _currentLearningRate;
    }

    /// <inheritdoc/>
    public virtual double GetLearningRateAtStep(int step)
    {
        if (step < 0)
            throw new ArgumentException("Step cannot be negative.", nameof(step));
        return Math.Max(_minLearningRate, ComputeLearningRate(step));
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        _currentStep = 0;
        _currentLearningRate = _baseLearningRate;
    }

    /// <summary>
    /// Computes the learning rate for a given step.
    /// </summary>
    /// <param name="step">The step number.</param>
    /// <returns>The computed learning rate.</returns>
    protected abstract double ComputeLearningRate(int step);

    /// <inheritdoc/>
    public virtual Dictionary<string, object> GetState()
    {
        return new Dictionary<string, object>
        {
            ["base_lr"] = _baseLearningRate,
            ["current_lr"] = _currentLearningRate,
            ["current_step"] = _currentStep,
            ["min_learning_rate"] = _minLearningRate
        };
    }

    /// <inheritdoc/>
    public virtual void LoadState(Dictionary<string, object> state)
    {
        // Support both new keys (base_lr, current_lr) and legacy keys (base_learning_rate, current_learning_rate)
        // for backward compatibility with previously serialized checkpoints
        if (state.TryGetValue("base_lr", out var baseLr))
            _baseLearningRate = Convert.ToDouble(baseLr);
        else if (state.TryGetValue("base_learning_rate", out var legacyBaseLr))
            _baseLearningRate = Convert.ToDouble(legacyBaseLr);

        if (state.TryGetValue("current_lr", out var currentLr))
            _currentLearningRate = Convert.ToDouble(currentLr);
        else if (state.TryGetValue("current_learning_rate", out var legacyCurrentLr))
            _currentLearningRate = Convert.ToDouble(legacyCurrentLr);

        if (state.TryGetValue("current_step", out var step))
            _currentStep = Convert.ToInt32(step);
        if (state.TryGetValue("min_learning_rate", out var minLr))
            _minLearningRate = Convert.ToDouble(minLr);
    }
}
