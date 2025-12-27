namespace AiDotNet.SelfSupervisedLearning.Infrastructure;

/// <summary>
/// Schedules temperature parameters during self-supervised learning training.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Temperature controls how "sharp" or "soft" the probability
/// distribution is in contrastive learning. Different methods schedule temperature differently
/// during training to improve learning dynamics.</para>
///
/// <para><b>Temperature effects:</b></para>
/// <list type="bullet">
/// <item><b>Low temperature (0.01-0.1):</b> Sharp distribution, hard negatives, more discriminative</item>
/// <item><b>High temperature (0.5-1.0):</b> Soft distribution, easier optimization, smoother gradients</item>
/// </list>
///
/// <para><b>Common scheduling strategies:</b></para>
/// <list type="bullet">
/// <item><b>Constant:</b> Fixed temperature throughout training (SimCLR, MoCo)</item>
/// <item><b>Linear warmup:</b> Start low, increase linearly (DINO teacher)</item>
/// <item><b>Cosine decay:</b> Start high, decay to final value</item>
/// </list>
/// </remarks>
public class TemperatureScheduler
{
    private readonly TemperatureScheduleType _scheduleType;
    private readonly double _initialTemperature;
    private readonly double _finalTemperature;
    private readonly int _warmupSteps;
    private readonly int _totalSteps;

    /// <summary>
    /// Gets the initial temperature value.
    /// </summary>
    public double InitialTemperature => _initialTemperature;

    /// <summary>
    /// Gets the final temperature value.
    /// </summary>
    public double FinalTemperature => _finalTemperature;

    /// <summary>
    /// Gets the schedule type being used.
    /// </summary>
    public TemperatureScheduleType ScheduleType => _scheduleType;

    /// <summary>
    /// Initializes a new instance of the TemperatureScheduler class.
    /// </summary>
    /// <param name="scheduleType">The type of schedule to use.</param>
    /// <param name="initialTemperature">Starting temperature value.</param>
    /// <param name="finalTemperature">Ending temperature value.</param>
    /// <param name="warmupSteps">Number of warmup steps (for warmup schedules).</param>
    /// <param name="totalSteps">Total training steps (for decay schedules).</param>
    public TemperatureScheduler(
        TemperatureScheduleType scheduleType = TemperatureScheduleType.Constant,
        double initialTemperature = 0.07,
        double finalTemperature = 0.07,
        int warmupSteps = 0,
        int totalSteps = 100000)
    {
        if (initialTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(initialTemperature), "Temperature must be positive");
        if (finalTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(finalTemperature), "Temperature must be positive");
        if (warmupSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(warmupSteps), "Warmup steps cannot be negative");
        if (totalSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalSteps), "Total steps must be positive");

        _scheduleType = scheduleType;
        _initialTemperature = initialTemperature;
        _finalTemperature = finalTemperature;
        _warmupSteps = warmupSteps;
        _totalSteps = totalSteps;
    }

    /// <summary>
    /// Gets the temperature value for the current training step.
    /// </summary>
    /// <param name="currentStep">The current training step.</param>
    /// <returns>The scheduled temperature value.</returns>
    public double GetTemperature(int currentStep)
    {
        if (currentStep < 0) currentStep = 0;

        return _scheduleType switch
        {
            TemperatureScheduleType.Constant => _initialTemperature,
            TemperatureScheduleType.LinearWarmup => GetLinearWarmupTemperature(currentStep),
            TemperatureScheduleType.CosineDecay => GetCosineDecayTemperature(currentStep),
            TemperatureScheduleType.LinearDecay => GetLinearDecayTemperature(currentStep),
            TemperatureScheduleType.ExponentialDecay => GetExponentialDecayTemperature(currentStep),
            TemperatureScheduleType.CosineWarmup => GetCosineWarmupTemperature(currentStep),
            _ => _initialTemperature
        };
    }

    /// <summary>
    /// Gets the temperature value for the current epoch.
    /// </summary>
    /// <param name="currentEpoch">The current training epoch.</param>
    /// <param name="totalEpochs">Total training epochs.</param>
    /// <returns>The scheduled temperature value.</returns>
    public double GetTemperatureForEpoch(int currentEpoch, int totalEpochs)
    {
        if (totalEpochs <= 0) return _finalTemperature;

        var progress = Math.Min(1.0, Math.Max(0.0, (double)currentEpoch / totalEpochs));
        var step = (int)(progress * _totalSteps);
        return GetTemperature(step);
    }

    private double GetLinearWarmupTemperature(int step)
    {
        if (step >= _warmupSteps)
        {
            // After warmup, use final temperature or decay
            if (_warmupSteps >= _totalSteps)
                return _finalTemperature;

            var postWarmupProgress = (double)(step - _warmupSteps) / (_totalSteps - _warmupSteps);
            return _initialTemperature + (_finalTemperature - _initialTemperature) * postWarmupProgress;
        }

        // During warmup: linear interpolation from initial to final
        var warmupProgress = (double)step / _warmupSteps;
        return _initialTemperature + (_finalTemperature - _initialTemperature) * warmupProgress;
    }

    private double GetCosineDecayTemperature(int step)
    {
        var progress = Math.Min(1.0, (double)step / _totalSteps);

        // Cosine decay from initial to final
        var cosineProgress = (1.0 - Math.Cos(Math.PI * progress)) / 2.0;
        return _initialTemperature + (_finalTemperature - _initialTemperature) * cosineProgress;
    }

    private double GetLinearDecayTemperature(int step)
    {
        var progress = Math.Min(1.0, (double)step / _totalSteps);
        return _initialTemperature + (_finalTemperature - _initialTemperature) * progress;
    }

    private double GetExponentialDecayTemperature(int step)
    {
        if (_initialTemperature == _finalTemperature) return _initialTemperature;

        var progress = Math.Min(1.0, (double)step / _totalSteps);

        // Exponential decay: initial * (final/initial)^progress
        var ratio = _finalTemperature / _initialTemperature;
        return _initialTemperature * Math.Pow(ratio, progress);
    }

    private double GetCosineWarmupTemperature(int step)
    {
        if (step >= _warmupSteps)
        {
            return _finalTemperature;
        }

        // Cosine warmup from initial to final
        var warmupProgress = (double)step / _warmupSteps;
        var cosineProgress = (1.0 - Math.Cos(Math.PI * warmupProgress)) / 2.0;
        return _initialTemperature + (_finalTemperature - _initialTemperature) * cosineProgress;
    }

    /// <summary>
    /// Creates a scheduler for constant temperature (SimCLR, MoCo default).
    /// </summary>
    /// <param name="temperature">The constant temperature value.</param>
    public static TemperatureScheduler Constant(double temperature = 0.07)
    {
        return new TemperatureScheduler(
            TemperatureScheduleType.Constant,
            initialTemperature: temperature,
            finalTemperature: temperature);
    }

    /// <summary>
    /// Creates a scheduler for DINO teacher temperature warmup.
    /// </summary>
    /// <param name="initialTemperature">Starting temperature (default: 0.04).</param>
    /// <param name="finalTemperature">Final temperature (default: 0.07).</param>
    /// <param name="warmupEpochs">Number of warmup epochs (default: 30).</param>
    /// <param name="totalEpochs">Total training epochs.</param>
    public static TemperatureScheduler ForDINOTeacher(
        double initialTemperature = 0.04,
        double finalTemperature = 0.07,
        int warmupEpochs = 30,
        int totalEpochs = 100)
    {
        return new TemperatureScheduler(
            TemperatureScheduleType.LinearWarmup,
            initialTemperature: initialTemperature,
            finalTemperature: finalTemperature,
            warmupSteps: warmupEpochs * 1000,  // Approximate steps
            totalSteps: totalEpochs * 1000);
    }

    /// <summary>
    /// Creates a scheduler with cosine annealing from high to low temperature.
    /// </summary>
    /// <param name="highTemperature">Starting (high) temperature.</param>
    /// <param name="lowTemperature">Final (low) temperature.</param>
    /// <param name="totalSteps">Total training steps.</param>
    public static TemperatureScheduler CosineAnneal(
        double highTemperature = 0.5,
        double lowTemperature = 0.07,
        int totalSteps = 100000)
    {
        return new TemperatureScheduler(
            TemperatureScheduleType.CosineDecay,
            initialTemperature: highTemperature,
            finalTemperature: lowTemperature,
            totalSteps: totalSteps);
    }
}

/// <summary>
/// Types of temperature scheduling strategies.
/// </summary>
public enum TemperatureScheduleType
{
    /// <summary>
    /// Constant temperature throughout training.
    /// </summary>
    Constant,

    /// <summary>
    /// Linear warmup from initial to final temperature.
    /// </summary>
    LinearWarmup,

    /// <summary>
    /// Cosine decay from initial to final temperature.
    /// </summary>
    CosineDecay,

    /// <summary>
    /// Linear decay from initial to final temperature.
    /// </summary>
    LinearDecay,

    /// <summary>
    /// Exponential decay from initial to final temperature.
    /// </summary>
    ExponentialDecay,

    /// <summary>
    /// Cosine warmup from initial to final temperature.
    /// </summary>
    CosineWarmup
}
