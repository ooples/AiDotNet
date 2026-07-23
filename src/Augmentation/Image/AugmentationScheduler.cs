namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Schedules augmentation strength changes during training.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AugmentationScheduler<T>
{
    private readonly IAugmentation<T, ImageTensor<T>> _augmenter;
    private readonly ScheduleType _scheduleType;
    private readonly int _totalEpochs;
    private readonly double _startStrength;
    private readonly double _endStrength;
    private int _currentEpoch;

    public AugmentationScheduler(IAugmentation<T, ImageTensor<T>> augmenter,
        ScheduleType scheduleType = ScheduleType.Linear,
        int totalEpochs = 100,
        double startStrength = 0.0,
        double endStrength = 1.0)
    {
        _augmenter = augmenter;
        _scheduleType = scheduleType;
        _totalEpochs = totalEpochs;
        _startStrength = startStrength;
        _endStrength = endStrength;
    }

    /// <summary>
    /// Gets the current augmentation strength factor [0, 1].
    /// </summary>
    public double CurrentStrength
    {
        get
        {
            double progress = (double)_currentEpoch / Math.Max(1, _totalEpochs);
            progress = Math.Max(0, Math.Min(1, progress));

            return _scheduleType switch
            {
                ScheduleType.Linear => _startStrength + (_endStrength - _startStrength) * progress,
                ScheduleType.Cosine => _startStrength + (_endStrength - _startStrength) *
                                       (1 - Math.Cos(progress * Math.PI)) / 2,
                ScheduleType.Step => progress >= 0.5 ? _endStrength : _startStrength,
                ScheduleType.Exponential => _startStrength + (_endStrength - _startStrength) *
                                            (Math.Exp(progress * 3) - 1) / (Math.Exp(3) - 1),
                _ => _startStrength
            };
        }
    }

    /// <summary>
    /// Updates the scheduler to the next epoch.
    /// </summary>
    public void Step()
    {
        _currentEpoch++;
    }

    /// <summary>
    /// Sets the current epoch.
    /// </summary>
    public void SetEpoch(int epoch)
    {
        _currentEpoch = epoch;
    }

    /// <summary>
    /// Gets the underlying augmenter.
    /// </summary>
    public IAugmentation<T, ImageTensor<T>> Augmenter => _augmenter;
}

/// <summary>
/// Schedule type for augmentation strength.
/// </summary>
public enum ScheduleType
{
    /// <summary>Linear interpolation from start to end.</summary>
    Linear,
    /// <summary>Cosine annealing schedule.</summary>
    Cosine,
    /// <summary>Step function at midpoint.</summary>
    Step,
    /// <summary>Exponential ramp-up.</summary>
    Exponential
}
