namespace AiDotNet.Statistics;


/// <summary>
/// Base class for prediction statistics that combine model statistics and intervals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[Serializable]
public abstract class PredictionStatisticsBase<T> : ModelStatisticsBase<T>, IPredictionStatistics<T>, IIntervalProvider<T>
{
    /// <summary>
    /// Dictionary to store calculated intervals as tuples.
    /// </summary>
    protected readonly Dictionary<IntervalType, (T Lower, T Upper)> _intervals = [];

    /// <summary>
    /// Set of intervals that have been calculated.
    /// </summary>
    protected readonly HashSet<IntervalType> _calculatedIntervals = [];

    /// <summary>
    /// Set of intervals valid for this provider.
    /// </summary>
    protected readonly HashSet<IntervalType> _validIntervals = [];

    /// <summary>
    /// The confidence level used for statistical intervals.
    /// </summary>
    protected readonly T _confidenceLevel;

    /// <summary>
    /// Gets the type of prediction (regression, classification, etc.).
    /// </summary>
    public PredictionType PredictionType { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="PredictionStatisticsBase{T}"/> class.
    /// </summary>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="featureCount">The number of features or parameters in the model.</param>
    /// <param name="predictionType">The type of prediction.</param>
    /// <param name="confidenceLevel">The confidence level to use for intervals (typically 0.95).</param>
    protected PredictionStatisticsBase(ModelType modelType, int featureCount,
                                      PredictionType predictionType, double confidenceLevel = 0.95)
        : base(modelType, featureCount)
    {
        PredictionType = predictionType;
        _confidenceLevel = _numOps.FromDouble(confidenceLevel);

        // Determine which intervals are valid for this prediction type
        DetermineValidIntervals();
    }

    /// <summary>
    /// Determines which intervals are valid for this prediction type.
    /// </summary>
    protected virtual void DetermineValidIntervals()
    {
        var allIntervalTypes = Enum.GetValues(typeof(IntervalType)).Cast<IntervalType>();
        foreach (var intervalType in allIntervalTypes)
        {
            if (IsValidIntervalForModelType(intervalType, ModelType) &&
                IsPredictionIntervalType(intervalType, PredictionType))
            {
                _validIntervals.Add(intervalType);
            }
        }
    }

    /// <summary>
    /// Determines if an interval type is valid for a specific model type.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <param name="modelType">The type of model.</param>
    /// <returns>True if the interval is valid for the model type; otherwise, false.</returns>
    protected abstract bool IsValidIntervalForModelType(IntervalType intervalType, ModelType modelType);

    /// <summary>
    /// Determines if an interval type is valid for a specific prediction type.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <param name="predictionType">The type of prediction.</param>
    /// <returns>True if the interval is valid for the prediction type; otherwise, false.</returns>
    protected abstract bool IsPredictionIntervalType(IntervalType intervalType, PredictionType predictionType);

    #region IIntervalProvider Implementation

    /// <summary>
    /// Gets an interval by type.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <returns>The interval as a tuple of (Lower, Upper) bounds.</returns>
    public virtual (T Lower, T Upper) GetInterval(IntervalType intervalType)
    {
        return _intervals.TryGetValue(intervalType, out var interval) ? interval : (_numOps.Zero, _numOps.Zero);
    }

    /// <summary>
    /// Tries to get a specific interval.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <param name="interval">The interval as a tuple of (Lower, Upper) bounds if successful.</param>
    /// <returns>True if the interval was successfully retrieved; otherwise, false.</returns>
    public virtual bool TryGetInterval(IntervalType intervalType, out (T Lower, T Upper) interval)
    {
        if (!IsValidInterval(intervalType))
        {
            interval = (_numOps.Zero, _numOps.Zero);
            return false;
        }

        return _intervals.TryGetValue(intervalType, out interval);
    }

    /// <summary>
    /// Checks if a specific interval is valid for this provider.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval is valid; otherwise, false.</returns>
    public virtual bool IsValidInterval(IntervalType intervalType)
    {
        return _validIntervals.Contains(intervalType);
    }

    /// <summary>
    /// Checks if a specific interval has been calculated.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval has been calculated; otherwise, false.</returns>
    public virtual bool IsCalculatedInterval(IntervalType intervalType)
    {
        return _calculatedIntervals.Contains(intervalType);
    }

    /// <summary>
    /// Gets all interval types that are valid for this provider.
    /// </summary>
    /// <returns>An array of valid interval types.</returns>
    public virtual IntervalType[] GetValidIntervalTypes()
    {
        return [.. _validIntervals];
    }

    /// <summary>
    /// Gets all interval types that have been calculated.
    /// </summary>
    /// <returns>An array of calculated interval types.</returns>
    public virtual IntervalType[] GetCalculatedIntervalTypes()
    {
        return [.. _calculatedIntervals];
    }

    #endregion
}