namespace AiDotNet.Statistics;

using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;

/// <summary>
/// Base class for interval-providing statistics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[Serializable]
public abstract class IntervalProviderBase<T> : StatisticsBase<T>, IIntervalProvider<T>
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
    /// Initializes a new instance of the <see cref="IntervalProviderBase{T}"/> class.
    /// </summary>
    /// <param name="confidenceLevel">The confidence level to use for intervals (typically 0.95).</param>
    protected IntervalProviderBase(ModelType modelType, double confidenceLevel = 0.95)
        : base(modelType)
    {
        _confidenceLevel = _numOps.FromDouble(confidenceLevel);

        // Determine which intervals are valid for this provider
        DetermineValidIntervals();
    }

    /// <summary>
    /// Determines which intervals are valid for this provider.
    /// </summary>
    protected abstract void DetermineValidIntervals();

    /// <summary>
    /// Gets an interval by type.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <returns>The interval as a tuple of (Lower, Upper) bounds.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the requested interval is not valid.</exception>
    public virtual (T Lower, T Upper) GetInterval(IntervalType intervalType)
    {
        if (!IsValidInterval(intervalType))
        {
            throw new InvalidOperationException($"Interval {intervalType} is not valid for this provider.");
        }

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
}