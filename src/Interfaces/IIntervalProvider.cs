using AiDotNet.Enums;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for statistics providers that support prediction intervals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IIntervalProvider<T>
{
    /// <summary>
    /// Gets an interval by type.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <returns>The interval as a tuple of (Lower, Upper) bounds.</returns>
    (T Lower, T Upper) GetInterval(IntervalType intervalType);

    /// <summary>
    /// Tries to get a specific interval.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <param name="interval">The interval as a tuple of (Lower, Upper) bounds if successful.</param>
    /// <returns>True if the interval was successfully retrieved; otherwise, false.</returns>
    bool TryGetInterval(IntervalType intervalType, out (T Lower, T Upper) interval);

    /// <summary>
    /// Checks if a specific interval is valid for this provider.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval is valid; otherwise, false.</returns>
    bool IsValidInterval(IntervalType intervalType);

    /// <summary>
    /// Checks if a specific interval has been calculated.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval has been calculated; otherwise, false.</returns>
    bool IsCalculatedInterval(IntervalType intervalType);

    /// <summary>
    /// Gets all interval types that are valid for this provider.
    /// </summary>
    /// <returns>An array of valid interval types.</returns>
    IntervalType[] GetValidIntervalTypes();

    /// <summary>
    /// Gets all interval types that have been calculated.
    /// </summary>
    /// <returns>An array of calculated interval types.</returns>
    IntervalType[] GetCalculatedIntervalTypes();
}