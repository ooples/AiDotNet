using AiDotNet.Enums;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for prediction-based statistics that combine model statistics and intervals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IPredictionStatistics<T> : IModelStatistics<T>, IIntervalProvider<T>
{
    /// <summary>
    /// Gets the type of prediction (regression, classification, etc.).
    /// </summary>
    PredictionType PredictionType { get; }
}