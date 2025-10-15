namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for model-based statistics providers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IModelStatistics<T> : IStatisticsProvider<T>
{
    /// <summary>
    /// Gets the number of features or parameters in the model.
    /// </summary>
    int FeatureCount { get; }
}