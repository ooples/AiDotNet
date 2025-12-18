namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that can provide information about their feature usage.
/// </summary>
public interface IFeatureAware
{
    /// <summary>
    /// Gets the indices of features that are actively used by this model.
    /// </summary>
    IEnumerable<int> GetActiveFeatureIndices();

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    void SetActiveFeatureIndices(IEnumerable<int> featureIndices);

    /// <summary>
    /// Checks if a specific feature is used by this model.
    /// </summary>
    bool IsFeatureUsed(int featureIndex);
}

/// <summary>
/// Interface for models that can provide feature importance scores.
/// </summary>
/// <typeparam name="T">The numeric type used for feature importance scores.</typeparam>
public interface IFeatureImportance<T>
{
    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    Dictionary<string, T> GetFeatureImportance();
}
