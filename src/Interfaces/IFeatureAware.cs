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
    /// Checks if a specific feature is used by this model.
    /// </summary>
    bool IsFeatureUsed(int featureIndex);

    /// <summary>
    /// Sets which feature indices should be considered active for this model.
    /// </summary>
    /// <param name="featureIndices">The collection of feature indices to activate.</param>
    void SetActiveFeatureIndices(IEnumerable<int> featureIndices);
}