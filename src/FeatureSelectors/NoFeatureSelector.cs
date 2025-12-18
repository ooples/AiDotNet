namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that passes through all features without any selection.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class implements the "pass-through" or "identity" pattern, where
/// the output is identical to the input. It's used when you want to maintain the same interface
/// as other feature selectors but don't actually want to remove any features.
/// </para>
/// <para>
/// Think of it like a filter that doesn't filter anything out - it lets everything pass through unchanged.
/// This is useful in scenarios where feature selection is optional but your code expects a feature
/// selector to be provided.
/// </para>
/// </remarks>
public class NoFeatureSelector<T, TInput> : FeatureSelectorBase<T, TInput>
{
    /// <summary>
    /// Returns all feature indices without any selection.
    /// </summary>
    /// <param name="allFeatures">The input data containing all features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features in the dataset.</param>
    /// <returns>A list containing all feature indices (0 to numFeatures-1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method simply returns a list of all feature indices, which means
    /// all features will be kept without any filtering.
    /// </para>
    /// <para>
    /// This is useful in several scenarios:
    /// <list type="bullet">
    /// <item><description>When you want to compare the performance of a model with and without feature selection</description></item>
    /// <item><description>When you're using a system that requires a feature selector but you don't want to perform any selection</description></item>
    /// <item><description>As a default option when no specific feature selection method is specified</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures)
    {
        // Return all feature indices (no filtering)
        return Enumerable.Range(0, numFeatures).ToList();
    }
}
