namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that passes through all features without any selection.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
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
public class NoFeatureSelector<T> : IFeatureSelector<T>
{
    /// <summary>
    /// Returns the input feature matrix without any selection or modification.
    /// </summary>
    /// <param name="allFeaturesMatrix">The matrix containing all features.</param>
    /// <returns>The same matrix that was provided as input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method simply returns the exact same data that was passed to it. 
    /// No features are removed or modified in any way.
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
    public Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix)
    {
        return allFeaturesMatrix;
    }
}