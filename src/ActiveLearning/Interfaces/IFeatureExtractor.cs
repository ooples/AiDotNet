namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for models that can extract feature representations from inputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Feature extraction converts raw input data into a
/// numerical representation (feature vector) that captures the important characteristics
/// of the input. This is useful for tasks like clustering, similarity comparisons,
/// and active learning strategies that need to measure diversity.</para>
///
/// <para><b>Common Uses:</b></para>
/// <list type="bullet">
/// <item><description>BADGE active learning strategy uses gradient embeddings as features</description></item>
/// <item><description>CoreSet strategy uses feature distances for diversity sampling</description></item>
/// <item><description>Similarity-based sample selection</description></item>
/// </list>
/// </remarks>
public interface IFeatureExtractor<T, TInput>
{
    /// <summary>
    /// Extracts feature representations from the given input.
    /// </summary>
    /// <param name="input">The input data to extract features from.</param>
    /// <returns>A vector containing the extracted features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method transforms raw input (like an image or text)
    /// into a list of numbers (feature vector) that represents the input's characteristics.
    /// Similar inputs should produce similar feature vectors.</para>
    /// </remarks>
    Vector<T> ExtractFeatures(TInput input);

    /// <summary>
    /// Gets the dimensionality of the extracted feature vectors.
    /// </summary>
    /// <remarks>
    /// <para>Returns the length of the feature vectors produced by <see cref="ExtractFeatures"/>.
    /// This is useful for pre-allocating storage or validating compatibility.</para>
    /// </remarks>
    int FeatureDimension { get; }
}
