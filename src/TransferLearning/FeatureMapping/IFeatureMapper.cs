namespace AiDotNet.TransferLearning.FeatureMapping;

/// <summary>
/// Defines the interface for mapping features from a source domain to a target domain.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A feature mapper is like a translator between two different languages.
/// When you have data from one domain (source) and want to use it in another domain (target),
/// the feature mapper transforms the data so it makes sense in the new context.
/// </para>
/// <para>
/// For example, if you trained a model on images (which might have thousands of features)
/// and want to use that knowledge for text (which has different features), a feature mapper
/// helps bridge that gap by finding a common representation.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("FeatureMapper")]
public interface IFeatureMapper<T>
{
    /// <summary>
    /// Maps features from the source domain to the target domain.
    /// </summary>
    /// <param name="sourceFeatures">The features from the source domain.</param>
    /// <param name="targetDimension">The desired number of dimensions in the target domain.</param>
    /// <returns>The mapped features with the target dimension.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes data from the source domain and transforms it
    /// to match the structure expected by the target domain. Think of it like resizing and
    /// reformatting a photo to fit a different frame.
    /// </para>
    /// </remarks>
    Matrix<T> MapToTarget(Matrix<T> sourceFeatures, int targetDimension);

    /// <summary>
    /// Maps features from the target domain back to the source domain.
    /// </summary>
    /// <param name="targetFeatures">The features from the target domain.</param>
    /// <param name="sourceDimension">The desired number of dimensions in the source domain.</param>
    /// <returns>The mapped features with the source dimension.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the reverse operation - taking data from the target domain
    /// and transforming it back to the source domain format. Like translating text back to
    /// the original language.
    /// </para>
    /// </remarks>
    Matrix<T> MapToSource(Matrix<T> targetFeatures, int sourceDimension);

    /// <summary>
    /// Trains the feature mapper on source and target data.
    /// </summary>
    /// <param name="sourceData">Training data from the source domain.</param>
    /// <param name="targetData">Training data from the target domain.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Before the mapper can translate between domains, it needs to learn
    /// how the two domains relate to each other. This method teaches the mapper by showing it
    /// examples from both domains.
    /// </para>
    /// </remarks>
    void Train(Matrix<T> sourceData, Matrix<T> targetData);

    /// <summary>
    /// Gets the confidence score for the mapping quality.
    /// </summary>
    /// <returns>A value between 0 and 1, where higher values indicate better mapping quality.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how confident the mapper is in its translations.
    /// A high score (close to 1) means the mapper thinks it can translate well between the domains.
    /// A low score (close to 0) means the domains might be too different to map effectively.
    /// </para>
    /// </remarks>
    T GetMappingConfidence();

    /// <summary>
    /// Determines if the mapper has been trained and is ready to use.
    /// </summary>
    /// <returns>True if the mapper is trained; otherwise, false.</returns>
    bool IsTrained { get; }
}
