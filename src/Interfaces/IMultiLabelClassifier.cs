namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for multi-label classification models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Multi-label classification assigns multiple labels to each sample,
/// unlike traditional classification which assigns exactly one label. For example, a news article
/// might be tagged with both "politics" and "economy".</para>
///
/// <para><b>Key differences from multi-class classification:</b>
/// <list type="bullet">
/// <item>Each sample can have zero, one, or multiple labels</item>
/// <item>Labels are not mutually exclusive</item>
/// <item>Output is a binary indicator matrix (1 = label present, 0 = absent)</item>
/// </list>
/// </para>
///
/// <para><b>Common approaches:</b>
/// <list type="bullet">
/// <item><b>Problem Transformation:</b> Convert to multiple binary problems (Binary Relevance, Classifier Chains)</item>
/// <item><b>Algorithm Adaptation:</b> Adapt algorithms to handle multiple labels (ML-kNN, RAkEL)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("MultiLabelClassifier")]
public interface IMultiLabelClassifier<T> : IFullModel<T, Matrix<T>, Matrix<T>>
{
    /// <summary>
    /// Gets the number of possible labels.
    /// </summary>
    int NumLabels { get; }

    /// <summary>
    /// Gets the label names if available.
    /// </summary>
    string[]? LabelNames { get; }

    /// <summary>
    /// Trains the multi-label classifier.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <param name="labels">Binary label matrix [n_samples, n_labels] where 1 indicates label presence.</param>
    new void Train(Matrix<T> features, Matrix<T> labels);

    /// <summary>
    /// Predicts binary label indicators for input samples.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <returns>Binary label matrix [n_samples, n_labels].</returns>
    new Matrix<T> Predict(Matrix<T> features);

    /// <summary>
    /// Predicts label probabilities for input samples.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <returns>Probability matrix [n_samples, n_labels] with values in [0, 1].</returns>
    Matrix<T> PredictProbabilities(Matrix<T> features);
}
