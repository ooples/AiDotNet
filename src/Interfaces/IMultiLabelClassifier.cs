namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for multi-label classifiers that can predict multiple labels per sample.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Multi-label classification differs from multi-class classification:
/// - Multi-class: One label per sample (mutually exclusive)
/// - Multi-label: Zero, one, or many labels per sample (not mutually exclusive)
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// In multi-label classification, each sample can have multiple labels:
///
/// Examples:
/// - An article tagged with "politics", "economy", and "international"
/// - A movie classified as "action", "comedy", and "romance"
/// - An image containing "dog", "person", and "outdoor"
///
/// The output is a binary matrix where each column is a label indicator.
/// </para>
/// </remarks>
public interface IMultiLabelClassifier<T> : IClassifier<T>
{
    /// <summary>
    /// Gets the number of labels that can be predicted.
    /// </summary>
    int NumLabels { get; }

    /// <summary>
    /// Predicts binary indicators for each label for each sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A binary matrix where each row is a sample and each column is a label indicator (1=present, 0=absent).</returns>
    Matrix<T> PredictMultiLabel(Matrix<T> input);

    /// <summary>
    /// Predicts probabilities for each label for each sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A probability matrix where each row is a sample and each column is the probability of that label.</returns>
    Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input);
}
