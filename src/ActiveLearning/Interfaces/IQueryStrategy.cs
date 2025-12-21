using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Represents a strategy for selecting which examples to label in active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A query strategy decides which unlabeled examples
/// are most worth labeling. Different strategies use different criteria:</para>
///
/// <para><b>Common Strategies:</b>
/// - <b>Uncertainty Sampling:</b> Select examples where the model is least confident
///   - Least Confidence: Select examples with lowest max probability
///   - Margin Sampling: Select examples with smallest margin between top 2 classes
///   - Entropy: Select examples with highest prediction entropy
/// </para>
///
/// <para>
/// - <b>Query-by-Committee:</b> Train multiple models and select examples where they disagree most
/// </para>
///
/// <para>
/// - <b>Expected Gradient Length:</b> Select examples that would cause the largest gradient update
/// </para>
///
/// <para>
/// - <b>Diversity-based:</b> Select examples that are different from already labeled data
/// </para>
/// </remarks>
public interface IQueryStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Scores unlabeled examples by informativeness.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="unlabeledData">The pool of unlabeled examples.</param>
    /// <param name="labeledData">Optional current labeled dataset (required for diversity-based strategies).</param>
    /// <returns>Vector of scores (higher = more informative) for each example.</returns>
    Vector<T> ScoreExamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData = null);

    /// <summary>
    /// Selects the top-k most informative examples.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="unlabeledData">The pool of unlabeled examples.</param>
    /// <param name="k">Number of examples to select.</param>
    /// <param name="labeledData">Optional current labeled dataset (required for diversity-based strategies).</param>
    /// <returns>Vector of indices of selected examples in the unlabeled pool.</returns>
    Vector<int> SelectBatch(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        int k,
        IDataset<T, TInput, TOutput>? labeledData = null);

    /// <summary>
    /// Gets the name of this query strategy.
    /// </summary>
    string Name { get; }
}
