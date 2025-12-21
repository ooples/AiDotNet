using AiDotNet.ActiveLearning.Results;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Represents an active learning system that intelligently selects which data to label.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Active learning is about training models more efficiently
/// by carefully choosing which examples to label. Instead of labeling all data randomly,
/// active learning selects the most informative examples.</para>
///
/// <para><b>Why Active Learning?</b>
/// - Labeling data is expensive (time, money, expertise)
/// - Most data is redundant - similar examples don't teach the model much new
/// - By selecting informative examples, we can achieve high accuracy with fewer labels
/// </para>
///
/// <para><b>How it works:</b>
/// 1. Start with a small labeled dataset
/// 2. Train a model on labeled data
/// 3. Use a query strategy to select the most informative unlabeled examples
/// 4. Get labels for selected examples (from human annotator or oracle)
/// 5. Add newly labeled examples to training set
/// 6. Repeat until budget is exhausted or performance is satisfactory
/// </para>
///
/// <para><b>Common Query Strategies:</b>
/// - Uncertainty Sampling: Select examples the model is most uncertain about
/// - Query-by-Committee: Use disagreement between multiple models
/// - Expected Gradient Length: Select examples that will change the model most
/// - Diversity-based: Ensure selected examples are diverse
/// </para>
/// </remarks>
public interface IActiveLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the underlying model being trained.
    /// </summary>
    IFullModel<T, TInput, TOutput> Model { get; }

    /// <summary>
    /// Gets the query strategy used to select examples.
    /// </summary>
    IQueryStrategy<T, TInput, TOutput> QueryStrategy { get; }

    /// <summary>
    /// Gets the number of labeled examples currently in the training set.
    /// </summary>
    int NumLabeledExamples { get; }

    /// <summary>
    /// Gets the total number of queries made so far.
    /// </summary>
    int TotalQueries { get; }

    /// <summary>
    /// Performs one iteration of active learning.
    /// </summary>
    /// <param name="unlabeledPool">The pool of unlabeled data to query from.</param>
    /// <param name="oracle">Function to get labels for selected examples.</param>
    /// <param name="batchSize">Number of examples to query in this iteration.</param>
    /// <returns>Result containing selected indices, model performance, and metrics.</returns>
    ActiveLearningIterationResult<T> QueryAndTrain(
        IDataset<T, TInput, TOutput> unlabeledPool,
        Func<TInput, TOutput> oracle,
        int batchSize = 1);

    /// <summary>
    /// Selects the most informative examples from the unlabeled pool without training.
    /// </summary>
    /// <param name="unlabeledPool">The pool of unlabeled data.</param>
    /// <param name="batchSize">Number of examples to select.</param>
    /// <returns>Vector of indices of selected examples in the unlabeled pool.</returns>
    Vector<int> SelectExamples(IDataset<T, TInput, TOutput> unlabeledPool, int batchSize);

    /// <summary>
    /// Trains the model on the current labeled dataset.
    /// </summary>
    /// <returns>Training result with loss and accuracy.</returns>
    TrainingResult<T> Train();

    /// <summary>
    /// Evaluates the model on a test set.
    /// </summary>
    /// <param name="testData">The test dataset.</param>
    /// <returns>Evaluation result with accuracy and loss.</returns>
    EvaluationResult<T> Evaluate(IDataset<T, TInput, TOutput> testData);

    /// <summary>
    /// Resets the active learner to initial state.
    /// </summary>
    void Reset();
}
