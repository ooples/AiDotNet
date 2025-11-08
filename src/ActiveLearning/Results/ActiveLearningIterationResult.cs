namespace AiDotNet.ActiveLearning.Results;

/// <summary>
/// Result of one active learning iteration (query + train).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ActiveLearningIterationResult<T>
{
    /// <summary>
    /// Indices of examples selected for labeling.
    /// </summary>
    public int[] SelectedIndices { get; }

    /// <summary>
    /// Informativeness scores for selected examples.
    /// </summary>
    public T[] SelectionScores { get; }

    /// <summary>
    /// Model accuracy after training on newly labeled examples.
    /// </summary>
    public T Accuracy { get; }

    /// <summary>
    /// Model loss after training.
    /// </summary>
    public T Loss { get; }

    /// <summary>
    /// Total number of labeled examples after this iteration.
    /// </summary>
    public int TotalLabeled { get; }

    /// <summary>
    /// Time taken for this iteration.
    /// </summary>
    public TimeSpan IterationTime { get; }

    /// <summary>
    /// Initializes a new instance of the ActiveLearningIterationResult class.
    /// </summary>
    public ActiveLearningIterationResult(
        int[] selectedIndices,
        T[] selectionScores,
        T accuracy,
        T loss,
        int totalLabeled,
        TimeSpan iterationTime)
    {
        SelectedIndices = selectedIndices;
        SelectionScores = selectionScores;
        Accuracy = accuracy;
        Loss = loss;
        TotalLabeled = totalLabeled;
        IterationTime = iterationTime;
    }
}
