namespace AiDotNet.ActiveLearning.Results;

/// <summary>
/// Result of evaluating a model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EvaluationResult<T>
{
    /// <summary>
    /// Test accuracy.
    /// </summary>
    public T Accuracy { get; }

    /// <summary>
    /// Test loss.
    /// </summary>
    public T Loss { get; }

    /// <summary>
    /// Number of test examples.
    /// </summary>
    public int NumExamples { get; }

    /// <summary>
    /// Evaluation time.
    /// </summary>
    public TimeSpan EvaluationTime { get; }

    /// <summary>
    /// Initializes a new instance of the EvaluationResult class.
    /// </summary>
    public EvaluationResult(T accuracy, T loss, int numExamples, TimeSpan evaluationTime)
    {
        Accuracy = accuracy;
        Loss = loss;
        NumExamples = numExamples;
        EvaluationTime = evaluationTime;
    }
}
