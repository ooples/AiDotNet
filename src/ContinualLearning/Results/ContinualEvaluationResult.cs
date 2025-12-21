namespace AiDotNet.ContinualLearning.Results;

/// <summary>
/// Comprehensive evaluation result across all learned tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class provides a complete picture of how well the model
/// performs across all tasks it has learned. It includes key metrics for measuring
/// continual learning effectiveness.</para>
///
/// <para><b>Key Metrics Explained:</b>
/// <list type="bullet">
/// <item><description><b>Average Accuracy:</b> Mean accuracy across all tasks (higher is better)</description></item>
/// <item><description><b>Backward Transfer:</b> How learning new tasks affects old task performance (positive = improvement, negative = forgetting)</description></item>
/// <item><description><b>Forward Transfer:</b> How old knowledge helps with new tasks (positive = positive transfer)</description></item>
/// <item><description><b>Forgetting:</b> Maximum accuracy drop on any previous task (lower is better)</description></item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Lopez-Paz and Ranzato "Gradient Episodic Memory for Continual Learning" (2017)</para>
/// </remarks>
public class ContinualEvaluationResult<T>
{
    /// <summary>
    /// Gets the accuracy on each task.
    /// </summary>
    public Vector<T> TaskAccuracies { get; }

    /// <summary>
    /// Gets the loss on each task.
    /// </summary>
    public Vector<T> TaskLosses { get; }

    /// <summary>
    /// Gets the average accuracy across all tasks.
    /// </summary>
    public T AverageAccuracy { get; }

    /// <summary>
    /// Gets the average loss across all tasks.
    /// </summary>
    public T AverageLoss { get; }

    /// <summary>
    /// Gets the backward transfer metric.
    /// </summary>
    /// <remarks>
    /// <para><b>Definition:</b> BWT = (1/T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})</para>
    /// <para>Where R_{j,i} is the accuracy on task i after learning task j.</para>
    /// <para>Negative BWT indicates forgetting. Positive BWT indicates that learning
    /// later tasks improved performance on earlier tasks.</para>
    /// </remarks>
    public T BackwardTransfer { get; }

    /// <summary>
    /// Gets the forward transfer metric.
    /// </summary>
    /// <remarks>
    /// <para><b>Definition:</b> FWT = (1/T-1) * sum_{i=2}^{T} (R_{i-1,i} - R_{0,i})</para>
    /// <para>Where R_{j,i} is the accuracy on task i after learning task j,
    /// and R_{0,i} is the accuracy on task i before any training.</para>
    /// <para>Positive FWT indicates that learning previous tasks helped with new tasks.</para>
    /// </remarks>
    public T ForwardTransfer { get; }

    /// <summary>
    /// Gets the time taken for evaluation.
    /// </summary>
    public TimeSpan EvaluationTime { get; }

    /// <summary>
    /// Gets the maximum forgetting observed on any task.
    /// </summary>
    /// <remarks>
    /// <para>MaxForgetting = max_i (max_j(R_{j,i}) - R_{T,i})</para>
    /// <para>This is the worst-case accuracy drop on any single task.</para>
    /// </remarks>
    public T? MaxForgetting { get; init; }

    /// <summary>
    /// Gets the per-task evaluation results.
    /// </summary>
    public IReadOnlyList<TaskEvaluationResult<T>>? PerTaskResults { get; init; }

    /// <summary>
    /// Gets the accuracy matrix R where R[i,j] is accuracy on task j after learning task i.
    /// </summary>
    /// <remarks>
    /// <para>This matrix enables detailed analysis of learning dynamics over time.</para>
    /// </remarks>
    public Matrix<T>? AccuracyMatrix { get; init; }

    /// <summary>
    /// Gets the intransigence metric (inability to learn new tasks).
    /// </summary>
    /// <remarks>
    /// <para>Measures how much worse the model is at learning new tasks compared to
    /// a model trained only on that task. Higher values indicate difficulty adapting.</para>
    /// </remarks>
    public T? Intransigence { get; init; }

    /// <summary>
    /// Gets the learning curve efficiency (area under the learning curve).
    /// </summary>
    public T? LearningEfficiency { get; init; }

    /// <summary>
    /// Gets the stability-plasticity ratio.
    /// </summary>
    /// <remarks>
    /// <para>Measures the trade-off between retaining old knowledge (stability) and
    /// learning new information (plasticity). Closer to 1 indicates good balance.</para>
    /// </remarks>
    public T? StabilityPlasticityRatio { get; init; }

    /// <summary>
    /// Gets the total number of tasks evaluated.
    /// </summary>
    public int TaskCount => TaskAccuracies.Length;

    /// <summary>
    /// Initializes a new instance of the <see cref="ContinualEvaluationResult{T}"/> class.
    /// </summary>
    public ContinualEvaluationResult(
        Vector<T> taskAccuracies,
        Vector<T> taskLosses,
        T averageAccuracy,
        T averageLoss,
        T backwardTransfer,
        T forwardTransfer,
        TimeSpan evaluationTime)
    {
        if (taskAccuracies == null)
            throw new ArgumentNullException(nameof(taskAccuracies));
        if (taskLosses == null)
            throw new ArgumentNullException(nameof(taskLosses));
        if (taskAccuracies.Length != taskLosses.Length)
            throw new ArgumentException("Task accuracies and losses must have the same length");

        TaskAccuracies = taskAccuracies;
        TaskLosses = taskLosses;
        AverageAccuracy = averageAccuracy;
        AverageLoss = averageLoss;
        BackwardTransfer = backwardTransfer;
        ForwardTransfer = forwardTransfer;
        EvaluationTime = evaluationTime;
    }

    /// <summary>
    /// Returns a string representation of the evaluation result.
    /// </summary>
    public override string ToString()
    {
        return $"Tasks: {TaskCount}, AvgAccuracy: {AverageAccuracy}, BWT: {BackwardTransfer}, FWT: {ForwardTransfer}";
    }

    /// <summary>
    /// Generates a detailed report of the evaluation results.
    /// </summary>
    public string GenerateReport()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("=== Continual Learning Evaluation Report ===");
        sb.AppendLine();
        sb.AppendLine($"Total Tasks: {TaskCount}");
        sb.AppendLine($"Evaluation Time: {EvaluationTime.TotalSeconds:F2}s");
        sb.AppendLine();
        sb.AppendLine("--- Aggregate Metrics ---");
        sb.AppendLine($"Average Accuracy: {AverageAccuracy}");
        sb.AppendLine($"Average Loss: {AverageLoss}");
        sb.AppendLine($"Backward Transfer (BWT): {BackwardTransfer}");
        sb.AppendLine($"Forward Transfer (FWT): {ForwardTransfer}");

        if (MaxForgetting != null)
            sb.AppendLine($"Maximum Forgetting: {MaxForgetting}");
        if (Intransigence != null)
            sb.AppendLine($"Intransigence: {Intransigence}");
        if (StabilityPlasticityRatio != null)
            sb.AppendLine($"Stability-Plasticity Ratio: {StabilityPlasticityRatio}");

        sb.AppendLine();
        sb.AppendLine("--- Per-Task Results ---");
        for (int i = 0; i < TaskCount; i++)
        {
            sb.AppendLine($"Task {i}: Accuracy={TaskAccuracies[i]}, Loss={TaskLosses[i]}");
        }

        return sb.ToString();
    }
}
