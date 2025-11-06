namespace AiDotNet.Models.Results;

/// <summary>
/// Results from a single meta-training step (one outer loop update).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This class represents metrics from one iteration of meta-training, tracking the
/// performance of a single outer loop update across a batch of tasks. It's a lightweight
/// snapshot designed for real-time monitoring during training.
/// </para>
/// <para><b>For Beginners:</b> Think of this as the "score" for one training iteration.
///
/// Each meta-training iteration:
/// 1. Samples a batch of tasks (e.g., 4 tasks)
/// 2. Adapts to each task (inner loop)
/// 3. Updates meta-parameters based on adaptation results (outer loop)
/// 4. Returns these metrics to show how well that update performed
///
/// You'll get one of these for each iteration during training, allowing you to:
/// - Monitor training progress in real-time
/// - Log metrics to TensorBoard or similar tools
/// - Implement early stopping or learning rate schedules
/// - Debug training issues as they occur
/// </para>
/// </remarks>
public class MetaTrainingStepResult<T>
{
    /// <summary>
    /// Gets the meta-loss for this training step.
    /// </summary>
    /// <value>
    /// The loss computed across all tasks in the meta-batch after inner loop adaptation.
    /// This is what the outer loop optimizes.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main number the meta-learner tries to minimize.
    /// Lower meta-loss means the model is getting better at adapting to new tasks quickly.
    /// </para>
    /// </remarks>
    public T MetaLoss { get; }

    /// <summary>
    /// Gets the average task-specific loss after adaptation.
    /// </summary>
    /// <value>
    /// The mean loss across all tasks in the batch, measured on query sets after
    /// inner loop adaptation.
    /// </value>
    public T TaskLoss { get; }

    /// <summary>
    /// Gets the average accuracy across tasks in this step.
    /// </summary>
    /// <value>
    /// The mean accuracy on query sets after adaptation, across all tasks in the batch.
    /// </value>
    public T Accuracy { get; }

    /// <summary>
    /// Gets the number of tasks processed in this meta-training step.
    /// </summary>
    /// <value>
    /// The meta-batch size for this iteration.
    /// </value>
    public int NumTasks { get; }

    /// <summary>
    /// Gets the iteration number for this training step.
    /// </summary>
    /// <value>
    /// The sequential iteration count, useful for time-series analysis of training progress.
    /// </value>
    public int Iteration { get; }

    /// <summary>
    /// Gets the time taken for this meta-training step in milliseconds.
    /// </summary>
    /// <value>
    /// The elapsed time for completing this outer loop update, including all inner loop adaptations.
    /// </value>
    public double TimeMs { get; }

    /// <summary>
    /// Gets algorithm-specific metrics for this training step.
    /// </summary>
    /// <value>
    /// A dictionary of custom metrics with generic T values.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> Common additional metrics include:
    /// - "gradient_norm": Magnitude of meta-gradients
    /// - "support_accuracy": Accuracy on support sets (should be very high)
    /// - "learning_rate": Current learning rate (if using schedules)
    /// - "parameter_norm": Magnitude of model parameters
    /// </para>
    /// </remarks>
    public Dictionary<string, T> AdditionalMetrics { get; }

    /// <summary>
    /// Initializes a new instance with metrics from one meta-training step.
    /// </summary>
    /// <param name="metaLoss">The meta-loss for this step.</param>
    /// <param name="taskLoss">The average task-specific loss.</param>
    /// <param name="accuracy">The average accuracy across tasks.</param>
    /// <param name="numTasks">The number of tasks in the meta-batch.</param>
    /// <param name="iteration">The iteration number.</param>
    /// <param name="timeMs">Time taken for this step in milliseconds.</param>
    /// <param name="additionalMetrics">Optional algorithm-specific metrics.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this after each meta-training step to package
    /// the results. This makes it easy to log, visualize, and monitor training progress.
    /// </para>
    /// </remarks>
    public MetaTrainingStepResult(
        T metaLoss,
        T taskLoss,
        T accuracy,
        int numTasks,
        int iteration,
        double timeMs,
        Dictionary<string, T>? additionalMetrics = null)
    {
        MetaLoss = metaLoss;
        TaskLoss = taskLoss;
        Accuracy = accuracy;
        NumTasks = numTasks;
        Iteration = iteration;
        TimeMs = timeMs;
        AdditionalMetrics = additionalMetrics != null ? new Dictionary<string, T>(additionalMetrics) : new Dictionary<string, T>();
    }

    /// <summary>
    /// Returns a concise string representation for logging.
    /// </summary>
    public override string ToString()
    {
        return $"[Iter {Iteration}] MetaLoss={MetaLoss}, TaskLoss={TaskLoss}, Acc={Accuracy}, Tasks={NumTasks}, Time={TimeMs:F1}ms";
    }
}
