using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Interfaces;

/// <summary>
/// Interface for continual learning trainers that can learn multiple tasks sequentially.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Continual learning (also called lifelong learning) is the ability
/// to learn new tasks over time without forgetting previously learned knowledge. Traditional
/// neural networks suffer from "catastrophic forgetting" - when trained on new data, they
/// forget what they learned before.</para>
///
/// <para><b>This interface provides methods to:</b>
/// <list type="bullet">
/// <item><description>Learn new tasks while protecting old knowledge</description></item>
/// <item><description>Evaluate performance on all learned tasks</description></item>
/// <item><description>Save and load model state for resuming training</description></item>
/// </list>
/// </para>
///
/// <para><b>Common Implementations:</b>
/// <list type="bullet">
/// <item><description><b>EWCTrainer:</b> Uses Elastic Weight Consolidation to protect important weights</description></item>
/// <item><description><b>LwFTrainer:</b> Uses Learning without Forgetting with knowledge distillation</description></item>
/// <item><description><b>GEMTrainer:</b> Uses Gradient Episodic Memory to constrain gradients</description></item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item><description>Parisi et al. "Continual Lifelong Learning with Neural Networks: A Review" (2019)</description></item>
/// <item><description>De Lange et al. "A Continual Learning Survey" (2021)</description></item>
/// </list>
/// </para>
/// </remarks>
public interface IContinualLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the underlying model being trained.
    /// </summary>
    IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Gets the configuration for the continual learner.
    /// </summary>
    IContinualLearnerConfig<T> Config { get; }

    /// <summary>
    /// Gets the number of tasks that have been learned.
    /// </summary>
    int TasksLearned { get; }

    /// <summary>
    /// Gets whether the learner is currently training.
    /// </summary>
    bool IsTraining { get; }

    /// <summary>
    /// Gets the current memory usage of the learner in bytes.
    /// </summary>
    long MemoryUsageBytes { get; }

    /// <summary>
    /// Learns a new task from the provided data.
    /// </summary>
    /// <param name="taskData">The dataset for the new task.</param>
    /// <returns>Result containing training metrics and performance information.</returns>
    /// <remarks>
    /// <para>This method trains the model on the new task while using the configured
    /// strategy to prevent forgetting of previously learned tasks.</para>
    /// </remarks>
    ContinualLearningResult<T> LearnTask(IDataset<T, TInput, TOutput> taskData);

    /// <summary>
    /// Learns a new task with a validation set for early stopping and monitoring.
    /// </summary>
    /// <param name="taskData">The training dataset for the new task.</param>
    /// <param name="validationData">The validation dataset for the new task.</param>
    /// <param name="earlyStoppingPatience">Number of epochs without improvement before stopping.</param>
    /// <returns>Result containing training metrics and performance information.</returns>
    ContinualLearningResult<T> LearnTask(
        IDataset<T, TInput, TOutput> taskData,
        IDataset<T, TInput, TOutput> validationData,
        int? earlyStoppingPatience = null);

    /// <summary>
    /// Evaluates the model on all learned tasks.
    /// </summary>
    /// <returns>Comprehensive evaluation result including backward/forward transfer.</returns>
    ContinualEvaluationResult<T> EvaluateAllTasks();

    /// <summary>
    /// Evaluates the model on a specific task.
    /// </summary>
    /// <param name="taskId">The task identifier (0-indexed).</param>
    /// <param name="testData">The test data for the task.</param>
    /// <returns>Evaluation result for the specified task.</returns>
    TaskEvaluationResult<T> EvaluateTask(int taskId, IDataset<T, TInput, TOutput> testData);

    /// <summary>
    /// Saves the learner state to a file.
    /// </summary>
    /// <param name="path">Path to save the state.</param>
    /// <remarks>
    /// <para>This saves all state needed to resume training, including:
    /// - Model parameters
    /// - Strategy state (e.g., Fisher Information for EWC)
    /// - Memory buffer contents
    /// - Training history
    /// </para>
    /// </remarks>
    void Save(string path);

    /// <summary>
    /// Loads the learner state from a file.
    /// </summary>
    /// <param name="path">Path to load the state from.</param>
    void Load(string path);

    /// <summary>
    /// Resets the learner to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>This clears all learned knowledge, including:
    /// - Model parameters (reset to initial values)
    /// - Strategy state
    /// - Memory buffer
    /// - Task history
    /// </para>
    /// </remarks>
    void Reset();

    /// <summary>
    /// Gets the training history for a specific task.
    /// </summary>
    /// <param name="taskId">The task identifier.</param>
    /// <returns>The training result for the task, or null if not available.</returns>
    ContinualLearningResult<T>? GetTaskHistory(int taskId);

    /// <summary>
    /// Gets all training history.
    /// </summary>
    /// <returns>List of all training results.</returns>
    IReadOnlyList<ContinualLearningResult<T>> GetAllHistory();

    /// <summary>
    /// Computes the current forgetting metric for all tasks.
    /// </summary>
    /// <returns>Dictionary mapping task ID to forgetting amount.</returns>
    IReadOnlyDictionary<int, T> ComputeForgetting();

    /// <summary>
    /// Event raised when a task starts training.
    /// </summary>
    event EventHandler<TaskEventArgs>? TaskStarted;

    /// <summary>
    /// Event raised when a task finishes training.
    /// </summary>
    event EventHandler<TaskCompletedEventArgs<T>>? TaskCompleted;

    /// <summary>
    /// Event raised when an epoch completes during training.
    /// </summary>
    event EventHandler<EpochEventArgs<T>>? EpochCompleted;
}

/// <summary>
/// Event arguments for task events.
/// </summary>
public class TaskEventArgs : EventArgs
{
    /// <summary>
    /// Gets the task identifier.
    /// </summary>
    public int TaskId { get; }

    /// <summary>
    /// Gets the number of samples in the task.
    /// </summary>
    public int SampleCount { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TaskEventArgs"/> class.
    /// </summary>
    public TaskEventArgs(int taskId, int sampleCount)
    {
        TaskId = taskId;
        SampleCount = sampleCount;
    }
}

/// <summary>
/// Event arguments for task completion events.
/// </summary>
public class TaskCompletedEventArgs<T> : TaskEventArgs
{
    /// <summary>
    /// Gets the training result.
    /// </summary>
    public ContinualLearningResult<T> Result { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TaskCompletedEventArgs{T}"/> class.
    /// </summary>
    public TaskCompletedEventArgs(int taskId, int sampleCount, ContinualLearningResult<T> result)
        : base(taskId, sampleCount)
    {
        Result = result;
    }
}

/// <summary>
/// Event arguments for epoch completion events.
/// </summary>
public class EpochEventArgs<T> : EventArgs
{
    /// <summary>
    /// Gets the task identifier.
    /// </summary>
    public int TaskId { get; }

    /// <summary>
    /// Gets the epoch number (0-indexed).
    /// </summary>
    public int Epoch { get; }

    /// <summary>
    /// Gets the total number of epochs.
    /// </summary>
    public int TotalEpochs { get; }

    /// <summary>
    /// Gets the training loss for this epoch.
    /// </summary>
    public T Loss { get; }

    /// <summary>
    /// Gets the validation loss for this epoch, if available.
    /// </summary>
    public T? ValidationLoss { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="EpochEventArgs{T}"/> class.
    /// </summary>
    public EpochEventArgs(int taskId, int epoch, int totalEpochs, T loss, T? validationLoss = default)
    {
        TaskId = taskId;
        Epoch = epoch;
        TotalEpochs = totalEpochs;
        Loss = loss;
        ValidationLoss = validationLoss;
    }
}
