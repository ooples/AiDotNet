using AiDotNet.Interfaces;
using AiDotNet.ContinualLearning.Results;

namespace AiDotNet.ContinualLearning.Interfaces;

/// <summary>
/// Represents a continual learning trainer that can learn new tasks without forgetting previous ones.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Continual learning (also called lifelong learning) is about
/// training models that can learn new tasks over time without forgetting what they learned before.
/// This is similar to how humans learn - we don't forget everything we knew when we learn something new.</para>
///
/// <para>Common challenges in continual learning include:
/// - <b>Catastrophic forgetting:</b> When learning new tasks causes the model to forget old tasks
/// - <b>Task interference:</b> When different tasks conflict with each other
/// - <b>Memory constraints:</b> Limited storage for remembering previous tasks
/// </para>
///
/// <para>This interface defines methods for:
/// - Training on new tasks sequentially
/// - Evaluating performance on all learned tasks
/// - Managing task boundaries and memory
/// </para>
/// </remarks>
public interface IContinualLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the base model being trained with continual learning.
    /// </summary>
    IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Gets the configuration for continual learning.
    /// </summary>
    IContinualLearnerConfig<T> Config { get; }

    /// <summary>
    /// Gets the number of tasks learned so far.
    /// </summary>
    int TasksLearned { get; }

    /// <summary>
    /// Trains the model on a new task while preserving knowledge from previous tasks.
    /// </summary>
    /// <param name="taskData">The data for the new task.</param>
    /// <returns>Result containing training metrics and performance on all tasks.</returns>
    ContinualLearningResult<T> LearnTask(IDataset<T, TInput, TOutput> taskData);

    /// <summary>
    /// Evaluates the model's performance on all learned tasks.
    /// </summary>
    /// <returns>Result containing accuracy and loss for each task.</returns>
    ContinualEvaluationResult<T> EvaluateAllTasks();

    /// <summary>
    /// Evaluates the model on a specific task.
    /// </summary>
    /// <param name="taskId">The task identifier (0-indexed).</param>
    /// <param name="testData">The test data for the task.</param>
    /// <returns>Result containing accuracy and loss for the specified task.</returns>
    TaskEvaluationResult<T> EvaluateTask(int taskId, IDataset<T, TInput, TOutput> testData);

    /// <summary>
    /// Saves the continual learning model and all task-specific information.
    /// </summary>
    /// <param name="directoryPath">Directory to save the model and metadata.</param>
    void Save(string directoryPath);

    /// <summary>
    /// Loads a previously saved continual learning model.
    /// </summary>
    /// <param name="directoryPath">Directory containing the saved model and metadata.</param>
    void Load(string directoryPath);

    /// <summary>
    /// Resets the learner to its initial state (forgets all tasks).
    /// </summary>
    void Reset();
}
