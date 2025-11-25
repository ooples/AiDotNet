namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Represents a batch of tasks for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A task batch groups multiple tasks together for efficient processing.
/// This is similar to how regular machine learning uses batches of examples, but here we're
/// batching entire tasks instead of individual examples.
///
/// For example, instead of processing one 5-way 1-shot task at a time, you might process
/// 32 tasks together in a batch for faster training.
/// </para>
/// </remarks>
public class TaskBatch<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the TaskBatch class.
    /// </summary>
    /// <param name="tasks">The array of tasks in this batch.</param>
    public TaskBatch(ITask<T, TInput, TOutput>[] tasks)
    {
        Tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        if (tasks.Length == 0)
        {
            throw new ArgumentException("Task batch cannot be empty.", nameof(tasks));
        }

        // Validate that all tasks have the same configuration
        var firstTask = tasks[0];
        NumWays = firstTask.NumWays;
        NumShots = firstTask.NumShots;
        NumQueryPerClass = firstTask.NumQueryPerClass;

        for (int i = 1; i < tasks.Length; i++)
        {
            if (tasks[i].NumWays != NumWays ||
                tasks[i].NumShots != NumShots ||
                tasks[i].NumQueryPerClass != NumQueryPerClass)
            {
                throw new ArgumentException(
                    "All tasks in a batch must have the same configuration (NumWays, NumShots, NumQueryPerClass).",
                    nameof(tasks));
            }
        }
    }

    /// <summary>
    /// Gets the array of tasks in this batch.
    /// </summary>
    public ITask<T, TInput, TOutput>[] Tasks { get; }

    /// <summary>
    /// Gets the number of tasks in this batch.
    /// </summary>
    public int BatchSize => Tasks.Length;

    /// <summary>
    /// Gets the number of classes (ways) for tasks in this batch.
    /// </summary>
    public int NumWays { get; }

    /// <summary>
    /// Gets the number of shots per class for tasks in this batch.
    /// </summary>
    public int NumShots { get; }

    /// <summary>
    /// Gets the number of query examples per class for tasks in this batch.
    /// </summary>
    public int NumQueryPerClass { get; }

    /// <summary>
    /// Gets a task at the specified index.
    /// </summary>
    /// <param name="index">The zero-based index of the task.</param>
    /// <returns>The task at the specified index.</returns>
    public ITask<T, TInput, TOutput> this[int index] => Tasks[index];
}
