using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;

namespace AiDotNet.MetaLearning.Adapters;

/// <summary>
/// Adapter that converts between Task and MetaLearningTask abstractions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// This adapter bridges the gap between two meta-learning abstractions:
/// - <see cref="ITask{T, TInput, TOutput}"/>: Simple task interface with support and query sets
/// - <see cref="MetaLearningTask{T, TInput, TOutput}"/>: Richer task interface with episode information
/// </para>
/// <para>
/// <b>For Beginners:</b> This adapter converts between different task representations:
///
/// Simple Task (ITask):
/// - Support set: Labeled examples for adaptation
/// - Query set: Examples for evaluation
///
/// Meta-Learning Task (MetaLearningTask):
/// - Support set: Same as above
/// - Query set: Same as above
/// - Episode ID: Which meta-training episode this belongs to
/// - Task ID: Which specific task within the episode
/// - Task metadata: Additional information about the task
///
/// The adapter preserves all the data while adding the meta-learning specific context.
/// </para>
/// </remarks>
public class TaskToMetaLearningTaskAdapter<T, TInput, TOutput>
{
    private int _currentEpisodeId = 0;
    private int _currentTaskId = 0;

    /// <summary>
    /// Converts a Task to a MetaLearningTask.
    /// </summary>
    /// <param name="task">The task to convert.</param>
    /// <param name="episodeId">The episode identifier. If null, auto-increments.</param>
    /// <param name="taskId">The task identifier within the episode. If null, auto-increments.</param>
    /// <returns>A MetaLearningTask with the converted data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public static MetaLearningTask<T, TInput, TOutput> Convert(
        ITask<T, TInput, TOutput> task,
        int? episodeId = null,
        int? taskId = null)
    {
        if (task == null)
            throw new ArgumentNullException(nameof(task));

        return new MetaLearningTask<T, TInput, TOutput>(
            supportSetX: task.SupportInput,
            supportSetY: task.SupportOutput,
            querySetX: task.QueryInput,
            querySetY: task.QueryOutput,
            episodeId: episodeId ?? 0,
            taskId: taskId ?? 0,
            taskName: task.Name,
            metadata: task.Metadata
        );
    }

    /// <summary>
    /// Converts a TaskBatch to a list of MetaLearningTasks.
    /// </summary>
    /// <param name="taskBatch">The batch of tasks to convert.</param>
    /// <param name="episodeId">The episode identifier for all tasks. If null, auto-increments.</param>
    /// <returns>A list of MetaLearningTasks.</returns>
    /// <exception cref="ArgumentNullException">Thrown when taskBatch is null.</exception>
    public static List<MetaLearningTask<T, TInput, TOutput>> ConvertBatch(
        TaskBatch<T, TInput, TOutput> taskBatch,
        int? episodeId = null)
    {
        if (taskBatch == null)
            throw new ArgumentNullException(nameof(taskBatch));

        var metaTasks = new List<MetaLearningTask<T, TInput, TOutput>>();

        for (int i = 0; i < taskBatch.Tasks.Count; i++)
        {
            var task = taskBatch.Tasks[i];
            var metaTask = Convert(task, episodeId, i);
            metaTasks.Add(metaTask);
        }

        return metaTasks;
    }

    /// <summary>
    /// Creates an adapter instance with automatic ID tracking.
    /// </summary>
    public TaskToMetaLearningTaskAdapter()
    {
    }

    /// <summary>
    /// Converts a Task to a MetaLearningTask with automatic ID tracking.
    /// </summary>
    /// <param name="task">The task to convert.</param>
    /// <returns>A MetaLearningTask with auto-generated episode and task IDs.</returns>
    public MetaLearningTask<T, TInput, TOutput> ConvertWithTracking(ITask<T, TInput, TOutput> task)
    {
        var metaTask = Convert(task, _currentEpisodeId, _currentTaskId);
        _currentTaskId++;

        // Reset task ID when it exceeds reasonable bounds
        if (_currentTaskId > 10000)
        {
            _currentTaskId = 0;
            _currentEpisodeId++;
        }

        return metaTask;
    }

    /// <summary>
    /// Advances to the next episode and resets task ID counter.
    /// </summary>
    public void NextEpisode()
    {
        _currentEpisodeId++;
        _currentTaskId = 0;
    }

    /// <summary>
    /// Resets the adapter to episode 0, task 0.
    /// </summary>
    public void Reset()
    {
        _currentEpisodeId = 0;
        _currentTaskId = 0;
    }

    /// <summary>
    /// Gets the current episode ID.
    /// </summary>
    public int CurrentEpisodeId => _currentEpisodeId;

    /// <summary>
    /// Gets the current task ID within the episode.
    /// </summary>
    public int CurrentTaskId => _currentTaskId;

    /// <summary>
    /// Converts a MetaLearningTask back to a Task.
    /// </summary>
    /// <param name="metaTask">The meta-learning task to convert.</param>
    /// <returns>A Task with the converted data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when metaTask is null.</exception>
    public static ITask<T, TInput, TOutput> ConvertBack(MetaLearningTask<T, TInput, TOutput> metaTask)
    {
        if (metaTask == null)
            throw new ArgumentNullException(nameof(metaTask));

        return new Task<T, TInput, TOutput>(
            supportInput: metaTask.SupportSetX,
            supportOutput: metaTask.SupportSetY,
            queryInput: metaTask.QuerySetX,
            queryOutput: metaTask.QuerySetY,
            name: metaTask.TaskName,
            metadata: metaTask.Metadata
        );
    }
}