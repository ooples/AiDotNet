namespace AiDotNet.Data.Structures;

/// <summary>
/// A meta-learning task implementation that tracks episode and task IDs for episodic training.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// EpisodicMetaLearningTask extends the basic task functionality by adding episode and task
/// tracking capabilities. This is essential for meta-learning algorithms that train on
/// episodes containing multiple tasks, enabling better organization and analysis of the
/// training process.
/// </para>
/// <para>
/// <b>Key Features:</b>
/// - Episode ID tracking for grouping related tasks
/// - Task ID for ordering within episodes
/// - Task difficulty estimation
/// - Performance metrics tracking
/// - Temporal information for curriculum learning
/// </para>
/// <para>
/// <b>Common Use Cases:</b>
/// - MAML and its variants with episodic training
/// - Curriculum meta-learning
/// - Meta-RL where tasks are presented in sequence
/// - Benchmark tracking and analysis
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an episodic task
/// var task = new EpisodicMetaLearningTask&lt;double&gt;(episodeId: 42, taskId: 3, "mini_imagenet_task");
///
/// // Set up data as usual
/// task.SupportInput = supportData;
/// task.SupportOutput = supportLabels;
/// task.QueryInput = queryData;
/// task.QueryOutput = queryLabels;
///
/// // Set episode-specific metadata
/// task.SetMetadata("curriculum_stage", 2);
/// task.SetMetadata("task_difficulty", 0.7);
/// task.SetMetadata("creation_timestamp", DateTime.UtcNow);
/// </code>
/// </example>
public class EpisodicMetaLearningTask<T, TInput, TOutput> : MetaLearningTaskBase<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private int _episodeId;
    private int _taskId;
    private DateTime _creationTime;
    private double? _difficulty;
    private Dictionary<string, double> _performanceMetrics;

    /// <summary>
    /// Initializes a new instance of the EpisodicMetaLearningTask class.
    /// </summary>
    /// <param name="episodeId">The episode identifier this task belongs to.</param>
    /// <param name="taskId">The task identifier within the episode.</param>
    /// <param name="name">Optional name for the task.</param>
    public EpisodicMetaLearningTask(int episodeId, int taskId, string? name = null)
        : base(name)
    {
        _episodeId = episodeId;
        _taskId = taskId;
        _creationTime = DateTime.UtcNow;
        _performanceMetrics = new Dictionary<string, double>();

        // Add episode information to metadata
        SetMetadata("episode_id", episodeId);
        SetMetadata("task_id", taskId);
        SetMetadata("creation_time", _creationTime);
    }

    /// <summary>
    /// Gets or sets the episode identifier.
    /// </summary>
    /// <value>
    /// The episode this task belongs to. Tasks within the same episode
    /// are typically processed together during meta-training.
    /// </value>
    public int EpisodeId
    {
        get => _episodeId;
        set
        {
            _episodeId = value;
            SetMetadata("episode_id", value);
        }
    }

    /// <summary>
    /// Gets or sets the task identifier within the episode.
    /// </summary>
    /// <value>
    /// The sequential ID of this task within its episode.
    /// </value>
    public int TaskId
    {
        get => _taskId;
        set
        {
            _taskId = value;
            SetMetadata("task_id", value);
        }
    }

    /// <summary>
    /// Gets the creation timestamp of the task.
    /// </summary>
    /// <value>
    /// The UTC timestamp when this task was created.
    /// </value>
    public DateTime CreationTime => _creationTime;

    /// <summary>
    /// Gets or sets the estimated difficulty of this task.
    /// </summary>
    /// <value>
    /// Difficulty score between 0 (easy) and 1 (hard), or null if not set.
    /// </value>
    public double? Difficulty
    {
        get => _difficulty;
        set
        {
            _difficulty = value;
            if (value.HasValue)
                SetMetadata("difficulty", value.Value);
        }
    }

    /// <summary>
    /// Gets the performance metrics dictionary for this task.
    /// </summary>
    /// <value>
    /// Dictionary of metric names to values (e.g., "accuracy", "loss").
    /// </value>
    public IReadOnlyDictionary<string, double> PerformanceMetrics => _performanceMetrics;

    /// <summary>
    /// Records a performance metric for this task.
    /// </summary>
    /// <param name="metricName">The name of the metric (e.g., "accuracy", "loss").</param>
    /// <param name="value">The metric value.</param>
    /// <exception cref="ArgumentNullException">Thrown when metricName is null.</exception>
    public void RecordMetric(string metricName, double value)
    {
        if (metricName == null)
            throw new ArgumentNullException(nameof(metricName));

        _performanceMetrics[metricName] = value;
        SetMetadata($"metric_{metricName}", value);
    }

    /// <summary>
    /// Gets a recorded performance metric.
    /// </summary>
    /// <param name="metricName">The name of the metric.</param>
    /// <returns>The metric value if it exists, otherwise null.</returns>
    public double? GetMetric(string metricName)
    {
        return _performanceMetrics.TryGetValue(metricName, out var value) ? value : null;
    }

    /// <summary>
    /// Advances to the next task within the same episode.
    /// </summary>
    /// <returns>A new EpisodicMetaLearningTask with incremented TaskId.</returns>
    public EpisodicMetaLearningTask<T, TInput, TOutput> CreateNextTask()
    {
        var nextTask = new EpisodicMetaLearningTask<T, TInput, TOutput>(
            _episodeId,
            _taskId + 1,
            Name?.Replace($"_task{_taskId}", $"_task{_taskId + 1}")
        );
        nextTask.Difficulty = Difficulty;
        return nextTask;
    }

    /// <summary>
    /// Creates a task for the next episode.
    /// </summary>
    /// <param name="taskName">Optional name for the new episode's first task.</param>
    /// <returns>A new EpisodicMetaLearningTask for the next episode.</returns>
    public EpisodicMetaLearningTask<T, TInput, TOutput> CreateNextEpisodeTask(string? taskName = null)
    {
        return new EpisodicMetaLearningTask<T, TInput, TOutput>(
            _episodeId + 1,
            0,
            taskName
        );
    }

    /// <summary>
    /// Creates a string representation with episode and task information.
    /// </summary>
    /// <returns>String containing episode, task, and configuration details.</returns>
    public override string ToString()
    {
        var name = string.IsNullOrEmpty(Name) ? "EpisodicMetaLearningTask" : Name;
        var ways = TryGetMetadata<int>("num_ways", out var w) ? w : -1;
        var shots = TryGetMetadata<int>("num_shots", out var s) ? s : -1;

        var result = $"{name} (Episode: {_episodeId}, Task: {_taskId}";

        if (ways > 0 && shots > 0)
            result += $", {ways}-way {shots}-shot";

        if (Difficulty.HasValue)
            result += $", Difficulty: {Difficulty.Value:F2}";

        result += ")";

        return result;
    }
}