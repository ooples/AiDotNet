namespace AiDotNet.Data.Structures;

/// <summary>
/// A meta-learning task implementation for continual learning scenarios with task boundaries.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// ContinualMetaLearningTask extends the basic task functionality by adding support for
/// continual learning scenarios where tasks arrive sequentially and the model must
/// adapt without forgetting previous knowledge. This implementation tracks task boundaries,
/// catastrophic forgetting metrics, and knowledge retention.
/// </para>
/// <para>
/// <b>Key Features:</b>
/// - Task sequence tracking
/// - Knowledge retention metrics
/// - Catastrophic forgetting detection
/// - Task boundary identification
/// - Continual learning strategies
/// - Memory replay support
/// </para>
/// <para>
/// <b>Common Use Cases:</b>
/// - Continual meta-learning (C-Meta)
/// - Online meta-learning
/// - Lifelong learning scenarios
/// - Streaming task adaptation
/// - Incremental meta-knowledge acquisition
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a continual learning task
/// var task = new ContinualMetaLearningTask&lt;double&gt;(
///     taskSequenceId: 15,
///     "online_learning_stream"
/// );
///
/// // Set up data as usual
/// task.SupportInput = supportData;
/// task.SupportOutput = supportLabels;
/// task.QueryInput = queryData;
/// task.QueryOutput = queryLabels;
///
/// // Track continual learning metrics
/// task.SetKnowledgeRetention(0.85);
/// task.MarkAsBoundaryTask();
/// task.SetReplayBufferSize(1000);
/// </code>
/// </example>
public class ContinualMetaLearningTask<T, TInput, TOutput> : MetaLearningTaskBase<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private int _taskSequenceId;
    private DateTime _arrivalTime;
    private bool _isBoundaryTask;
    private double? _knowledgeRetention;
    private double? _forgettingRate;
    private int _replayBufferSize;
    private readonly List<string> _previousTaskIds;
    private readonly Dictionary<string, double> _continualMetrics;

    /// <summary>
    /// Initializes a new instance of the ContinualMetaLearningTask class.
    /// </summary>
    /// <param name="taskSequenceId">The sequence ID of this task in the continual learning stream.</param>
    /// <param name="name">Optional name for the task.</param>
    public ContinualMetaLearningTask(int taskSequenceId, string? name = null)
        : base(name)
    {
        _taskSequenceId = taskSequenceId;
        _arrivalTime = DateTime.UtcNow;
        _isBoundaryTask = false;
        _replayBufferSize = 0;
        _previousTaskIds = new List<string>();
        _continualMetrics = new Dictionary<string, double>();

        // Add continual learning information to metadata
        SetMetadata("sequence_id", taskSequenceId);
        SetMetadata("arrival_time", _arrivalTime);
    }

    /// <summary>
    /// Gets or sets the sequence ID of this task in the continual learning stream.
    /// </summary>
    /// <value>
    /// The position of this task in the sequence of arriving tasks.
    /// </value>
    public int TaskSequenceId
    {
        get => _taskSequenceId;
        set
        {
            _taskSequenceId = value;
            SetMetadata("sequence_id", value);
        }
    }

    /// <summary>
    /// Gets the arrival time of this task.
    /// </summary>
    /// <value>
    /// The UTC timestamp when this task was created/arrived.
    /// </value>
    public DateTime ArrivalTime => _arrivalTime;

    /// <summary>
    /// Gets or sets whether this task is a boundary task (significant task distribution shift).
    /// </summary>
    /// <value>
    /// True if this task marks a boundary in the task distribution.
    /// </value>
    public bool IsBoundaryTask
    {
        get => _isBoundaryTask;
        set
        {
            _isBoundaryTask = value;
            SetMetadata("is_boundary_task", value);
        }
    }

    /// <summary>
    /// Gets or sets the knowledge retention rate from previous tasks.
    /// </summary>
    /// <value>
    /// Retention rate between 0 (complete forgetting) and 1 (perfect retention), or null if not measured.
    /// </value>
    public double? KnowledgeRetention
    {
        get => _knowledgeRetention;
        set
        {
            _knowledgeRetention = value;
            if (value.HasValue)
            {
                SetMetadata("knowledge_retention", value.Value);
                SetMetadata("forgetting_detected", value.Value < 0.7);
            }
        }
    }

    /// <summary>
    /// Gets or sets the catastrophic forgetting rate.
    /// </summary>
    /// <value>
    /// Forgetting rate (0 = no forgetting, positive values indicate forgetting), or null if not measured.
    /// </value>
    public double? ForgettingRate
    {
        get => _forgettingRate;
        set
        {
            _forgettingRate = value;
            if (value.HasValue)
                SetMetadata("forgetting_rate", value.Value);
        }
    }

    /// <summary>
    /// Gets or sets the replay buffer size for this task.
    /// </summary>
    /// <value>
    /// Number of examples stored in replay buffer for future tasks.
    /// </value>
    public int ReplayBufferSize
    {
        get => _replayBufferSize;
        set
        {
            _replayBufferSize = value;
            SetMetadata("replay_buffer_size", value);
        }
    }

    /// <summary>
    /// Gets the collection of previous task IDs that this task builds upon.
    /// </summary>
    /// <value>
    /// Read-only collection of previous task identifiers.
    /// </value>
    public IReadOnlyCollection<string> PreviousTaskIds => _previousTaskIds.AsReadOnly();

    /// <summary>
    /// Gets the continual learning metrics dictionary.
    /// </summary>
    /// <value>
    /// Dictionary of continual learning-specific metrics.
    /// </value>
    public IReadOnlyDictionary<string, double> ContinualMetrics => _continualMetrics;

    /// <summary>
    /// Adds a previous task ID that this task builds upon.
    /// </summary>
    /// <param name="taskId">The ID of the previous task.</param>
    public void AddPreviousTask(string taskId)
    {
        if (!string.IsNullOrEmpty(taskId) && !_previousTaskIds.Contains(taskId))
        {
            _previousTaskIds.Add(taskId);
            SetMetadata($"previous_task_{_previousTaskIds.Count - 1}", taskId);
        }
    }

    /// <summary>
    /// Records a continual learning specific metric.
    /// </summary>
    /// <param name="metricName">The name of the metric (e.g., "plasticity", "stability").</param>
    /// <param name="value">The metric value.</param>
    public void RecordContinualMetric(string metricName, double value)
    {
        if (!string.IsNullOrEmpty(metricName))
        {
            _continualMetrics[metricName] = value;
            SetMetadata($"continual_{metricName}", value);
        }
    }

    /// <summary>
    /// Calculates the time elapsed since the previous task.
    /// </summary>
    /// <param name="previousTaskArrival">The arrival time of the previous task.</param>
    /// <returns>The time elapsed between tasks.</returns>
    public TimeSpan GetTimeSincePreviousTask(DateTime previousTaskArrival)
    {
        return _arrivalTime - previousTaskArrival;
    }

    /// <summary>
    /// Checks if catastrophic forgetting is detected.
    /// </summary>
    /// <param name="threshold">The threshold for detecting forgetting (default: 0.3).</param>
    /// <returns>True if forgetting is detected above the threshold.</returns>
    public bool IsCatastrophicForgettingDetected(double threshold = 0.3)
    {
        return KnowledgeRetention.HasValue && KnowledgeRetention.Value < (1 - threshold) ||
               ForgettingRate.HasValue && ForgettingRate.Value > threshold;
    }

    /// <summary>
    /// Marks this task as a boundary task and records the reason.
    /// </summary>
    /// <param name="boundaryReason">The reason for marking as boundary task.</param>
    public void MarkAsBoundaryTask(string? boundaryReason = null)
    {
        IsBoundaryTask = true;
        if (!string.IsNullOrEmpty(boundaryReason))
            SetMetadata("boundary_reason", boundaryReason);
    }

    /// <summary>
    /// Sets the learning strategy for this task in the continual learning context.
    /// </summary>
    /// <param name="strategy">The learning strategy name.</param>
    /// <param name="parameters">Strategy-specific parameters.</param>
    public void SetLearningStrategy(string strategy, Dictionary<string, object>? parameters = null)
    {
        SetMetadata("learning_strategy", strategy);
        if (parameters != null)
        {
            foreach (var kvp in parameters)
                SetMetadata($"strategy_{kvp.Key}", kvp.Value);
        }
    }

    /// <summary>
    /// Creates a string representation with continual learning information.
    /// </summary>
    /// <returns>String containing continual learning details and metrics.</returns>
    public override string ToString()
    {
        var name = string.IsNullOrEmpty(Name) ? "ContinualMetaLearningTask" : Name;
        var result = $"{name} (Sequence: {_taskSequenceId}";

        if (IsBoundaryTask)
            result += ", Boundary";

        if (KnowledgeRetention.HasValue)
            result += $", Retention: {KnowledgeRetention.Value:P1}";

        if (_replayBufferSize > 0)
            result += $", Replay: {_replayBufferSize}";

        result += ")";

        return result;
    }
}