using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements PackNet for continual learning through parameter isolation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> PackNet achieves continual learning by dynamically pruning
/// and freezing network weights. After learning each task, unimportant weights are pruned,
/// and the remaining weights are frozen. New tasks can only use the pruned (free) weights,
/// effectively isolating each task's parameters.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Train the network on Task 1 using all available weights.</description></item>
/// <item><description>After training, prune weights with smallest magnitude (keep top k%).</description></item>
/// <item><description>Freeze the remaining (important) weights - they now belong to Task 1.</description></item>
/// <item><description>For Task 2, only train on the pruned (free) weights.</description></item>
/// <item><description>Repeat: prune, freeze, move to next task.</description></item>
/// </list>
///
/// <para><b>Key Concepts:</b></para>
/// <list type="bullet">
/// <item><description><b>Free Weights:</b> Weights available for the current task (pruned in previous tasks).</description></item>
/// <item><description><b>Frozen Weights:</b> Weights dedicated to previous tasks (cannot be modified).</description></item>
/// <item><description><b>Pruning Ratio:</b> Percentage of free weights to prune after each task.</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Zero forgetting - previous task weights are completely protected.</description></item>
/// <item><description>No replay or regularization needed during training.</description></item>
/// <item><description>Network compression as a side effect of pruning.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Mallya, A. and Lazebnik, S. "PackNet: Adding Multiple Tasks to a
/// Single Network by Iterative Pruning" (2018). CVPR.</para>
/// </remarks>
public class PackNet<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Dictionary<int, HashSet<int>> _taskMasks;  // Which weights belong to which task
    private HashSet<int> _freeWeights;                          // Currently available weights
    private readonly double _pruningRatio;
    private double _lambda;
    private int _currentTaskId;
    private int _totalParameters;

    /// <summary>
    /// Initializes a new instance of the PackNet class.
    /// </summary>
    /// <param name="pruningRatio">Ratio of weights to prune after each task (default: 0.5 = 50%).</param>
    /// <param name="lambda">Strength of weight freezing enforcement (default: 1000.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Pruning ratio determines how much of the remaining capacity to use per task.</description></item>
    /// <item><description>With ratio 0.5, you can fit approximately log2(1/ratio) tasks before running out.</description></item>
    /// <item><description>Lambda should be very high to enforce weight freezing.</description></item>
    /// </list>
    /// </remarks>
    public PackNet(double pruningRatio = 0.5, double lambda = 1000.0)
    {
        if (pruningRatio <= 0 || pruningRatio >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(pruningRatio),
                "Pruning ratio must be between 0 and 1 (exclusive)");
        }

        _numOps = MathHelper.GetNumericOperations<T>();
        _taskMasks = [];
        _freeWeights = [];
        _pruningRatio = pruningRatio;
        _lambda = lambda;
        _currentTaskId = -1;
        _totalParameters = 0;
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets the number of tasks stored.
    /// </summary>
    public int TaskCount => _taskMasks.Count;

    /// <summary>
    /// Gets the pruning ratio.
    /// </summary>
    public double PruningRatio => _pruningRatio;

    /// <summary>
    /// Gets the number of free weights available for new tasks.
    /// </summary>
    public int FreeWeightCount => _freeWeights.Count;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int TotalParameters => _totalParameters;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        Guard.NotNull(network);

        _currentTaskId = taskId;
        _totalParameters = network.ParameterCount;

        // Initialize free weights if this is the first task
        if (_freeWeights.Count == 0 && _taskMasks.Count == 0)
        {
            _freeWeights = new HashSet<int>(Enumerable.Range(0, _totalParameters));
        }
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        Guard.NotNull(network);

        // Prune weights and assign to current task
        var currentParams = network.GetParameters();
        var taskWeights = PruneAndAssign(currentParams);

        _taskMasks[taskId] = taskWeights;

        // Update free weights (remove those assigned to current task)
        _freeWeights.ExceptWith(taskWeights);
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        // PackNet uses gradient masking, not loss-based regularization
        return _numOps.Zero;
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        Guard.NotNull(network);
        Guard.NotNull(gradients);

        // Zero out gradients for frozen weights (belonging to previous tasks)
        for (int i = 0; i < gradients.Length; i++)
        {
            if (!_freeWeights.Contains(i))
            {
                // This weight is frozen (belongs to a previous task)
                gradients[i] = _numOps.Zero;
            }
        }

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _taskMasks.Clear();
        _freeWeights.Clear();
        _currentTaskId = -1;
        _totalParameters = 0;
    }

    /// <summary>
    /// Gets the weight mask for a specific task.
    /// </summary>
    /// <param name="taskId">The task ID.</param>
    /// <returns>Set of weight indices belonging to the task.</returns>
    public HashSet<int> GetTaskMask(int taskId)
    {
        return _taskMasks.TryGetValue(taskId, out var mask)
            ? new HashSet<int>(mask)
            : [];
    }

    /// <summary>
    /// Applies the task-specific mask to network parameters for inference.
    /// </summary>
    /// <param name="network">The neural network.</param>
    /// <param name="taskId">The task to configure for.</param>
    /// <remarks>
    /// This zeros out weights that don't belong to the specified task,
    /// which is useful for task-specific inference.
    /// </remarks>
    public void ApplyTaskMask(INeuralNetwork<T> network, int taskId)
    {
        Guard.NotNull(network);

        if (!_taskMasks.ContainsKey(taskId))
        {
            throw new ArgumentException($"Task {taskId} not found", nameof(taskId));
        }

        var parameters = network.GetParameters();
        var taskWeights = GetAllWeightsUpToTask(taskId);

        for (int i = 0; i < parameters.Length; i++)
        {
            if (!taskWeights.Contains(i))
            {
                parameters[i] = _numOps.Zero;
            }
        }

        network.SetParameters(parameters);
    }

    /// <summary>
    /// Gets all weights used by tasks up to and including the specified task.
    /// </summary>
    private HashSet<int> GetAllWeightsUpToTask(int maxTaskId)
    {
        var result = new HashSet<int>();
        foreach (var kvp in _taskMasks)
        {
            if (kvp.Key <= maxTaskId)
            {
                result.UnionWith(kvp.Value);
            }
        }
        return result;
    }

    /// <summary>
    /// Prunes weights and assigns the important ones to the current task.
    /// </summary>
    private HashSet<int> PruneAndAssign(Vector<T> parameters)
    {
        // Get magnitudes of free weights
        var freeWeightMagnitudes = new List<(int index, double magnitude)>();
        foreach (var idx in _freeWeights)
        {
            var magnitude = Math.Abs(_numOps.ToDouble(parameters[idx]));
            freeWeightMagnitudes.Add((idx, magnitude));
        }

        // Sort by magnitude descending
        freeWeightMagnitudes.Sort((a, b) => b.magnitude.CompareTo(a.magnitude));

        // Keep top (1 - pruningRatio) weights for this task
        var keepCount = (int)(freeWeightMagnitudes.Count * (1 - _pruningRatio));
        keepCount = Math.Max(1, keepCount); // Keep at least 1 weight

        var taskWeights = new HashSet<int>();
        for (int i = 0; i < keepCount && i < freeWeightMagnitudes.Count; i++)
        {
            taskWeights.Add(freeWeightMagnitudes[i].index);
        }

        return taskWeights;
    }

    /// <summary>
    /// Gets statistics about weight allocation across tasks.
    /// </summary>
    /// <returns>Dictionary mapping task IDs to weight counts.</returns>
    public Dictionary<int, int> GetWeightAllocationStats()
    {
        var stats = new Dictionary<int, int>();
        foreach (var kvp in _taskMasks)
        {
            stats[kvp.Key] = kvp.Value.Count;
        }
        stats[-1] = _freeWeights.Count; // -1 represents free weights
        return stats;
    }
}
