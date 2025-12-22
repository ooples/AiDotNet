using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for PackNet strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PackNetOptions<T>
{
    /// <summary>
    /// Gets or sets the pruning ratio (fraction of parameters to prune after each task).
    /// Values between 0.5 and 0.9 are typical. Higher values allow more tasks.
    /// </summary>
    public double? PruningRatio { get; set; }

    /// <summary>
    /// Gets or sets whether to use magnitude-based pruning.
    /// If false, uses gradient-based importance for pruning.
    /// </summary>
    public bool? UseMagnitudePruning { get; set; }

    /// <summary>
    /// Gets or sets the number of fine-tuning epochs after pruning.
    /// </summary>
    public int? FineTuningEpochs { get; set; }

    /// <summary>
    /// Gets or sets whether to allow retraining of pruned parameters.
    /// If true, pruned parameters can be reactivated for new tasks.
    /// </summary>
    public bool? AllowPrunedReuse { get; set; }

    /// <summary>
    /// Gets or sets the minimum weight magnitude to keep (absolute value).
    /// Parameters below this threshold are pruned regardless of ratio.
    /// </summary>
    public double? MinWeightMagnitude { get; set; }

    /// <summary>
    /// Gets or sets whether to use layer-wise pruning ratios.
    /// If true, each layer is pruned according to its own ratio.
    /// </summary>
    public bool? LayerWisePruning { get; set; }
}

/// <summary>
/// PackNet strategy for continual learning through network pruning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> PackNet "packs" multiple tasks into a single network
/// by pruning unimportant parameters after each task and using the freed capacity
/// for new tasks. It's like having multiple neural networks compressed into one.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Train the network on task 1</description></item>
/// <item><description>Prune less important parameters (e.g., keep top 50% by magnitude)</description></item>
/// <item><description>Freeze the remaining parameters (they now "belong" to task 1)</description></item>
/// <item><description>Train on task 2 using only the pruned (freed) parameters</description></item>
/// <item><description>Repeat pruning and freezing for each new task</description></item>
/// </list>
///
/// <para><b>The Math:</b></para>
/// <para>After training on task t:</para>
/// <para>1. Compute importance: I_i = |θ_i| (magnitude-based) or I_i = |∂L/∂θ_i| * |θ_i|</para>
/// <para>2. Prune: mask_t = I > percentile(I, prune_ratio * 100)</para>
/// <para>3. Freeze: θ_frozen = θ * mask_t</para>
/// <para>4. Free capacity: available = (1 - mask_t)</para>
///
/// <para><b>Comparison to Other Methods:</b></para>
/// <list type="bullet">
/// <item><description><b>PNN:</b> Creates new columns (linear memory growth)</description></item>
/// <item><description><b>EWC/MAS/SI:</b> Uses regularization (allows some forgetting)</description></item>
/// <item><description><b>PackNet:</b> Fixed network size, but limited task capacity</description></item>
/// </list>
///
/// <para>PackNet has O(1) memory per task but is limited by network capacity.
/// Best for scenarios where network size is constrained and tasks can share features.</para>
/// </remarks>
public class PackNet<T, TInput, TOutput> : ContinualLearningStrategyBase<T, TInput, TOutput>
{
    private readonly T _pruningRatio;
    private readonly bool _useMagnitudePruning;
    private readonly int _fineTuningEpochs;
    private readonly bool _allowPrunedReuse;
    private readonly T _minWeightMagnitude;
    private readonly bool _layerWisePruning;

    // Mask tracking: 0 = available, task_id = assigned to that task
    private int[]? _parameterOwnership;

    // Accumulated gradient importance for gradient-based pruning
    private Vector<T>? _gradientImportance;
    private int _gradientCount;

    // Parameters after each task's training (before pruning)
    private Vector<T>? _preTaskParameters;

    // Task masks for inference
    private readonly Dictionary<int, bool[]> _taskMasks = [];

    /// <summary>
    /// Initializes a new PackNet strategy with default options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="pruningRatio">Fraction of parameters to prune (0.5-0.9 typical).</param>
    public PackNet(
        ILossFunction<T> lossFunction,
        double pruningRatio = 0.75)
        : this(lossFunction, new PackNetOptions<T> { PruningRatio = pruningRatio })
    {
    }

    /// <summary>
    /// Initializes a new PackNet strategy with custom options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options.</param>
    public PackNet(ILossFunction<T> lossFunction, PackNetOptions<T>? options = null)
        : base(lossFunction)
    {
        var opts = options ?? new PackNetOptions<T>();

        _pruningRatio = NumOps.FromDouble(opts.PruningRatio ?? 0.75);
        _useMagnitudePruning = opts.UseMagnitudePruning ?? true;
        _fineTuningEpochs = opts.FineTuningEpochs ?? 5;
        _allowPrunedReuse = opts.AllowPrunedReuse ?? true;
        _minWeightMagnitude = NumOps.FromDouble(opts.MinWeightMagnitude ?? 1e-6);
        _layerWisePruning = opts.LayerWisePruning ?? false;
    }

    /// <inheritdoc/>
    public override string Name => "PackNet";

    /// <inheritdoc/>
    public override bool RequiresMemoryBuffer => false;

    /// <inheritdoc/>
    public override bool ModifiesArchitecture => false; // Same architecture, different active parameters

    /// <inheritdoc/>
    public override long MemoryUsageBytes
    {
        get
        {
            long bytes = 0;
            if (_parameterOwnership != null)
                bytes += _parameterOwnership.Length * sizeof(int);
            bytes += EstimateVectorMemory(_gradientImportance);
            bytes += EstimateVectorMemory(_preTaskParameters);

            // Task masks
            foreach (var mask in _taskMasks.Values)
            {
                bytes += mask.Length; // bool array
            }

            return bytes;
        }
    }

    /// <summary>
    /// Gets the pruning ratio.
    /// </summary>
    public T PruningRatio => _pruningRatio;

    /// <summary>
    /// Gets the available parameter capacity (fraction of parameters not yet assigned).
    /// </summary>
    public double AvailableCapacity
    {
        get
        {
            if (_parameterOwnership == null) return 1.0;
            int available = _parameterOwnership.Count(o => o == 0);
            return (double)available / _parameterOwnership.Length;
        }
    }

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        int numParams = model.ParameterCount;

        if (_parameterOwnership == null)
        {
            // First task: all parameters are available
            _parameterOwnership = new int[numParams];
        }
        else if (_parameterOwnership.Length != numParams)
        {
            throw new InvalidOperationException(
                $"Parameter count changed: expected {_parameterOwnership.Length}, got {numParams}");
        }

        // Initialize gradient importance accumulator
        _gradientImportance = new Vector<T>(numParams);
        for (int i = 0; i < numParams; i++)
        {
            _gradientImportance[i] = NumOps.Zero;
        }
        _gradientCount = 0;

        // Store pre-task parameters
        _preTaskParameters = CloneVector(model.GetParameters());

        RecordMetric($"Task{TaskCount}_AvailableCapacity", AvailableCapacity);
        RecordMetric($"Task{TaskCount}_PrepareTime", DateTime.UtcNow);
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        // PackNet doesn't use regularization - it uses masks to prevent interference
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        if (_parameterOwnership == null)
        {
            return gradients;
        }

        // Accumulate gradient importance (for gradient-based pruning)
        if (!_useMagnitudePruning && _gradientImportance != null)
        {
            for (int i = 0; i < Math.Min(gradients.Length, _gradientImportance.Length); i++)
            {
                double absGrad = Math.Abs(Convert.ToDouble(gradients[i]));
                _gradientImportance[i] = NumOps.Add(
                    _gradientImportance[i],
                    NumOps.FromDouble(absGrad));
            }
            _gradientCount++;
        }

        // Zero out gradients for parameters owned by previous tasks
        var adjustedGradients = CloneVector(gradients);

        for (int i = 0; i < Math.Min(adjustedGradients.Length, _parameterOwnership.Length); i++)
        {
            if (_parameterOwnership[i] > 0 && _parameterOwnership[i] <= TaskCount)
            {
                // Parameter is frozen for a previous task
                adjustedGradients[i] = NumOps.Zero;
            }
        }

        return adjustedGradients;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        var currentParams = model.GetParameters();
        int taskId = TaskCount + 1; // 1-indexed task IDs

        // Compute importance scores for pruning
        var importance = ComputeImportanceScores(currentParams);

        // Determine which parameters to keep for this task
        var taskMask = ComputeTaskMask(importance, currentParams);

        // Update parameter ownership
        UpdateParameterOwnership(taskMask, taskId);

        // Store the task mask for inference
        _taskMasks[taskId] = taskMask;

        TaskCount++;

        // Record metrics
        int keptParams = taskMask.Count(m => m);
        RecordMetric($"Task{TaskCount}_KeptParameters", keptParams);
        RecordMetric($"Task{TaskCount}_PrunedParameters", taskMask.Length - keptParams);
        RecordMetric($"Task{TaskCount}_RemainingCapacity", AvailableCapacity);
    }

    /// <summary>
    /// Computes importance scores for each parameter.
    /// </summary>
    private Vector<T> ComputeImportanceScores(Vector<T> parameters)
    {
        var importance = new Vector<T>(parameters.Length);

        if (_useMagnitudePruning)
        {
            // Magnitude-based: importance = |θ|
            for (int i = 0; i < parameters.Length; i++)
            {
                double absVal = Math.Abs(Convert.ToDouble(parameters[i]));
                importance[i] = NumOps.FromDouble(absVal);
            }
        }
        else if (_gradientImportance != null && _gradientCount > 0)
        {
            // Gradient-based: importance = |∂L/∂θ| * |θ|
            for (int i = 0; i < parameters.Length; i++)
            {
                double avgGrad = Convert.ToDouble(_gradientImportance[i]) / _gradientCount;
                double magnitude = Math.Abs(Convert.ToDouble(parameters[i]));
                importance[i] = NumOps.FromDouble(avgGrad * magnitude);
            }
        }
        else
        {
            // Fallback to magnitude-based
            for (int i = 0; i < parameters.Length; i++)
            {
                double absVal = Math.Abs(Convert.ToDouble(parameters[i]));
                importance[i] = NumOps.FromDouble(absVal);
            }
        }

        return importance;
    }

    /// <summary>
    /// Computes the mask for which parameters to keep for this task.
    /// </summary>
    private bool[] ComputeTaskMask(Vector<T> importance, Vector<T> parameters)
    {
        var mask = new bool[parameters.Length];
        double pruneRatio = Convert.ToDouble(_pruningRatio);
        double minMagnitude = Convert.ToDouble(_minWeightMagnitude);

        // Get indices of available parameters only
        var availableIndices = new List<int>();
        for (int i = 0; i < parameters.Length; i++)
        {
            if (_parameterOwnership![i] == 0)
            {
                availableIndices.Add(i);
            }
        }

        if (availableIndices.Count == 0)
        {
            // No available parameters - can't learn this task
            return mask;
        }

        // Sort available parameters by importance
        var sortedIndices = availableIndices
            .OrderByDescending(i => Convert.ToDouble(importance[i]))
            .ToList();

        // Keep top (1 - pruneRatio) fraction
        int keepCount = (int)((1 - pruneRatio) * sortedIndices.Count);
        keepCount = Math.Max(1, keepCount); // Keep at least 1 parameter

        for (int i = 0; i < keepCount; i++)
        {
            int paramIdx = sortedIndices[i];
            double magnitude = Math.Abs(Convert.ToDouble(parameters[paramIdx]));

            // Only keep if above minimum magnitude
            if (magnitude >= minMagnitude)
            {
                mask[paramIdx] = true;
            }
        }

        return mask;
    }

    /// <summary>
    /// Updates parameter ownership based on the task mask.
    /// </summary>
    private void UpdateParameterOwnership(bool[] taskMask, int taskId)
    {
        for (int i = 0; i < taskMask.Length; i++)
        {
            if (taskMask[i] && _parameterOwnership![i] == 0)
            {
                // Assign this parameter to the current task
                _parameterOwnership[i] = taskId;
            }
        }
    }

    /// <summary>
    /// Gets the mask for a specific task (which parameters to use during inference).
    /// </summary>
    /// <param name="taskId">The task ID (1-indexed).</param>
    public bool[]? GetTaskMask(int taskId)
    {
        return _taskMasks.TryGetValue(taskId, out var mask) ? mask : null;
    }

    /// <summary>
    /// Gets ownership information for all parameters.
    /// </summary>
    /// <returns>Array where each element indicates which task owns that parameter (0 = unassigned).</returns>
    public int[]? GetParameterOwnership() => _parameterOwnership?.ToArray();

    /// <summary>
    /// Gets the count of parameters assigned to each task.
    /// </summary>
    public Dictionary<int, int> GetParameterCountByTask()
    {
        var counts = new Dictionary<int, int>();

        if (_parameterOwnership == null) return counts;

        foreach (int owner in _parameterOwnership)
        {
            if (!counts.ContainsKey(owner))
                counts[owner] = 0;
            counts[owner]++;
        }

        return counts;
    }

    /// <summary>
    /// Applies the task mask to parameters for inference on a specific task.
    /// </summary>
    /// <param name="parameters">The full parameter vector.</param>
    /// <param name="taskId">The task to get parameters for.</param>
    /// <returns>Parameters with non-task parameters zeroed out.</returns>
    public Vector<T> ApplyTaskMask(Vector<T> parameters, int taskId)
    {
        var masked = CloneVector(parameters);

        if (!_taskMasks.TryGetValue(taskId, out var mask))
        {
            return masked;
        }

        for (int i = 0; i < Math.Min(masked.Length, mask.Length); i++)
        {
            if (!mask[i])
            {
                masked[i] = NumOps.Zero;
            }
        }

        return masked;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _parameterOwnership = null;
        _gradientImportance = null;
        _gradientCount = 0;
        _preTaskParameters = null;
        _taskMasks.Clear();
    }

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();
        state["PruningRatio"] = Convert.ToDouble(_pruningRatio);
        state["UseMagnitudePruning"] = _useMagnitudePruning;
        state["FineTuningEpochs"] = _fineTuningEpochs;
        state["TaskCount"] = _taskMasks.Count;
        return state;
    }
}
