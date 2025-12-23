using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for Progressive Neural Networks strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PNNOptions<T>
{
    /// <summary>
    /// Gets or sets the number of hidden units per column.
    /// If null, uses the same size as the original model.
    /// </summary>
    public int? HiddenUnitsPerColumn { get; set; }

    /// <summary>
    /// Gets or sets the lateral connection scaling factor.
    /// Controls the influence of previous columns on new columns.
    /// </summary>
    public double? LateralScaling { get; set; }

    /// <summary>
    /// Gets or sets whether to use adapter layers for lateral connections.
    /// Adapter layers reduce the parameter count of lateral connections.
    /// </summary>
    public bool? UseAdapters { get; set; }

    /// <summary>
    /// Gets or sets the adapter bottleneck dimension.
    /// Only used when UseAdapters is true.
    /// </summary>
    public int? AdapterDimension { get; set; }

    /// <summary>
    /// Gets or sets whether to share the output layer across all columns.
    /// </summary>
    public bool? ShareOutputLayer { get; set; }

    /// <summary>
    /// Gets or sets the initialization scale for lateral connection weights.
    /// </summary>
    public double? LateralInitScale { get; set; }
}

/// <summary>
/// Progressive Neural Networks (PNN) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> PNN prevents catastrophic forgetting by creating
/// a new "column" (copy of the network) for each new task. Previous columns
/// are frozen, so old knowledge is never overwritten. New columns can use
/// lateral connections to leverage features from previous columns.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>For the first task, train a standard network (column 1)</description></item>
/// <item><description>For each new task, add a new column with lateral connections to previous columns</description></item>
/// <item><description>Freeze all previous columns and only train the new column</description></item>
/// <item><description>Lateral connections allow the new column to reuse features from old columns</description></item>
/// </list>
///
/// <para><b>The Math:</b></para>
/// <para>For layer l in column k:</para>
/// <para>h_k^l = f(W_k^l * h_k^(l-1) + Σ_{j&lt;k} U_{k,j}^l * h_j^(l-1))</para>
/// <para>Where U_{k,j}^l are the lateral connection weights from column j to column k</para>
///
/// <para><b>Comparison to Other Methods:</b></para>
/// <list type="bullet">
/// <item><description><b>EWC/MAS/SI:</b> Add regularization to protect important parameters</description></item>
/// <item><description><b>PackNet:</b> Prunes and freezes subnetworks for each task</description></item>
/// <item><description><b>PNN:</b> Creates entirely new columns, completely prevents interference</description></item>
/// </list>
///
/// <para>PNN guarantees no forgetting but has linear memory growth with tasks.
/// Best for scenarios with a small number of tasks where zero forgetting is critical.</para>
/// </remarks>
public class ProgressiveNeuralNetworks<T, TInput, TOutput> : ContinualLearningStrategyBase<T, TInput, TOutput>
{
    private readonly int? _hiddenUnitsPerColumn;
    private readonly T _lateralScaling;
    private readonly bool _useAdapters;
    private readonly int _adapterDimension;
    private readonly bool _shareOutputLayer;
    private readonly T _lateralInitScale;

    // Track columns and their parameters
    private readonly List<ColumnInfo> _columns = [];
    private int _baseParameterCount;

    /// <summary>
    /// Information about a PNN column.
    /// </summary>
    private class ColumnInfo
    {
        public int TaskId { get; set; }
        public Vector<T>? Parameters { get; set; }
        public Vector<T>? LateralWeights { get; set; }
        public bool IsFrozen { get; set; }
    }

    /// <summary>
    /// Initializes a new PNN strategy with default options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    public ProgressiveNeuralNetworks(ILossFunction<T> lossFunction)
        : this(lossFunction, null)
    {
    }

    /// <summary>
    /// Initializes a new PNN strategy with custom options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options.</param>
    public ProgressiveNeuralNetworks(ILossFunction<T> lossFunction, PNNOptions<T>? options = null)
        : base(lossFunction)
    {
        var opts = options ?? new PNNOptions<T>();

        _hiddenUnitsPerColumn = opts.HiddenUnitsPerColumn;
        _lateralScaling = NumOps.FromDouble(opts.LateralScaling ?? 0.1);
        _useAdapters = opts.UseAdapters ?? false;
        _adapterDimension = opts.AdapterDimension ?? 64;
        _shareOutputLayer = opts.ShareOutputLayer ?? false;
        _lateralInitScale = NumOps.FromDouble(opts.LateralInitScale ?? 0.01);
    }

    /// <inheritdoc/>
    public override string Name => "PNN";

    /// <inheritdoc/>
    public override bool RequiresMemoryBuffer => false;

    /// <inheritdoc/>
    public override bool ModifiesArchitecture => true;

    /// <inheritdoc/>
    public override long MemoryUsageBytes
    {
        get
        {
            long bytes = 0;
            foreach (var column in _columns)
            {
                bytes += EstimateVectorMemory(column.Parameters);
                bytes += EstimateVectorMemory(column.LateralWeights);
            }
            return bytes;
        }
    }

    /// <summary>
    /// Gets the number of columns (one per task).
    /// </summary>
    public int ColumnCount => _columns.Count;

    /// <summary>
    /// Gets the lateral connection scaling factor.
    /// </summary>
    public T LateralScaling => _lateralScaling;

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        if (_columns.Count == 0)
        {
            // First task: store base parameter count
            _baseParameterCount = model.ParameterCount;
        }
        else
        {
            // Freeze all previous columns
            foreach (var column in _columns)
            {
                column.IsFrozen = true;
            }
        }

        // Create new column for this task
        var newColumn = new ColumnInfo
        {
            TaskId = TaskCount,
            IsFrozen = false
        };

        // If not first task, initialize lateral connections
        if (_columns.Count > 0)
        {
            int lateralSize = ComputeLateralConnectionSize();
            newColumn.LateralWeights = InitializeLateralWeights(lateralSize);
        }

        _columns.Add(newColumn);
        RecordMetric($"Task{TaskCount}_ColumnCreated", DateTime.UtcNow);
        RecordMetric($"Task{TaskCount}_TotalColumns", _columns.Count);
    }

    /// <summary>
    /// Computes the size of lateral connections needed.
    /// </summary>
    private int ComputeLateralConnectionSize()
    {
        // Lateral connections from all previous columns to new column
        // Size depends on whether we use adapters
        if (_useAdapters)
        {
            // Adapter: bottleneck projection reduces parameters
            return _columns.Count * _adapterDimension * 2; // down + up projection
        }
        else
        {
            // Full lateral connections
            int hiddenSize = _hiddenUnitsPerColumn ?? (_baseParameterCount / 10);
            return _columns.Count * hiddenSize;
        }
    }

    /// <summary>
    /// Initializes lateral connection weights with small random values.
    /// </summary>
    private Vector<T> InitializeLateralWeights(int size)
    {
        var weights = new Vector<T>(size);
        double scale = Convert.ToDouble(_lateralInitScale);

        for (int i = 0; i < size; i++)
        {
            // Small random initialization
            double value = (RandomHelper.ThreadSafeRandom.NextDouble() - 0.5) * 2 * scale;
            weights[i] = NumOps.FromDouble(value);
        }

        return weights;
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        // PNN doesn't need regularization loss - it prevents forgetting by architecture
        // Previous columns are frozen, so there's no interference
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        if (_columns.Count <= 1)
        {
            // First task: no modifications needed
            return gradients;
        }

        // For subsequent tasks, we need to:
        // 1. Zero out gradients for frozen column parameters
        // 2. Scale lateral connection gradients

        var adjustedGradients = CloneVector(gradients);
        int frozenParamCount = (_columns.Count - 1) * _baseParameterCount;

        // Zero out gradients for frozen columns
        for (int i = 0; i < Math.Min(frozenParamCount, adjustedGradients.Length); i++)
        {
            adjustedGradients[i] = NumOps.Zero;
        }

        // Scale lateral gradients (if present in the gradient vector)
        // Lateral gradients come after all column parameters
        int lateralStart = _columns.Count * _baseParameterCount;
        for (int i = lateralStart; i < adjustedGradients.Length; i++)
        {
            adjustedGradients[i] = NumOps.Multiply(adjustedGradients[i], _lateralScaling);
        }

        return adjustedGradients;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        // Store the parameters for the current column
        var currentColumn = _columns[^1];
        var allParams = model.GetParameters();

        // Extract parameters for this column
        int startIdx = (_columns.Count - 1) * _baseParameterCount;
        int endIdx = Math.Min(startIdx + _baseParameterCount, allParams.Length);
        int length = endIdx - startIdx;

        if (length > 0)
        {
            currentColumn.Parameters = new Vector<T>(length);
            for (int i = 0; i < length; i++)
            {
                currentColumn.Parameters[i] = allParams[startIdx + i];
            }
        }

        TaskCount++;

        // Record metrics
        RecordMetric($"Task{TaskCount}_TotalParameters", GetTotalParameterCount());
        RecordMetric($"Task{TaskCount}_MemoryMB", MemoryUsageBytes / (1024.0 * 1024.0));
    }

    /// <summary>
    /// Gets the total parameter count across all columns.
    /// </summary>
    public int GetTotalParameterCount()
    {
        int total = _columns.Count * _baseParameterCount;

        // Add lateral connection parameters
        foreach (var column in _columns)
        {
            if (column.LateralWeights != null)
            {
                total += column.LateralWeights.Length;
            }
        }

        return total;
    }

    /// <summary>
    /// Gets information about a specific column.
    /// </summary>
    /// <param name="columnIndex">The column index (0-based).</param>
    public (int taskId, bool isFrozen, int paramCount, int lateralCount) GetColumnInfo(int columnIndex)
    {
        if (columnIndex < 0 || columnIndex >= _columns.Count)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        var column = _columns[columnIndex];
        return (
            column.TaskId,
            column.IsFrozen,
            column.Parameters?.Length ?? _baseParameterCount,
            column.LateralWeights?.Length ?? 0
        );
    }

    /// <summary>
    /// Computes the forward pass through lateral connections.
    /// </summary>
    /// <param name="previousOutputs">Outputs from previous columns.</param>
    /// <param name="columnIndex">The current column index.</param>
    /// <returns>The lateral contribution to add to the current column's activations.</returns>
    public Vector<T> ComputeLateralContribution(List<Vector<T>> previousOutputs, int columnIndex)
    {
        if (columnIndex <= 0 || columnIndex >= _columns.Count)
        {
            return new Vector<T>(0);
        }

        var column = _columns[columnIndex];
        if (column.LateralWeights == null)
        {
            return new Vector<T>(0);
        }

        // Compute weighted sum of previous column outputs
        int outputSize = previousOutputs[0].Length;
        var contribution = new Vector<T>(outputSize);

        for (int i = 0; i < outputSize; i++)
        {
            contribution[i] = NumOps.Zero;
        }

        int weightIdx = 0;
        for (int prevCol = 0; prevCol < columnIndex; prevCol++)
        {
            var prevOutput = previousOutputs[prevCol];
            for (int i = 0; i < Math.Min(outputSize, prevOutput.Length); i++)
            {
                if (weightIdx < column.LateralWeights.Length)
                {
                    var weighted = NumOps.Multiply(prevOutput[i], column.LateralWeights[weightIdx]);
                    var scaled = NumOps.Multiply(weighted, _lateralScaling);
                    contribution[i] = NumOps.Add(contribution[i], scaled);
                    weightIdx++;
                }
            }
        }

        return contribution;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _columns.Clear();
        _baseParameterCount = 0;
    }

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();
        state["ColumnCount"] = _columns.Count;
        state["BaseParameterCount"] = _baseParameterCount;
        state["UseAdapters"] = _useAdapters;
        state["AdapterDimension"] = _adapterDimension;
        return state;
    }
}
