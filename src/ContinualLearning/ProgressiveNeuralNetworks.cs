using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Progressive Neural Networks for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Progressive Neural Networks prevent forgetting by freezing
/// previously learned networks and adding new "columns" (networks) for each new task.
/// The new columns can receive input from all previous columns through lateral connections,
/// enabling knowledge transfer without forgetting.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Train a neural network column for Task 1.</description></item>
/// <item><description>Freeze the Task 1 column completely.</description></item>
/// <item><description>Add a new column for Task 2 with lateral connections from Task 1's hidden layers.</description></item>
/// <item><description>Train only the new column (Task 1 column remains frozen).</description></item>
/// <item><description>Repeat for each new task, adding lateral connections from all previous columns.</description></item>
/// </list>
///
/// <para><b>Architecture:</b></para>
/// <code>
///    Task 1      Task 2      Task 3
///    Column      Column      Column
///      │           │           │
///    [L1]──────>[L1]──────>[L1]    (Lateral connections)
///      │           │           │
///    [L2]──────>[L2]──────>[L2]
///      │           │           │
///    [Out]       [Out]       [Out]
/// </code>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Zero forgetting - previous columns are completely frozen.</description></item>
/// <item><description>Positive transfer - new tasks can leverage previous knowledge.</description></item>
/// <item><description>Clear task separation - each task has its own output.</description></item>
/// </list>
///
/// <para><b>Disadvantages:</b></para>
/// <list type="bullet">
/// <item><description>Linear growth in parameters with number of tasks.</description></item>
/// <item><description>Memory usage increases with each task.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Rusu, A.A. et al. "Progressive Neural Networks" (2016). arXiv.</para>
/// </remarks>
public class ProgressiveNeuralNetworks<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<ColumnInfo> _columns;
    private readonly bool _useLateralConnections;
    private double _lambda;
    private int _currentTaskId;

    /// <summary>
    /// Information about a column in the progressive network.
    /// </summary>
    private class ColumnInfo
    {
        public int TaskId { get; set; }
        public Vector<T>? FrozenParameters { get; set; }
        public int ParameterCount { get; set; }
        public bool IsFrozen { get; set; }
    }

    /// <summary>
    /// Initializes a new instance of the ProgressiveNeuralNetworks class.
    /// </summary>
    /// <param name="useLateralConnections">Whether to use lateral connections between columns (default: true).</param>
    /// <param name="lambda">Regularization strength for lateral connections (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Lateral connections allow new tasks to use features learned by previous tasks.</description></item>
    /// <item><description>Without lateral connections, this becomes simple multi-head training.</description></item>
    /// </list>
    /// </remarks>
    public ProgressiveNeuralNetworks(bool useLateralConnections = true, double lambda = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _columns = [];
        _useLateralConnections = useLateralConnections;
        _lambda = lambda;
        _currentTaskId = -1;
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets the number of columns (tasks) in the progressive network.
    /// </summary>
    public int ColumnCount => _columns.Count;

    /// <summary>
    /// Gets whether lateral connections are enabled.
    /// </summary>
    public bool UseLateralConnections => _useLateralConnections;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        _currentTaskId = taskId;

        // Freeze all existing columns
        foreach (var column in _columns)
        {
            column.IsFrozen = true;
        }

        // Add a new column for this task (will be populated in AfterTask)
        _columns.Add(new ColumnInfo
        {
            TaskId = taskId,
            FrozenParameters = null,
            ParameterCount = network.ParameterCount,
            IsFrozen = false
        });
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        // Find the column for this task and store its parameters
        var column = _columns.FirstOrDefault(c => c.TaskId == taskId);
        if (column != null)
        {
            column.FrozenParameters = network.GetParameters().Clone();
            column.IsFrozen = true;
        }
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        // Progressive networks use parameter freezing, not loss-based regularization
        // However, we can add a small regularization for lateral connection weights
        return _numOps.Zero;
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = gradients ?? throw new ArgumentNullException(nameof(gradients));

        // In a true progressive network implementation, the frozen columns
        // would be separate networks. Here we simulate by zeroing gradients
        // for parameters belonging to frozen columns.

        // For simplicity, we assume the network manages column separation internally
        // and we just ensure frozen columns don't update

        // Get the current column's start index (cumulative parameters)
        var frozenParamCount = GetFrozenParameterCount();

        // Zero gradients for frozen parameters
        for (int i = 0; i < Math.Min(frozenParamCount, gradients.Length); i++)
        {
            gradients[i] = _numOps.Zero;
        }

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _columns.Clear();
        _currentTaskId = -1;
    }

    /// <summary>
    /// Gets the total number of frozen parameters across all completed tasks.
    /// </summary>
    private int GetFrozenParameterCount()
    {
        return _columns
            .Where(c => c.IsFrozen && c.TaskId != _currentTaskId)
            .Sum(c => c.ParameterCount);
    }

    /// <summary>
    /// Gets parameters for a specific task's column.
    /// </summary>
    /// <param name="taskId">The task ID.</param>
    /// <returns>The frozen parameters for that task, or null if not found.</returns>
    public Vector<T>? GetColumnParameters(int taskId)
    {
        var column = _columns.FirstOrDefault(c => c.TaskId == taskId);
        return column?.FrozenParameters?.Clone();
    }

    /// <summary>
    /// Computes the lateral connection activation from previous columns.
    /// </summary>
    /// <param name="previousActivations">Activations from previous columns at a given layer.</param>
    /// <param name="lateralWeights">Lateral connection weights.</param>
    /// <returns>Combined lateral input for the current column.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lateral connections take the hidden activations from
    /// all previous columns and combine them (weighted sum) as additional input to
    /// the current column's layers. This allows knowledge transfer.</para>
    /// </remarks>
    public Tensor<T> ComputeLateralInput(
        List<Tensor<T>> previousActivations,
        List<Tensor<T>> lateralWeights)
    {
        if (!_useLateralConnections || previousActivations.Count == 0)
        {
            return new Tensor<T>([0], new Vector<T>(0));
        }

        // Combine lateral inputs: Σ W_lateral_i * h_i
        var firstActivation = previousActivations[0];
        var outputSize = firstActivation.Shape.Length > 1 ? firstActivation.Shape[1] : firstActivation.Length;
        var result = new Vector<T>(outputSize);

        for (int c = 0; c < previousActivations.Count && c < lateralWeights.Count; c++)
        {
            var activation = previousActivations[c];
            var weights = lateralWeights[c];

            // Simple weighted sum (in practice, this would be a matrix multiply)
            for (int i = 0; i < Math.Min(outputSize, activation.Length); i++)
            {
                var weighted = _numOps.Multiply(activation[i], weights[i % weights.Length]);
                result[i] = _numOps.Add(result[i], weighted);
            }
        }

        return new Tensor<T>([1, outputSize], result);
    }

    /// <summary>
    /// Gets statistics about the progressive network structure.
    /// </summary>
    /// <returns>Dictionary with network statistics.</returns>
    public Dictionary<string, object> GetNetworkStats()
    {
        return new Dictionary<string, object>
        {
            ["ColumnCount"] = _columns.Count,
            ["TotalParameters"] = _columns.Sum(c => c.ParameterCount),
            ["FrozenParameters"] = _columns.Where(c => c.IsFrozen).Sum(c => c.ParameterCount),
            ["ActiveParameters"] = _columns.Where(c => !c.IsFrozen).Sum(c => c.ParameterCount),
            ["UseLateralConnections"] = _useLateralConnections,
            ["TaskIds"] = _columns.Select(c => c.TaskId).ToList()
        };
    }

    /// <summary>
    /// Estimates the total memory usage of the progressive network.
    /// </summary>
    /// <returns>Estimated memory in bytes (assuming 4 bytes per parameter for float).</returns>
    public long EstimateMemoryUsage()
    {
        // Each parameter typically takes 4 bytes (float) or 8 bytes (double)
        var bytesPerParam = typeof(T) == typeof(double) ? 8 : 4;
        var totalParams = _columns.Sum(c => c.ParameterCount);
        return totalParams * bytesPerParam;
    }
}
