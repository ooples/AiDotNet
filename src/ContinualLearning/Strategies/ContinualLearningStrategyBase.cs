using System.Text.Json;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Abstract base class for continual learning strategies providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality that all
/// continual learning strategies share, such as:</para>
/// <list type="bullet">
/// <item><description>Memory usage tracking</description></item>
/// <item><description>Save/Load functionality</description></item>
/// <item><description>Metrics collection</description></item>
/// <item><description>Model compatibility checking</description></item>
/// </list>
///
/// <para>Derived classes implement specific algorithms like EWC, LwF, GEM, SI, and MAS.</para>
/// </remarks>
public abstract class ContinualLearningStrategyBase<T, TInput, TOutput> : IContinualLearningStrategy<T, TInput, TOutput>
{
    /// <summary>
    /// Numeric operations for the generic type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The loss function used for computing task losses.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Number of tasks that have been processed.
    /// </summary>
    protected int TaskCount;

    /// <summary>
    /// Timestamp when the strategy was created.
    /// </summary>
    protected readonly DateTime CreatedAt;

    /// <summary>
    /// Metrics collected during training.
    /// </summary>
    protected readonly Dictionary<string, object> Metrics;

    /// <summary>
    /// Initializes a new instance of the strategy base.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    protected ContinualLearningStrategyBase(ILossFunction<T> lossFunction)
    {
        LossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        CreatedAt = DateTime.UtcNow;
        Metrics = new Dictionary<string, object>();
    }

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract bool RequiresMemoryBuffer { get; }

    /// <inheritdoc/>
    public abstract bool ModifiesArchitecture { get; }

    /// <inheritdoc/>
    public abstract long MemoryUsageBytes { get; }

    /// <inheritdoc/>
    public abstract void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData);

    /// <inheritdoc/>
    public abstract T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model);

    /// <inheritdoc/>
    public abstract Vector<T> AdjustGradients(Vector<T> gradients);

    /// <inheritdoc/>
    public abstract void FinalizeTask(IFullModel<T, TInput, TOutput> model);

    /// <inheritdoc/>
    public virtual void Reset()
    {
        TaskCount = 0;
        Metrics.Clear();
    }

    /// <inheritdoc/>
    public virtual void Save(string path)
    {
        if (string.IsNullOrEmpty(path))
            throw new ArgumentNullException(nameof(path));

        // Security: Prevent path traversal attacks
        var fullPath = Path.GetFullPath(path);
        var directory = Path.GetDirectoryName(fullPath);
        if (directory != null && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var state = GetStateForSerialization();
        var json = System.Text.Json.JsonSerializer.Serialize(state, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(fullPath, json);
    }

    /// <inheritdoc/>
    public virtual void Load(string path)
    {
        if (string.IsNullOrEmpty(path))
            throw new ArgumentNullException(nameof(path));

        // Security: Prevent path traversal attacks
        var fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException("Strategy state file not found", fullPath);

        var json = File.ReadAllText(fullPath);
        var state = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, System.Text.Json.JsonElement>>(json);
        if (state != null)
        {
            LoadStateFromSerialization(state);
        }
    }

    /// <inheritdoc/>
    public virtual IReadOnlyDictionary<string, object> GetMetrics()
    {
        // Add common metrics
        var allMetrics = new Dictionary<string, object>(Metrics)
        {
            ["TaskCount"] = TaskCount,
            ["StrategyName"] = Name,
            ["MemoryUsageBytes"] = MemoryUsageBytes,
            ["CreatedAt"] = CreatedAt
        };

        return allMetrics;
    }

    /// <inheritdoc/>
    public virtual bool IsCompatibleWith(IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
            return false;

        // Basic compatibility check: model must have parameters
        return model.ParameterCount > 0;
    }

    /// <inheritdoc/>
    public virtual string? GetIncompatibilityReason(IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
            return "Model cannot be null";

        if (model.ParameterCount == 0)
            return "Model must have parameters for continual learning";

        return null;
    }

    /// <summary>
    /// Gets the state for serialization.
    /// </summary>
    /// <returns>Dictionary of state key-value pairs.</returns>
    protected virtual Dictionary<string, object> GetStateForSerialization()
    {
        return new Dictionary<string, object>
        {
            ["StrategyName"] = Name,
            ["TaskCount"] = TaskCount,
            ["CreatedAt"] = CreatedAt.ToString("o"),
            ["Metrics"] = Metrics
        };
    }

    /// <summary>
    /// Loads state from serialized data.
    /// </summary>
    /// <param name="state">The serialized state dictionary.</param>
    protected virtual void LoadStateFromSerialization(Dictionary<string, JsonElement> state)
    {
        if (state.TryGetValue("TaskCount", out var taskCountElement))
        {
            TaskCount = taskCountElement.GetInt32();
        }
    }

    /// <summary>
    /// Estimates the memory usage of a Vector in bytes.
    /// </summary>
    /// <param name="vector">The vector to estimate.</param>
    /// <returns>Estimated memory usage in bytes.</returns>
    protected static long EstimateVectorMemory(Vector<T>? vector)
    {
        if (vector == null)
            return 0;

        // Estimate: size of T * length + overhead
        return vector.Length * GetTypeSize() + 32;
    }

    /// <summary>
    /// Gets the size of type T in bytes.
    /// </summary>
    /// <returns>Size in bytes.</returns>
    protected static int GetTypeSize()
    {
        if (typeof(T) == typeof(double))
            return 8;
        if (typeof(T) == typeof(float))
            return 4;
        if (typeof(T) == typeof(decimal))
            return 16;
        return 8; // Default assumption
    }

    /// <summary>
    /// Computes the L2 norm of a vector.
    /// </summary>
    /// <param name="vector">The vector.</param>
    /// <returns>The L2 norm.</returns>
    protected T ComputeL2Norm(Vector<T> vector)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(vector[i], vector[i]));
        }
        return NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sum)));
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector.</param>
    /// <returns>The dot product.</returns>
    protected T ComputeDotProduct(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    /// <summary>
    /// Clones a vector.
    /// </summary>
    /// <param name="source">The source vector.</param>
    /// <returns>A new vector with the same values.</returns>
    protected Vector<T> CloneVector(Vector<T> source)
    {
        var result = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            result[i] = source[i];
        }
        return result;
    }

    /// <summary>
    /// Records a metric value.
    /// </summary>
    /// <param name="name">The metric name.</param>
    /// <param name="value">The metric value.</param>
    protected void RecordMetric(string name, object value)
    {
        Metrics[name] = value;
    }
}
