using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Structures;

/// <summary>
/// A basic meta-learning task implementation with simple support and query sets.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// BasicMetaLearningTask provides a simple implementation for standard few-shot learning
/// scenarios. It's suitable for most meta-learning algorithms like MAML, Reptile,
/// ProtoNets, and Matching Networks where tasks are independent episodes.
/// </para>
/// <para>
/// <b>Common Use Cases:</b>
/// - N-way K-shot classification
/// - Few-shot regression
/// - Standard meta-learning benchmarks (Omniglot, Mini-ImageNet)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a 5-way 3-shot classification task
/// var task = new BasicMetaLearningTask&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;("animal_classification");
///
/// // Set up support set (15 examples: 5 classes × 3 shots)
/// task.SupportInput = supportImages;  // Shape: [15, 784]
/// task.SupportOutput = supportLabels; // Shape: [15, 5] (one-hot)
///
/// // Set up query set (50 examples for evaluation)
/// task.QueryInput = queryImages;     // Shape: [50, 784]
/// task.QueryOutput = queryLabels;    // Shape: [50, 5] (one-hot)
///
/// // Add metadata
/// task.SetMetadata("num_ways", 5);
/// task.SetMetadata("num_shots", 3);
/// task.SetMetadata("num_queries", 50);
/// </code>
/// </example>
public class BasicMetaLearningTask<T, TInput, TOutput> : MetaLearningTaskBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the BasicMetaLearningTask class.
    /// </summary>
    public BasicMetaLearningTask() : base()
    {
        InitializeWithDefaults();
    }

    /// <summary>
    /// Initializes a new instance of the BasicMetaLearningTask class with a name.
    /// </summary>
    /// <param name="name">The name of the task.</param>
    public BasicMetaLearningTask(string name) : base(name)
    {
        InitializeWithDefaults();
    }

    /// <summary>
    /// Initializes a new instance of the BasicMetaLearningTask class with data.
    /// </summary>
    /// <param name="supportInput">Support set inputs.</param>
    /// <param name="supportOutput">Support set outputs.</param>
    /// <param name="queryInput">Query set inputs.</param>
    /// <param name="queryOutput">Query set outputs.</param>
    /// <param name="name">Optional task name.</param>
    public BasicMetaLearningTask(
        TInput supportInput,
        TOutput supportOutput,
        TInput queryInput,
        TOutput queryOutput,
        string? name = null) : base(name)
    {
        SupportInput = supportInput;
        SupportOutput = supportOutput;
        QueryInput = queryInput;
        QueryOutput = queryOutput;
    }

    /// <summary>
    /// Initializes the task with default values based on common configurations.
    /// </summary>
    private void InitializeWithDefaults()
    {
        // Try to create sensible defaults if ModelHelper is available
        try
        {
            var defaultData = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
            if (defaultData.X != null && defaultData.Y != null && defaultData.Predictions != null)
            {
                SupportInput = defaultData.X;
                SupportOutput = defaultData.Y;
                QueryInput = defaultData.X;
                QueryOutput = defaultData.Y;
            }
        }
        catch
        {
            // If ModelHelper fails, we'll leave properties as default values
            // Users must set them explicitly before using the task
        }
    }

    /// <summary>
    /// Gets the number of classes (ways) in this task.
    /// </summary>
    /// <returns>Number of classes if available in metadata, otherwise -1.</returns>
    public int GetNumberOfWays()
    {
        if (TryGetMetadata<int>("num_ways", out var ways))
            return ways;
        return -1;
    }

    /// <summary>
    /// Gets the number of examples per class (shots) in the support set.
    /// </summary>
    /// <returns>Number of shots if available in metadata, otherwise -1.</returns>
    public int GetNumberOfShots()
    {
        if (TryGetMetadata<int>("num_shots", out var shots))
            return shots;
        return -1;
    }

    /// <summary>
    /// Gets the number of query examples.
    /// </summary>
    /// <returns>Number of queries if available in metadata, otherwise -1.</returns>
    public int GetNumberOfQueries()
    {
        if (TryGetMetadata<int>("num_queries", out var queries))
            return queries;
        return -1;
    }

    /// <summary>
    /// Creates a string representation with task configuration.
    /// </summary>
    /// <returns>String containing task name and configuration.</returns>
    public override string ToString()
    {
        var name = string.IsNullOrEmpty(Name) ? "BasicMetaLearningTask" : Name;
        var ways = GetNumberOfWays();
        var shots = GetNumberOfShots();
        var queries = GetNumberOfQueries();

        if (ways > 0 && shots > 0)
        {
            return $"{name} ({ways}-way {shots}-shot" +
                   (queries > 0 ? $", {queries} queries)" : ")");
        }
        return base.ToString();
    }
}