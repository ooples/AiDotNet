namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Concrete implementation of a meta-learning task.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class holds all the data for a single meta-learning task,
/// including the support set (training examples) and query set (test examples).
/// </para>
/// </remarks>
public class Task<T, TInput, TOutput> : ITask<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the Task class.
    /// </summary>
    /// <param name="supportInput">The support set input data.</param>
    /// <param name="supportOutput">The support set output labels.</param>
    /// <param name="queryInput">The query set input data.</param>
    /// <param name="queryOutput">The query set output labels.</param>
    /// <param name="numWays">The number of classes in this task.</param>
    /// <param name="numShots">The number of examples per class in the support set.</param>
    /// <param name="numQueryPerClass">The number of query examples per class.</param>
    /// <param name="taskId">The task identifier.</param>
    public Task(
        TInput supportInput,
        TOutput supportOutput,
        TInput queryInput,
        TOutput queryOutput,
        int numWays,
        int numShots,
        int numQueryPerClass,
        string? taskId = null)
    {
        SupportInput = supportInput ?? throw new ArgumentNullException(nameof(supportInput));
        SupportOutput = supportOutput ?? throw new ArgumentNullException(nameof(supportOutput));
        QueryInput = queryInput ?? throw new ArgumentNullException(nameof(queryInput));
        QueryOutput = queryOutput ?? throw new ArgumentNullException(nameof(queryOutput));

        if (numWays <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numWays), "NumWays must be positive.");
        }

        if (numShots <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numShots), "NumShots must be positive.");
        }

        if (numQueryPerClass <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numQueryPerClass), "NumQueryPerClass must be positive.");
        }

        NumWays = numWays;
        NumShots = numShots;
        NumQueryPerClass = numQueryPerClass;
        TaskId = taskId ?? Guid.NewGuid().ToString();
    }

    /// <inheritdoc/>
    public TInput SupportInput { get; }

    /// <inheritdoc/>
    public TOutput SupportOutput { get; }

    /// <inheritdoc/>
    public TInput QueryInput { get; }

    /// <inheritdoc/>
    public TOutput QueryOutput { get; }

    /// <inheritdoc/>
    public int NumWays { get; }

    /// <inheritdoc/>
    public int NumShots { get; }

    /// <inheritdoc/>
    public int NumQueryPerClass { get; }

    /// <inheritdoc/>
    public string TaskId { get; }

    /// <summary>
    /// Gets the total number of support examples (NumWays * NumShots).
    /// </summary>
    public int TotalSupportExamples => NumWays * NumShots;

    /// <summary>
    /// Gets the total number of query examples (NumWays * NumQueryPerClass).
    /// </summary>
    public int TotalQueryExamples => NumWays * NumQueryPerClass;
}
