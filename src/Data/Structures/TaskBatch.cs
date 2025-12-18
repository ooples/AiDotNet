using AiDotNet.Interfaces;

namespace AiDotNet.Data.Structures;

/// <summary>
/// Represents a batch of meta-learning tasks for efficient meta-training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// TaskBatch groups multiple IMetaLearningTasks together for efficient meta-training.
/// Meta-learning algorithms typically process tasks in batches to compute
/// more stable meta-gradients and improve convergence through averaging.
/// </para>
/// <para><b>For Beginners:</b> A TaskBatch is like processing multiple few-shot
/// learning problems simultaneously:</para>
///
/// Example with 4 tasks in a batch:
/// - Task 1: 5-way classification of animals
/// - Task 2: 5-way classification of vehicles
/// - Task 3: 5-way classification of furniture
/// - Task 4: 5-way classification of food
///
/// The meta-learner learns from all 4 tasks simultaneously, discovering
/// patterns that help it adapt to new tasks faster. This batch processing
/// provides more stable gradients and better convergence.
/// </para>
/// </remarks>
public class TaskBatch<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the collection of meta-learning tasks in this batch.
    /// </summary>
    /// <value>
    /// List of meta-learning tasks to be processed together.
    /// </value>
    public List<IMetaLearningTask<T, TInput, TOutput>> Tasks { get; }

    /// <summary>
    /// Gets the batch size (number of tasks).
    /// </summary>
    /// <value>
    /// Number of tasks in the batch.
    /// </value>
    public int BatchSize => Tasks.Count;

    /// <summary>
    /// Gets the number of ways (classes) in the tasks.
    /// </summary>
    /// <value>
    /// Number of classes per task. Assumes all tasks have same number of classes.
    /// </value>
    public int NumberOfWays { get; private set; }

    /// <summary>
    /// Gets the number of shots per class.
    /// </summary>
    /// <value>
    /// Number of support examples per class. Assumes uniform across tasks.
    /// </value>
    public int NumberOfShots { get; private set; }

    /// <summary>
    /// Gets the number of query examples per class.
    /// </summary>
    /// <value>
    /// Number of query examples per class. Assumes uniform across tasks.
    /// </value>
    public int NumberOfQueries { get; private set; }

    /// <summary>
    /// Gets optional metadata about the task batch.
    /// </summary>
    /// <value>
    /// Dictionary containing batch-level information.
    /// </value>
    public Dictionary<string, object>? BatchMetadata { get; set; }

    /// <summary>
    /// Initializes a new instance of the TaskBatch class.
    /// </summary>
    public TaskBatch()
    {
        Tasks = new List<IMetaLearningTask<T, TInput, TOutput>>();
    }

    /// <summary>
    /// Initializes a new instance of the TaskBatch class with specified tasks.
    /// </summary>
    /// <param name="tasks">Collection of meta-learning tasks to include in the batch.</param>
    /// <exception cref="ArgumentNullException">Thrown when tasks is null.</exception>
    public TaskBatch(IEnumerable<IMetaLearningTask<T, TInput, TOutput>> tasks)
    {
        if (tasks == null)
            throw new ArgumentNullException(nameof(tasks));

        Tasks = new List<IMetaLearningTask<T, TInput, TOutput>>(tasks);
        AnalyzeTaskStatistics();
    }

    /// <summary>
    /// Adds a task to the batch.
    /// </summary>
    /// <param name="task">The task to add.</param>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public void AddTask(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
            throw new ArgumentNullException(nameof(task));

        Tasks.Add(task);
        AnalyzeTaskStatistics();
    }

    /// <summary>
    /// Adds multiple tasks to the batch.
    /// </summary>
    /// <param name="tasks">Collection of tasks to add.</param>
    /// <exception cref="ArgumentNullException">Thrown when tasks is null.</exception>
    public void AddTasks(IEnumerable<IMetaLearningTask<T, TInput, TOutput>> tasks)
    {
        if (tasks == null)
            throw new ArgumentNullException(nameof(tasks));

        Tasks.AddRange(tasks);
        AnalyzeTaskStatistics();
    }

    /// <summary>
    /// Gets a task by index.
    /// </summary>
    /// <param name="index">Zero-based index of the task.</param>
    /// <returns>The task at the specified index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
    public IMetaLearningTask<T, TInput, TOutput> GetTask(int index)
    {
        if (index < 0 || index >= Tasks.Count)
            throw new ArgumentOutOfRangeException(nameof(index), $"Task index out of range: {index}");

        return Tasks[index];
    }

    /// <summary>
    /// Removes all tasks from the batch.
    /// </summary>
    public void Clear()
    {
        Tasks.Clear();
        NumberOfWays = 0;
        NumberOfShots = 0;
        NumberOfQueries = 0;
    }

    /// <summary>
    /// Gets all support inputs from all tasks.
    /// </summary>
    /// <returns>Array of support inputs from each task.</returns>
    public TInput[] GetSupportInputs()
    {
        return Tasks.Select(t => t.SupportInput).ToArray();
    }

    /// <summary>
    /// Gets all support outputs from all tasks.
    /// </summary>
    /// <returns>Array of support outputs from each task.</returns>
    public TOutput[] GetSupportOutputs()
    {
        return Tasks.Select(t => t.SupportOutput).ToArray();
    }

    /// <summary>
    /// Gets all query inputs from all tasks.
    /// </summary>
    /// <returns>Array of query inputs from each task.</returns>
    public TInput[] GetQueryInputs()
    {
        return Tasks.Select(t => t.QueryInput).ToArray();
    }

    /// <summary>
    /// Gets all query outputs from all tasks.
    /// </summary>
    /// <returns>Array of query outputs from each task.</returns>
    public TOutput[] GetQueryOutputs()
    {
        return Tasks.Select(t => t.QueryOutput).ToArray();
    }

    /// <summary>
    /// Shuffles the order of tasks in the batch.
    /// </summary>
    /// <param name="random">Random number generator for shuffling.</param>
    /// <exception cref="ArgumentNullException">Thrown when random is null.</exception>
    public void Shuffle(Random random)
    {
        if (random == null)
            throw new ArgumentNullException(nameof(random));

        for (int i = Tasks.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (Tasks[i], Tasks[j]) = (Tasks[j], Tasks[i]);
        }
    }

    /// <summary>
    /// Splits the batch into smaller sub-batches.
    /// </summary>
    /// <param name="subBatchSize">Size of each sub-batch.</param>
    /// <returns>List of sub-batches.</returns>
    /// <exception cref="ArgumentException">Thrown when subBatchSize is less than 1.</exception>
    public List<TaskBatch<T, TInput, TOutput>> Split(int subBatchSize)
    {
        if (subBatchSize < 1)
            throw new ArgumentException("Sub-batch size must be at least 1", nameof(subBatchSize));

        var subBatches = new List<TaskBatch<T, TInput, TOutput>>();

        for (int i = 0; i < Tasks.Count; i += subBatchSize)
        {
            var subBatchTasks = Tasks.Skip(i).Take(subBatchSize);
            subBatches.Add(new TaskBatch<T, TInput, TOutput>(subBatchTasks));
        }

        return subBatches;
    }

    /// <summary>
    /// Analyzes task statistics to extract common properties.
    /// </summary>
    private void AnalyzeTaskStatistics()
    {
        if (Tasks.Count == 0)
        {
            NumberOfWays = 0;
            NumberOfShots = 0;
            NumberOfQueries = 0;
            return;
        }

        // Extract metadata from first task as reference
        var firstTask = Tasks[0];
        if (firstTask.Metadata != null)
        {
            NumberOfWays = firstTask.Metadata.TryGetValue("num_ways", out var ways) ? Convert.ToInt32(ways) : 5;
            NumberOfShots = firstTask.Metadata.TryGetValue("num_shots", out var shots) ? Convert.ToInt32(shots) : 1;
            NumberOfQueries = firstTask.Metadata.TryGetValue("num_queries", out var queries) ? Convert.ToInt32(queries) : 15;
        }
    }

    /// <summary>
    /// Creates a string representation of the task batch.
    /// </summary>
    /// <returns>
    /// String containing batch size, number of ways, shots, and queries.
    /// </returns>
    public override string ToString()
    {
        return $"TaskBatch(Size: {BatchSize}, Ways: {NumberOfWays}, Shots: {NumberOfShots}, Queries: {NumberOfQueries})";
    }
}