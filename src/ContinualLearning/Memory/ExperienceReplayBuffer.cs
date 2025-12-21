using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Memory;

/// <summary>
/// A memory buffer for storing examples from previous tasks for experience replay.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Experience replay stores a small number of examples from
/// previous tasks and intermixes them with new task data during training. This helps
/// prevent catastrophic forgetting by reminding the model of what it learned before.</para>
/// </remarks>
public class ExperienceReplayBuffer<T, TInput, TOutput>
{
    private readonly int _maxSize;
    private readonly List<DataPoint<T, TInput, TOutput>> _buffer;
    private readonly Random _random;

    /// <summary>
    /// Gets the current number of stored examples.
    /// </summary>
    public int Count => _buffer.Count;

    /// <summary>
    /// Gets whether the buffer is at capacity.
    /// </summary>
    public bool IsFull => _buffer.Count >= _maxSize;

    /// <summary>
    /// Initializes a new experience replay buffer.
    /// </summary>
    /// <param name="maxSize">Maximum number of examples to store.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public ExperienceReplayBuffer(int maxSize, int? seed = null)
    {
        if (maxSize < 0)
            throw new ArgumentException("Max size must be non-negative", nameof(maxSize));

        _maxSize = maxSize;
        _buffer = new List<DataPoint<T, TInput, TOutput>>(maxSize);
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Adds examples from a task to the buffer using reservoir sampling.
    /// </summary>
    /// <param name="taskData">The task data to sample from.</param>
    /// <param name="samplesPerTask">Number of samples to store from this task.</param>
    public void AddTaskExamples(IDataset<T, TInput, TOutput> taskData, int samplesPerTask)
    {
        if (taskData == null)
            throw new ArgumentNullException(nameof(taskData));
        if (samplesPerTask < 0)
            throw new ArgumentException("Samples per task must be non-negative", nameof(samplesPerTask));

        var dataPoints = taskData.GetAllDataPoints().ToList();
        var sampled = ReservoirSample(dataPoints, samplesPerTask);

        foreach (var point in sampled)
        {
            if (_buffer.Count < _maxSize)
            {
                _buffer.Add(point);
            }
            else
            {
                // Replace random element if buffer is full
                int replaceIndex = _random.Next(_buffer.Count);
                _buffer[replaceIndex] = point;
            }
        }
    }

    /// <summary>
    /// Samples a batch of examples from the buffer.
    /// </summary>
    /// <param name="batchSize">Number of examples to sample.</param>
    /// <returns>A list of sampled data points.</returns>
    public List<DataPoint<T, TInput, TOutput>> SampleBatch(int batchSize)
    {
        if (batchSize < 0)
            throw new ArgumentException("Batch size must be non-negative", nameof(batchSize));
        if (_buffer.Count == 0)
            return new List<DataPoint<T, TInput, TOutput>>();

        int actualBatchSize = Math.Min(batchSize, _buffer.Count);
        var indices = Enumerable.Range(0, _buffer.Count)
            .OrderBy(_ => _random.Next())
            .Take(actualBatchSize);

        return indices.Select(i => _buffer[i]).ToList();
    }

    /// <summary>
    /// Gets all stored examples.
    /// </summary>
    public IReadOnlyList<DataPoint<T, TInput, TOutput>> GetAll()
    {
        return _buffer.AsReadOnly();
    }

    /// <summary>
    /// Clears all stored examples.
    /// </summary>
    public void Clear()
    {
        _buffer.Clear();
    }

    /// <summary>
    /// Performs reservoir sampling to select k items from a list.
    /// </summary>
    private List<DataPoint<T, TInput, TOutput>> ReservoirSample(
        List<DataPoint<T, TInput, TOutput>> items,
        int k)
    {
        if (k >= items.Count)
            return new List<DataPoint<T, TInput, TOutput>>(items);

        var reservoir = new List<DataPoint<T, TInput, TOutput>>(k);

        // Fill reservoir with first k items
        for (int i = 0; i < k; i++)
        {
            reservoir.Add(items[i]);
        }

        // Process remaining items
        for (int i = k; i < items.Count; i++)
        {
            int j = _random.Next(i + 1);
            if (j < k)
            {
                reservoir[j] = items[i];
            }
        }

        return reservoir;
    }
}

/// <summary>
/// Represents a single data point with input and output.
/// </summary>
public class DataPoint<T, TInput, TOutput>
{
    public TInput Input { get; }
    public TOutput Output { get; }
    public int TaskId { get; }

    public DataPoint(TInput input, TOutput output, int taskId)
    {
        Input = input;
        Output = output;
        TaskId = taskId;
    }
}

/// <summary>
/// Extension methods for IDataset to convert to DataPoints.
/// </summary>
public static class DatasetExtensions
{
    /// <summary>
    /// Converts a dataset to an enumerable of DataPoint objects.
    /// </summary>
    /// <param name="dataset">The dataset to convert.</param>
    /// <param name="taskId">Optional task ID to assign to all data points (default: 0).</param>
    /// <returns>Enumerable of DataPoint objects containing inputs, outputs, and task IDs.</returns>
    public static IEnumerable<DataPoint<T, TInput, TOutput>> GetAllDataPoints<T, TInput, TOutput>(
        this IDataset<T, TInput, TOutput> dataset,
        int taskId = 0)
    {
        if (dataset == null)
            throw new ArgumentNullException(nameof(dataset));

        for (int i = 0; i < dataset.Count; i++)
        {
            var input = dataset.GetInput(i);
            var output = dataset.GetOutput(i);
            yield return new DataPoint<T, TInput, TOutput>(input, output, taskId);
        }
    }
}
