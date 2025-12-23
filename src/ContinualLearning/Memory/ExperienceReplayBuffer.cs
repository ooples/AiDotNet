using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Memory;

/// <summary>
/// Strategy for sampling during replay (training time).
/// </summary>
public enum ReplaySamplingStrategy
{
    /// <summary>
    /// Uniform random sampling from the buffer.
    /// </summary>
    Uniform,

    /// <summary>
    /// Task-balanced sampling ensures equal representation of each task.
    /// Prevents bias toward tasks with more stored examples.
    /// </summary>
    TaskBalanced,

    /// <summary>
    /// Priority-based sampling using importance weights.
    /// Examples that caused high loss are sampled more frequently.
    /// </summary>
    PriorityBased,

    /// <summary>
    /// Recency-weighted sampling favors more recent examples.
    /// Based on the hypothesis that recent examples are more relevant.
    /// </summary>
    RecencyWeighted
}

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
///
/// <para><b>Why It Works:</b> When a neural network learns a new task, it adjusts its weights
/// to minimize error on that task. Without replay, these adjustments can increase error on
/// previous tasks. By mixing old examples with new training data, the network must maintain
/// performance on both old and new tasks.</para>
///
/// <para><b>Memory Management:</b> Since storing all examples is impractical, we use smart
/// sampling strategies to select the most representative examples. Different strategies
/// work better for different scenarios:
/// <list type="bullet">
/// <item><description><b>Reservoir:</b> Fair representation, good for general use</description></item>
/// <item><description><b>ClassBalanced:</b> Equal class representation, best for imbalanced data</description></item>
/// <item><description><b>Herding:</b> Class-mean exemplars, from iCaRL paper</description></item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item><description>Chaudhry et al. "Continual Learning with Tiny Episodic Memories" (2019)</description></item>
/// <item><description>Rebuffi et al. "iCaRL: Incremental Classifier and Representation Learning" (2017)</description></item>
/// <item><description>Rolnick et al. "Experience Replay for Continual Learning" (2019)</description></item>
/// </list>
/// </para>
/// </remarks>
public class ExperienceReplayBuffer<T, TInput, TOutput>
{
    private readonly int _maxSize;
    private readonly List<DataPoint<T, TInput, TOutput>> _buffer;
    private readonly Random _random;
    private readonly MemorySamplingStrategy _addStrategy;
    private readonly ReplaySamplingStrategy _replayStrategy;
    private readonly Dictionary<int, List<int>> _taskIndices;
    private readonly Dictionary<int, double> _priorities;
    private int _totalSamplesProcessed;
    private int _totalReplaySamples;
    private long _estimatedMemoryBytes;

    /// <summary>
    /// Gets the current number of stored examples.
    /// </summary>
    public int Count => _buffer.Count;

    /// <summary>
    /// Gets the maximum capacity of the buffer.
    /// </summary>
    public int MaxSize => _maxSize;

    /// <summary>
    /// Gets whether the buffer is at capacity.
    /// </summary>
    public bool IsFull => _buffer.Count >= _maxSize;

    /// <summary>
    /// Gets the number of distinct tasks represented in the buffer.
    /// </summary>
    public int TaskCount => _taskIndices.Count;

    /// <summary>
    /// Gets the total number of samples processed (added) since creation.
    /// </summary>
    public int TotalSamplesProcessed => _totalSamplesProcessed;

    /// <summary>
    /// Gets the total number of samples returned via replay.
    /// </summary>
    public int TotalReplaySamples => _totalReplaySamples;

    /// <summary>
    /// Gets the estimated memory usage in bytes.
    /// </summary>
    public long EstimatedMemoryBytes => _estimatedMemoryBytes;

    /// <summary>
    /// Gets the sampling strategy used when adding examples.
    /// </summary>
    public MemorySamplingStrategy AddStrategy => _addStrategy;

    /// <summary>
    /// Gets the sampling strategy used during replay.
    /// </summary>
    public ReplaySamplingStrategy ReplayStrategy => _replayStrategy;

    /// <summary>
    /// Initializes a new experience replay buffer.
    /// </summary>
    /// <param name="maxSize">Maximum number of examples to store.</param>
    /// <param name="addStrategy">Strategy for selecting which examples to add.</param>
    /// <param name="replayStrategy">Strategy for sampling during replay.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public ExperienceReplayBuffer(
        int maxSize,
        MemorySamplingStrategy addStrategy = MemorySamplingStrategy.Reservoir,
        ReplaySamplingStrategy replayStrategy = ReplaySamplingStrategy.TaskBalanced,
        int? seed = null)
    {
        if (maxSize < 0)
            throw new ArgumentException("Max size must be non-negative", nameof(maxSize));

        _maxSize = maxSize;
        _buffer = new List<DataPoint<T, TInput, TOutput>>(maxSize);
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _addStrategy = addStrategy;
        _replayStrategy = replayStrategy;
        _taskIndices = new Dictionary<int, List<int>>();
        _priorities = new Dictionary<int, double>();
        _totalSamplesProcessed = 0;
        _totalReplaySamples = 0;
        _estimatedMemoryBytes = 0;
    }

    /// <summary>
    /// Adds examples from a task to the buffer using the configured sampling strategy.
    /// </summary>
    /// <param name="taskData">The task data to sample from.</param>
    /// <param name="taskId">The task identifier.</param>
    /// <param name="samplesPerTask">Number of samples to store from this task. If null, distributes evenly.</param>
    public void AddTaskExamples(IDataset<T, TInput, TOutput> taskData, int taskId, int? samplesPerTask = null)
    {
        if (taskData == null)
            throw new ArgumentNullException(nameof(taskData));

        int targetSamples = samplesPerTask ?? CalculateTargetSamplesPerTask();

        var dataPoints = taskData.GetAllDataPoints(taskId).ToList();
        _totalSamplesProcessed += dataPoints.Count;

        var sampled = _addStrategy switch
        {
            MemorySamplingStrategy.Reservoir => ReservoirSample(dataPoints, targetSamples),
            MemorySamplingStrategy.Random => RandomSample(dataPoints, targetSamples),
            MemorySamplingStrategy.ClassBalanced => ClassBalancedSample(dataPoints, targetSamples),
            MemorySamplingStrategy.Herding => HerdingSample(dataPoints, targetSamples),
            MemorySamplingStrategy.KCenter => KCenterSample(dataPoints, targetSamples),
            MemorySamplingStrategy.Boundary => BoundarySample(dataPoints, targetSamples),
            _ => ReservoirSample(dataPoints, targetSamples)
        };

        // Make room if needed by removing from overrepresented tasks
        if (IsFull && sampled.Count > 0)
        {
            MakeRoomForTask(taskId, sampled.Count);
        }

        // Add new samples and track indices
        if (!_taskIndices.ContainsKey(taskId))
        {
            _taskIndices[taskId] = new List<int>();
        }

        foreach (var point in sampled)
        {
            if (_buffer.Count < _maxSize)
            {
                int index = _buffer.Count;
                _buffer.Add(point);
                _taskIndices[taskId].Add(index);
                _priorities[index] = 1.0; // Default priority
                _estimatedMemoryBytes += EstimateDataPointSize(point);
            }
        }
    }

    /// <summary>
    /// Samples a batch of examples from the buffer using the configured replay strategy.
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

        var indices = _replayStrategy switch
        {
            ReplaySamplingStrategy.Uniform => UniformReplaySample(actualBatchSize),
            ReplaySamplingStrategy.TaskBalanced => TaskBalancedReplaySample(actualBatchSize),
            ReplaySamplingStrategy.PriorityBased => PriorityBasedReplaySample(actualBatchSize),
            ReplaySamplingStrategy.RecencyWeighted => RecencyWeightedReplaySample(actualBatchSize),
            _ => UniformReplaySample(actualBatchSize)
        };

        _totalReplaySamples += indices.Count;
        return indices.Select(i => _buffer[i]).ToList();
    }

    /// <summary>
    /// Samples examples from a specific task.
    /// </summary>
    /// <param name="taskId">The task to sample from.</param>
    /// <param name="count">Number of examples to sample.</param>
    /// <returns>List of data points from the specified task.</returns>
    public List<DataPoint<T, TInput, TOutput>> SampleFromTask(int taskId, int count)
    {
        if (!_taskIndices.ContainsKey(taskId) || _taskIndices[taskId].Count == 0)
            return new List<DataPoint<T, TInput, TOutput>>();

        var taskList = _taskIndices[taskId];
        int actualCount = Math.Min(count, taskList.Count);

        var indices = Enumerable.Range(0, taskList.Count)
            .OrderBy(_ => _random.Next())
            .Take(actualCount)
            .Select(i => taskList[i]);

        return indices.Select(i => _buffer[i]).ToList();
    }

    /// <summary>
    /// Updates the priority of a sample (for priority-based replay).
    /// </summary>
    /// <param name="index">The buffer index of the sample.</param>
    /// <param name="priority">The new priority value (higher = more likely to be sampled).</param>
    public void UpdatePriority(int index, double priority)
    {
        if (index < 0 || index >= _buffer.Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        _priorities[index] = Math.Max(0.0001, priority); // Prevent zero priority
    }

    /// <summary>
    /// Gets all stored examples.
    /// </summary>
    public IReadOnlyList<DataPoint<T, TInput, TOutput>> GetAll()
    {
        return _buffer.AsReadOnly();
    }

    /// <summary>
    /// Gets all examples for a specific task.
    /// </summary>
    /// <param name="taskId">The task identifier.</param>
    /// <returns>Read-only list of examples for the task.</returns>
    public IReadOnlyList<DataPoint<T, TInput, TOutput>> GetTaskExamples(int taskId)
    {
        if (!_taskIndices.ContainsKey(taskId))
            return Array.Empty<DataPoint<T, TInput, TOutput>>().ToList().AsReadOnly();

        return _taskIndices[taskId].Select(i => _buffer[i]).ToList().AsReadOnly();
    }

    /// <summary>
    /// Gets the count of examples per task.
    /// </summary>
    public IReadOnlyDictionary<int, int> GetTaskCounts()
    {
        return _taskIndices.ToDictionary(kv => kv.Key, kv => kv.Value.Count);
    }

    /// <summary>
    /// Gets statistics about the buffer.
    /// </summary>
    public BufferStatistics GetStatistics()
    {
        return new BufferStatistics
        {
            Count = _buffer.Count,
            MaxSize = _maxSize,
            TaskCount = _taskIndices.Count,
            TaskDistribution = _taskIndices.ToDictionary(kv => kv.Key, kv => kv.Value.Count),
            TotalSamplesProcessed = _totalSamplesProcessed,
            TotalReplaySamples = _totalReplaySamples,
            EstimatedMemoryBytes = _estimatedMemoryBytes,
            AveragePriority = _priorities.Count > 0 ? _priorities.Values.Average() : 0,
            FillRatio = _maxSize > 0 ? (double)_buffer.Count / _maxSize : 0
        };
    }

    /// <summary>
    /// Clears all stored examples.
    /// </summary>
    public void Clear()
    {
        _buffer.Clear();
        _taskIndices.Clear();
        _priorities.Clear();
        _estimatedMemoryBytes = 0;
    }

    /// <summary>
    /// Removes all examples from a specific task.
    /// </summary>
    /// <param name="taskId">The task to remove.</param>
    public void RemoveTask(int taskId)
    {
        if (!_taskIndices.ContainsKey(taskId))
            return;

        // Get indices to remove, sorted descending to avoid index shifting issues
        var indicesToRemove = _taskIndices[taskId].OrderByDescending(i => i).ToList();

        foreach (var index in indicesToRemove)
        {
            _estimatedMemoryBytes -= EstimateDataPointSize(_buffer[index]);
            _buffer.RemoveAt(index);
            _priorities.Remove(index);
        }

        _taskIndices.Remove(taskId);

        // Rebuild index mappings after removal
        RebuildIndices();
    }

    #region Private Sampling Methods

    private List<DataPoint<T, TInput, TOutput>> ReservoirSample(
        List<DataPoint<T, TInput, TOutput>> items, int k)
    {
        if (k >= items.Count)
            return new List<DataPoint<T, TInput, TOutput>>(items);

        var reservoir = new List<DataPoint<T, TInput, TOutput>>(k);

        for (int i = 0; i < k; i++)
        {
            reservoir.Add(items[i]);
        }

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

    private List<DataPoint<T, TInput, TOutput>> RandomSample(
        List<DataPoint<T, TInput, TOutput>> items, int k)
    {
        if (k >= items.Count)
            return new List<DataPoint<T, TInput, TOutput>>(items);

        return items.OrderBy(_ => _random.Next()).Take(k).ToList();
    }

    private List<DataPoint<T, TInput, TOutput>> ClassBalancedSample(
        List<DataPoint<T, TInput, TOutput>> items, int k)
    {
        // Group by output (assuming output represents class)
        var groups = items.GroupBy(p => p.Output?.GetHashCode() ?? 0).ToList();
        int samplesPerClass = Math.Max(1, k / groups.Count);

        var result = new List<DataPoint<T, TInput, TOutput>>();
        foreach (var group in groups)
        {
            var groupItems = group.ToList();
            int take = Math.Min(samplesPerClass, groupItems.Count);
            result.AddRange(groupItems.OrderBy(_ => _random.Next()).Take(take));
        }

        // If we have room for more, add randomly
        while (result.Count < k && result.Count < items.Count)
        {
            var remaining = items.Except(result).ToList();
            if (remaining.Count == 0) break;
            result.Add(remaining[_random.Next(remaining.Count)]);
        }

        return result.Take(k).ToList();
    }

    private List<DataPoint<T, TInput, TOutput>> HerdingSample(
        List<DataPoint<T, TInput, TOutput>> items, int k)
    {
        // iCaRL-style herding: select exemplars closest to class mean
        // Groups samples by class and selects those closest to the class centroid
        if (k >= items.Count)
            return new List<DataPoint<T, TInput, TOutput>>(items);

        // Group by output (class)
        var groups = items.GroupBy(p => p.Output?.GetHashCode() ?? 0).ToList();
        int samplesPerClass = Math.Max(1, k / groups.Count);
        var selected = new List<DataPoint<T, TInput, TOutput>>();

        foreach (var group in groups)
        {
            var groupItems = group.ToList();
            if (groupItems.Count == 0) continue;

            int take = Math.Min(samplesPerClass, groupItems.Count);

            // Try to extract features for proper herding
            var features = groupItems.Select(p => ExtractFeatures(p.Input)).ToList();

            if (features.All(f => f != null && f.Length > 0))
            {
                // Compute class mean
                int featureDim = features[0]!.Length;
                double[] classMean = new double[featureDim];
                foreach (var f in features)
                {
                    for (int d = 0; d < featureDim; d++)
                        classMean[d] += f![d];
                }
                for (int d = 0; d < featureDim; d++)
                    classMean[d] /= features.Count;

                // Greedy herding: select exemplars that minimize distance to class mean
                var runningMean = new double[featureDim];
                var selectedIndices = new HashSet<int>();

                for (int i = 0; i < take; i++)
                {
                    int bestIdx = -1;
                    double bestDist = double.MaxValue;

                    for (int j = 0; j < groupItems.Count; j++)
                    {
                        if (selectedIndices.Contains(j)) continue;

                        // Compute hypothetical running mean if we added this sample
                        double[] hypotheticalMean = new double[featureDim];
                        for (int d = 0; d < featureDim; d++)
                        {
                            hypotheticalMean[d] = (runningMean[d] * selectedIndices.Count + features[j]![d]) / (selectedIndices.Count + 1);
                        }

                        // Distance from class mean
                        double dist = 0;
                        for (int d = 0; d < featureDim; d++)
                        {
                            double diff = hypotheticalMean[d] - classMean[d];
                            dist += diff * diff;
                        }

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx = j;
                        }
                    }

                    if (bestIdx >= 0)
                    {
                        selectedIndices.Add(bestIdx);
                        selected.Add(groupItems[bestIdx]);
                        // Update running mean
                        for (int d = 0; d < featureDim; d++)
                        {
                            runningMean[d] = (runningMean[d] * (selectedIndices.Count - 1) + features[bestIdx]![d]) / selectedIndices.Count;
                        }
                    }
                }
            }
            else
            {
                // Fallback to random sampling if features can't be extracted
                selected.AddRange(groupItems.OrderBy(_ => _random.Next()).Take(take));
            }
        }

        // Fill remaining slots with random samples if needed
        while (selected.Count < k && selected.Count < items.Count)
        {
            var remaining = items.Except(selected).ToList();
            if (remaining.Count == 0) break;
            selected.Add(remaining[_random.Next(remaining.Count)]);
        }

        return selected.Take(k).ToList();
    }

    private List<DataPoint<T, TInput, TOutput>> KCenterSample(
        List<DataPoint<T, TInput, TOutput>> items, int k)
    {
        // K-Center greedy algorithm: select points that maximize coverage
        // Greedily selects the point farthest from all selected points
        if (k >= items.Count) return new List<DataPoint<T, TInput, TOutput>>(items);

        // Extract features for all items
        var features = items.Select(p => ExtractFeatures(p.Input)).ToList();

        // Check if we can use feature-based distance
        if (features.All(f => f != null && f.Length > 0))
        {
            var selected = new List<DataPoint<T, TInput, TOutput>>();
            var selectedIndices = new HashSet<int>();
            var selectedFeatures = new List<double[]>();

            // Start with a random point
            int firstIdx = _random.Next(items.Count);
            selected.Add(items[firstIdx]);
            selectedIndices.Add(firstIdx);
            selectedFeatures.Add(features[firstIdx]!);

            // Greedily add points that maximize minimum distance to selected set
            while (selected.Count < k)
            {
                int bestIdx = -1;
                double maxMinDist = -1;

                for (int i = 0; i < items.Count; i++)
                {
                    if (selectedIndices.Contains(i)) continue;

                    // Find minimum distance to any selected point
                    double minDist = double.MaxValue;
                    foreach (var sf in selectedFeatures)
                    {
                        double dist = ComputeSquaredDistance(features[i]!, sf);
                        if (dist < minDist) minDist = dist;
                    }

                    // Track the point with maximum minimum distance
                    if (minDist > maxMinDist)
                    {
                        maxMinDist = minDist;
                        bestIdx = i;
                    }
                }

                if (bestIdx >= 0)
                {
                    selected.Add(items[bestIdx]);
                    selectedIndices.Add(bestIdx);
                    selectedFeatures.Add(features[bestIdx]!);
                }
                else break;
            }

            return selected;
        }
        else
        {
            // Fallback: use hash-based diversity when features can't be extracted
            var selected = new List<DataPoint<T, TInput, TOutput>> { items[_random.Next(items.Count)] };
            var selectedHashes = new HashSet<int> { selected[0].Input?.GetHashCode() ?? 0 };

            while (selected.Count < k)
            {
                DataPoint<T, TInput, TOutput>? farthest = null;
                int maxMinDist = int.MinValue;

                foreach (var item in items.Except(selected))
                {
                    int itemHash = item.Input?.GetHashCode() ?? 0;
                    int minDist = selectedHashes.Min(h => Math.Abs(itemHash - h));
                    if (minDist > maxMinDist)
                    {
                        maxMinDist = minDist;
                        farthest = item;
                    }
                }

                if (farthest != null)
                {
                    selected.Add(farthest);
                    selectedHashes.Add(farthest.Input?.GetHashCode() ?? 0);
                }
                else break;
            }

            return selected;
        }
    }

    private List<DataPoint<T, TInput, TOutput>> BoundarySample(
        List<DataPoint<T, TInput, TOutput>> items, int k)
    {
        // Without model access, fall back to class-balanced sampling
        // In a full implementation, this would select examples near decision boundaries
        return ClassBalancedSample(items, k);
    }

    private List<int> UniformReplaySample(int count)
    {
        return Enumerable.Range(0, _buffer.Count)
            .OrderBy(_ => _random.Next())
            .Take(count)
            .ToList();
    }

    private List<int> TaskBalancedReplaySample(int count)
    {
        if (_taskIndices.Count == 0) return new List<int>();

        int samplesPerTask = Math.Max(1, count / _taskIndices.Count);
        var indices = new List<int>();

        foreach (var taskList in _taskIndices.Values)
        {
            int take = Math.Min(samplesPerTask, taskList.Count);
            indices.AddRange(taskList.OrderBy(_ => _random.Next()).Take(take));
        }

        // Fill remaining quota
        while (indices.Count < count && indices.Count < _buffer.Count)
        {
            int idx = _random.Next(_buffer.Count);
            if (!indices.Contains(idx)) indices.Add(idx);
        }

        return indices.Take(count).ToList();
    }

    private List<int> PriorityBasedReplaySample(int count)
    {
        double totalPriority = _priorities.Values.Sum();
        var selected = new List<int>();

        while (selected.Count < count)
        {
            double r = _random.NextDouble() * totalPriority;
            double cumulative = 0;

            foreach (var kvp in _priorities)
            {
                cumulative += kvp.Value;
                if (cumulative >= r && !selected.Contains(kvp.Key))
                {
                    selected.Add(kvp.Key);
                    break;
                }
            }

            // Fallback to prevent infinite loop
            if (selected.Count == 0)
            {
                selected.Add(_random.Next(_buffer.Count));
            }
        }

        return selected;
    }

    private List<int> RecencyWeightedReplaySample(int count)
    {
        // Weight by index (higher index = more recent = higher weight)
        var weights = Enumerable.Range(0, _buffer.Count)
            .Select(i => (Index: i, Weight: (double)(i + 1)))
            .ToList();

        double totalWeight = weights.Sum(w => w.Weight);
        var selected = new List<int>();

        while (selected.Count < count)
        {
            double r = _random.NextDouble() * totalWeight;
            double cumulative = 0;

            foreach (var w in weights)
            {
                cumulative += w.Weight;
                if (cumulative >= r && !selected.Contains(w.Index))
                {
                    selected.Add(w.Index);
                    break;
                }
            }
        }

        return selected;
    }

    #endregion

    #region Private Helper Methods

    private int CalculateTargetSamplesPerTask()
    {
        int currentTasks = Math.Max(1, _taskIndices.Count + 1); // +1 for the new task
        return _maxSize / currentTasks;
    }

    private void MakeRoomForTask(int newTaskId, int neededSpace)
    {
        // Remove samples from overrepresented tasks
        int targetPerTask = (_maxSize - neededSpace) / Math.Max(1, _taskIndices.Count);

        foreach (var kvp in _taskIndices.ToList())
        {
            if (kvp.Key == newTaskId) continue;

            int excess = kvp.Value.Count - targetPerTask;
            if (excess > 0)
            {
                var toRemove = kvp.Value.OrderBy(_ => _random.Next()).Take(excess).ToList();
                foreach (var idx in toRemove.OrderByDescending(i => i))
                {
                    _estimatedMemoryBytes -= EstimateDataPointSize(_buffer[idx]);
                    _buffer.RemoveAt(idx);
                    _priorities.Remove(idx);
                    kvp.Value.Remove(idx);
                }
            }
        }

        RebuildIndices();
    }

    private void RebuildIndices()
    {
        _taskIndices.Clear();
        _priorities.Clear();

        for (int i = 0; i < _buffer.Count; i++)
        {
            var point = _buffer[i];
            if (!_taskIndices.ContainsKey(point.TaskId))
            {
                _taskIndices[point.TaskId] = new List<int>();
            }
            _taskIndices[point.TaskId].Add(i);
            _priorities[i] = 1.0;
        }
    }

    private long EstimateDataPointSize(DataPoint<T, TInput, TOutput> point)
    {
        // Rough estimation: base object overhead + input/output sizes
        long size = 24; // Object header + TaskId int

        if (point.Input is Array inputArray)
            size += inputArray.Length * 8; // Assume 8 bytes per element
        else
            size += 32; // Estimated reference object size

        if (point.Output is Array outputArray)
            size += outputArray.Length * 8;
        else
            size += 32;

        return size;
    }

    /// <summary>
    /// Extracts feature values from an input for distance computation.
    /// Handles various input types like Tensor, Vector, arrays, etc.
    /// </summary>
    /// <param name="input">The input to extract features from.</param>
    /// <returns>Array of feature values, or null if extraction fails.</returns>
    private double[]? ExtractFeatures(TInput input)
    {
        if (input == null) return null;

        // Handle Tensor<T>
        if (input is Tensor<T> tensor)
        {
            var data = tensor.Data;
            var features = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                features[i] = Convert.ToDouble(data[i]);
            }
            return features;
        }

        // Handle Vector<T>
        if (input is Vector<T> vector)
        {
            var features = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                features[i] = Convert.ToDouble(vector[i]);
            }
            return features;
        }

        // Handle double[]
        if (input is double[] doubleArray)
        {
            return (double[])doubleArray.Clone();
        }

        // Handle float[]
        if (input is float[] floatArray)
        {
            return floatArray.Select(f => (double)f).ToArray();
        }

        // Handle T[]
        if (input is T[] tArray)
        {
            return tArray.Select(t => Convert.ToDouble(t)).ToArray();
        }

        // Handle double[][]  - flatten
        if (input is double[][] double2DArray)
        {
            return double2DArray.SelectMany(row => row).ToArray();
        }

        // Handle float[][] - flatten
        if (input is float[][] float2DArray)
        {
            return float2DArray.SelectMany(row => row.Select(f => (double)f)).ToArray();
        }

        // Handle generic array
        if (input is Array array)
        {
            var features = new List<double>();
            foreach (var item in array)
            {
                if (item is IConvertible convertible)
                {
                    features.Add(convertible.ToDouble(null));
                }
            }
            return features.Count > 0 ? features.ToArray() : null;
        }

        // Handle single numeric value
        if (input is IConvertible singleValue)
        {
            return new[] { singleValue.ToDouble(null) };
        }

        return null;
    }

    /// <summary>
    /// Computes the squared Euclidean distance between two feature vectors.
    /// </summary>
    private double ComputeSquaredDistance(double[] a, double[] b)
    {
        if (a == null || b == null || a.Length != b.Length)
            return double.MaxValue;

        double dist = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            dist += diff * diff;
        }
        return dist;
    }

    #endregion
}

/// <summary>
/// Statistics about the experience replay buffer.
/// </summary>
public class BufferStatistics
{
    /// <summary>
    /// Current number of stored examples.
    /// </summary>
    public int Count { get; init; }

    /// <summary>
    /// Maximum buffer capacity.
    /// </summary>
    public int MaxSize { get; init; }

    /// <summary>
    /// Number of distinct tasks in the buffer.
    /// </summary>
    public int TaskCount { get; init; }

    /// <summary>
    /// Number of examples per task.
    /// </summary>
    public Dictionary<int, int> TaskDistribution { get; init; } = new();

    /// <summary>
    /// Total samples ever added to the buffer.
    /// </summary>
    public int TotalSamplesProcessed { get; init; }

    /// <summary>
    /// Total samples returned via replay.
    /// </summary>
    public int TotalReplaySamples { get; init; }

    /// <summary>
    /// Estimated memory usage in bytes.
    /// </summary>
    public long EstimatedMemoryBytes { get; init; }

    /// <summary>
    /// Average priority of stored samples.
    /// </summary>
    public double AveragePriority { get; init; }

    /// <summary>
    /// Ratio of current count to max size.
    /// </summary>
    public double FillRatio { get; init; }
}

/// <summary>
/// Represents a single data point with input and output.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A data point represents one example from the training data.
/// It contains the input (what we show the model), the expected output (what we want it to predict),
/// and a task ID (which task this example belongs to).</para>
/// </remarks>
public class DataPoint<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the input data.
    /// </summary>
    public TInput Input { get; }

    /// <summary>
    /// Gets the expected output data.
    /// </summary>
    public TOutput Output { get; }

    /// <summary>
    /// Gets the task identifier this data point belongs to.
    /// </summary>
    public int TaskId { get; }

    /// <summary>
    /// Gets or sets the timestamp when this data point was added.
    /// </summary>
    public DateTime? AddedAt { get; init; }

    /// <summary>
    /// Gets or sets additional metadata about this data point.
    /// </summary>
    public IReadOnlyDictionary<string, object>? Metadata { get; init; }

    /// <summary>
    /// Initializes a new data point.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="output">The expected output data.</param>
    /// <param name="taskId">The task identifier.</param>
    public DataPoint(TInput input, TOutput output, int taskId)
    {
        Input = input;
        Output = output;
        TaskId = taskId;
    }

    /// <summary>
    /// Returns a string representation of the data point.
    /// </summary>
    public override string ToString()
    {
        return $"DataPoint[TaskId={TaskId}, Input={Input}, Output={Output}]";
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
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a dataset into individual data points that can be
    /// stored in the experience replay buffer. Each data point includes the task ID so we can
    /// track which task each example came from.</para>
    /// </remarks>
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
            yield return new DataPoint<T, TInput, TOutput>(input, output, taskId)
            {
                AddedAt = DateTime.UtcNow
            };
        }
    }

    /// <summary>
    /// Gets a subset of data points from the dataset.
    /// </summary>
    /// <param name="dataset">The dataset to sample from.</param>
    /// <param name="indices">The indices to retrieve.</param>
    /// <param name="taskId">Optional task ID to assign (default: 0).</param>
    /// <returns>Enumerable of DataPoint objects at the specified indices.</returns>
    public static IEnumerable<DataPoint<T, TInput, TOutput>> GetDataPointsAt<T, TInput, TOutput>(
        this IDataset<T, TInput, TOutput> dataset,
        IEnumerable<int> indices,
        int taskId = 0)
    {
        if (dataset == null)
            throw new ArgumentNullException(nameof(dataset));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        foreach (var i in indices)
        {
            if (i >= 0 && i < dataset.Count)
            {
                var input = dataset.GetInput(i);
                var output = dataset.GetOutput(i);
                yield return new DataPoint<T, TInput, TOutput>(input, output, taskId)
                {
                    AddedAt = DateTime.UtcNow
                };
            }
        }
    }
}

/// <summary>
/// Alias for DataPoint used in experience replay contexts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is an alias for DataPoint that emphasizes
/// its use in experience replay for continual learning.</para>
/// </remarks>
public class ExperienceDataPoint<T, TInput, TOutput> : DataPoint<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new experience data point.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="output">The expected output data.</param>
    /// <param name="taskId">The task identifier.</param>
    public ExperienceDataPoint(TInput input, TOutput output, int taskId)
        : base(input, output, taskId)
    {
    }

    /// <summary>
    /// Creates an ExperienceDataPoint from an existing DataPoint.
    /// </summary>
    /// <param name="dataPoint">The data point to convert.</param>
    /// <returns>A new ExperienceDataPoint with the same data.</returns>
    public static ExperienceDataPoint<T, TInput, TOutput> FromDataPoint(DataPoint<T, TInput, TOutput> dataPoint)
    {
        return new ExperienceDataPoint<T, TInput, TOutput>(
            dataPoint.Input,
            dataPoint.Output,
            dataPoint.TaskId)
        {
            AddedAt = dataPoint.AddedAt,
            Metadata = dataPoint.Metadata
        };
    }
}
