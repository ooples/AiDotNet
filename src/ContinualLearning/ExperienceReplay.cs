using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Experience Replay for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Experience Replay is one of the simplest and most effective
/// continual learning strategies. It stores examples from previous tasks and mixes them with
/// new task data during training, directly rehearsing old knowledge.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After each task, store a subset of examples in a replay buffer.</description></item>
/// <item><description>During training on new tasks, sample from the buffer and train on both
/// new data and replayed old data.</description></item>
/// <item><description>This directly prevents forgetting by repeatedly practicing old tasks.</description></item>
/// </list>
///
/// <para><b>Buffer Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Reservoir:</b> Each sample has equal probability of being kept.</description></item>
/// <item><description><b>Ring:</b> FIFO queue, oldest samples are removed first.</description></item>
/// <item><description><b>ClassBalanced:</b> Maintains equal samples per class.</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Simple to implement and understand.</description></item>
/// <item><description>Very effective in practice.</description></item>
/// <item><description>Can be combined with other strategies.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Ratcliff, R. "Connectionist models of recognition memory" (1990).
/// Psychological Review. (Original concept); Rolnick et al. "Experience Replay for Continual
/// Learning" (2019). NeurIPS. (Modern application)</para>
/// </remarks>
public class ExperienceReplay<T> : IContinualLearningStrategy<T>
{
    /// <summary>
    /// Defines the buffer management strategy.
    /// </summary>
    public enum BufferStrategy
    {
        /// <summary>Reservoir sampling - uniform random replacement.</summary>
        Reservoir,
        /// <summary>Ring buffer - FIFO queue.</summary>
        Ring,
        /// <summary>Class-balanced sampling.</summary>
        ClassBalanced
    }

    private readonly INumericOperations<T> _numOps;
    private readonly List<(Tensor<T> input, Tensor<T> target, int taskId)> _buffer;
    private readonly int _maxBufferSize;
    private readonly BufferStrategy _strategy;
    private readonly double _replayRatio;
    private double _lambda;
    private readonly Random _random;
    private int _totalSamplesSeen;

    /// <summary>
    /// Initializes a new instance of the ExperienceReplay class.
    /// </summary>
    /// <param name="maxBufferSize">Maximum samples to store in buffer (default: 1000).</param>
    /// <param name="replayRatio">Ratio of replay samples to new samples (default: 0.5).</param>
    /// <param name="strategy">Buffer management strategy (default: Reservoir).</param>
    /// <param name="lambda">Weight for replay loss contribution (default: 1.0).</param>
    /// <param name="seed">Random seed for reproducibility (default: null).</param>
    public ExperienceReplay(
        int maxBufferSize = 1000,
        double replayRatio = 0.5,
        BufferStrategy strategy = BufferStrategy.Reservoir,
        double lambda = 1.0,
        int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _buffer = [];
        _maxBufferSize = maxBufferSize;
        _replayRatio = replayRatio;
        _strategy = strategy;
        _lambda = lambda;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _totalSamplesSeen = 0;
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets the current buffer size.
    /// </summary>
    public int BufferSize => _buffer.Count;

    /// <summary>
    /// Gets the replay ratio.
    /// </summary>
    public double ReplayRatio => _replayRatio;

    /// <summary>
    /// Gets the buffer strategy.
    /// </summary>
    public BufferStrategy Strategy => _strategy;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        // Experience replay doesn't need to do anything before a task
        Guard.NotNull(network);
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        Guard.NotNull(network);
        _ = taskData.inputs ?? throw new ArgumentNullException(nameof(taskData));
        _ = taskData.targets ?? throw new ArgumentNullException(nameof(taskData));

        // Add task samples to buffer
        AddToBuffer(taskData.inputs, taskData.targets, taskId);
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        Guard.NotNull(network);

        if (_buffer.Count == 0)
        {
            return _numOps.Zero;
        }

        // Sample from buffer and compute replay loss
        var (inputs, targets) = SampleReplayBatch();

        // Forward pass on replay samples
        var outputs = network.Predict(inputs);

        // Compute MSE loss
        var loss = _numOps.Zero;
        for (int i = 0; i < outputs.Length; i++)
        {
            var diff = _numOps.Subtract(outputs[i], targets[i]);
            var squared = _numOps.Multiply(diff, diff);
            loss = _numOps.Add(loss, squared);
        }

        var n = _numOps.FromDouble(outputs.Length);
        loss = _numOps.Divide(loss, n);

        // Scale by lambda
        var lambdaT = _numOps.FromDouble(_lambda);
        return _numOps.Multiply(lambdaT, loss);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        // Experience replay works through loss, not gradient modification
        Guard.NotNull(network);
        Guard.NotNull(gradients);
        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _buffer.Clear();
        _totalSamplesSeen = 0;
    }

    /// <summary>
    /// Samples a batch from the replay buffer for training.
    /// </summary>
    /// <returns>Tuple of inputs and targets tensors.</returns>
    public (Tensor<T> inputs, Tensor<T> targets) SampleReplayBatch()
    {
        if (_buffer.Count == 0)
        {
            throw new InvalidOperationException("Replay buffer is empty");
        }

        var batchSize = Math.Min(_buffer.Count, (int)(_maxBufferSize * _replayRatio / 10));
        batchSize = Math.Max(1, batchSize);

        var indices = Enumerable.Range(0, _buffer.Count)
            .OrderBy(_ => _random.Next())
            .Take(batchSize)
            .ToList();

        return BuildBatchFromIndices(indices);
    }

    /// <summary>
    /// Samples a mixed batch combining current task data with replay data.
    /// </summary>
    public (Tensor<T> inputs, Tensor<T> targets) SampleMixedBatch(
        Tensor<T> currentInputs,
        Tensor<T> currentTargets,
        int batchSize)
    {
        if (_buffer.Count == 0)
        {
            return (currentInputs, currentTargets);
        }

        var replayCount = (int)(batchSize * _replayRatio);
        var currentCount = batchSize - replayCount;

        var mixedInputs = new List<Tensor<T>>();
        var mixedTargets = new List<Tensor<T>>();

        // Add current task samples
        var currentSize = currentInputs.Shape[0];
        var currentIndices = Enumerable.Range(0, currentSize)
            .OrderBy(_ => _random.Next())
            .Take(Math.Min(currentCount, currentSize))
            .ToList();

        foreach (var idx in currentIndices)
        {
            mixedInputs.Add(ExtractSample(currentInputs, idx));
            mixedTargets.Add(ExtractSample(currentTargets, idx));
        }

        // Add replay samples
        var replayIndices = Enumerable.Range(0, _buffer.Count)
            .OrderBy(_ => _random.Next())
            .Take(Math.Min(replayCount, _buffer.Count))
            .ToList();

        foreach (var idx in replayIndices)
        {
            mixedInputs.Add(_buffer[idx].input);
            mixedTargets.Add(_buffer[idx].target);
        }

        return CombineTensors(mixedInputs, mixedTargets);
    }

    /// <summary>
    /// Adds samples from a task to the replay buffer.
    /// </summary>
    private void AddToBuffer(Tensor<T> inputs, Tensor<T> targets, int taskId)
    {
        var batchSize = inputs.Shape[0];

        for (int i = 0; i < batchSize; i++)
        {
            var input = ExtractSample(inputs, i);
            var target = ExtractSample(targets, i);

            _totalSamplesSeen++;

            switch (_strategy)
            {
                case BufferStrategy.Reservoir:
                    AddReservoir(input, target, taskId);
                    break;
                case BufferStrategy.Ring:
                    AddRing(input, target, taskId);
                    break;
                case BufferStrategy.ClassBalanced:
                    AddClassBalanced(input, target, taskId);
                    break;
            }
        }
    }

    /// <summary>
    /// Adds sample using reservoir sampling.
    /// </summary>
    private void AddReservoir(Tensor<T> input, Tensor<T> target, int taskId)
    {
        if (_buffer.Count < _maxBufferSize)
        {
            _buffer.Add((input, target, taskId));
        }
        else
        {
            // Reservoir sampling: replace with probability maxSize/totalSeen
            var replaceProb = (double)_maxBufferSize / _totalSamplesSeen;
            if (_random.NextDouble() < replaceProb)
            {
                var replaceIdx = _random.Next(_buffer.Count);
                _buffer[replaceIdx] = (input, target, taskId);
            }
        }
    }

    /// <summary>
    /// Adds sample using ring buffer (FIFO).
    /// </summary>
    private void AddRing(Tensor<T> input, Tensor<T> target, int taskId)
    {
        if (_buffer.Count < _maxBufferSize)
        {
            _buffer.Add((input, target, taskId));
        }
        else
        {
            // Remove oldest and add new
            _buffer.RemoveAt(0);
            _buffer.Add((input, target, taskId));
        }
    }

    /// <summary>
    /// Adds sample maintaining class balance.
    /// </summary>
    private void AddClassBalanced(Tensor<T> input, Tensor<T> target, int taskId)
    {
        // For class-balanced, we group by task and maintain equal per-task
        var tasksInBuffer = _buffer.Select(b => b.taskId).Distinct().ToList();
        if (!tasksInBuffer.Contains(taskId))
        {
            tasksInBuffer.Add(taskId);
        }

        var maxPerTask = _maxBufferSize / Math.Max(1, tasksInBuffer.Count);
        var currentTaskCount = _buffer.Count(b => b.taskId == taskId);

        if (currentTaskCount < maxPerTask)
        {
            _buffer.Add((input, target, taskId));
        }
        else
        {
            // Replace random sample from same task
            var taskIndices = _buffer
                .Select((item, idx) => (item, idx))
                .Where(x => x.item.taskId == taskId)
                .Select(x => x.idx)
                .ToList();

            if (taskIndices.Count > 0)
            {
                var replaceIdx = taskIndices[_random.Next(taskIndices.Count)];
                _buffer[replaceIdx] = (input, target, taskId);
            }
        }

        // Rebalance if buffer is over capacity
        while (_buffer.Count > _maxBufferSize)
        {
            // Remove from task with most samples
            var taskCounts = _buffer.GroupBy(b => b.taskId)
                .Select(g => (TaskId: g.Key, Count: g.Count()))
                .OrderByDescending(x => x.Count)
                .First();

            var removeIdx = _buffer
                .Select((item, idx) => (item, idx))
                .Where(x => x.item.taskId == taskCounts.TaskId)
                .Select(x => x.idx)
                .First();

            _buffer.RemoveAt(removeIdx);
        }
    }

    /// <summary>
    /// Extracts a single sample from a batch tensor.
    /// </summary>
    private Tensor<T> ExtractSample(Tensor<T> batch, int index)
    {
        var newShape = new int[batch.Shape.Length];
        newShape[0] = 1;
        for (int i = 1; i < batch.Shape.Length; i++)
        {
            newShape[i] = batch.Shape[i];
        }

        var sampleSize = 1;
        for (int i = 1; i < batch.Shape.Length; i++)
        {
            sampleSize *= batch.Shape[i];
        }

        var data = new Vector<T>(sampleSize);
        var startIdx = index * sampleSize;
        for (int i = 0; i < sampleSize; i++)
        {
            data[i] = batch[startIdx + i];
        }

        return new Tensor<T>(newShape, data);
    }

    /// <summary>
    /// Builds a batch from buffer indices.
    /// </summary>
    private (Tensor<T> inputs, Tensor<T> targets) BuildBatchFromIndices(List<int> indices)
    {
        if (indices.Count == 0)
        {
            throw new ArgumentException("No indices provided", nameof(indices));
        }

        var firstSample = _buffer[indices[0]];
        var inputSampleSize = firstSample.input.Length;
        var targetSampleSize = firstSample.target.Length;

        var inputShape = new int[firstSample.input.Shape.Length];
        inputShape[0] = indices.Count;
        for (int i = 1; i < firstSample.input.Shape.Length; i++)
        {
            inputShape[i] = firstSample.input.Shape[i];
        }

        var targetShape = new int[firstSample.target.Shape.Length];
        targetShape[0] = indices.Count;
        for (int i = 1; i < firstSample.target.Shape.Length; i++)
        {
            targetShape[i] = firstSample.target.Shape[i];
        }

        var inputData = new Vector<T>(indices.Count * inputSampleSize);
        var targetData = new Vector<T>(indices.Count * targetSampleSize);

        for (int i = 0; i < indices.Count; i++)
        {
            var sample = _buffer[indices[i]];
            for (int j = 0; j < inputSampleSize; j++)
            {
                inputData[i * inputSampleSize + j] = sample.input[j];
            }
            for (int j = 0; j < targetSampleSize; j++)
            {
                targetData[i * targetSampleSize + j] = sample.target[j];
            }
        }

        return (new Tensor<T>(inputShape, inputData), new Tensor<T>(targetShape, targetData));
    }

    /// <summary>
    /// Combines multiple tensors into single batch tensors.
    /// </summary>
    private (Tensor<T> inputs, Tensor<T> targets) CombineTensors(
        List<Tensor<T>> inputs,
        List<Tensor<T>> targets)
    {
        if (inputs.Count == 0)
        {
            throw new ArgumentException("No tensors to combine", nameof(inputs));
        }

        var firstInput = inputs[0];
        var firstTarget = targets[0];

        var inputSampleSize = firstInput.Length;
        var targetSampleSize = firstTarget.Length;

        var inputShape = new int[firstInput.Shape.Length];
        inputShape[0] = inputs.Count;
        for (int i = 1; i < firstInput.Shape.Length; i++)
        {
            inputShape[i] = firstInput.Shape[i];
        }

        var targetShape = new int[firstTarget.Shape.Length];
        targetShape[0] = targets.Count;
        for (int i = 1; i < firstTarget.Shape.Length; i++)
        {
            targetShape[i] = firstTarget.Shape[i];
        }

        var inputData = new Vector<T>(inputs.Count * inputSampleSize);
        var targetData = new Vector<T>(targets.Count * targetSampleSize);

        for (int i = 0; i < inputs.Count; i++)
        {
            for (int j = 0; j < inputSampleSize; j++)
            {
                inputData[i * inputSampleSize + j] = inputs[i][j];
            }
            for (int j = 0; j < targetSampleSize; j++)
            {
                targetData[i * targetSampleSize + j] = targets[i][j];
            }
        }

        return (new Tensor<T>(inputShape, inputData), new Tensor<T>(targetShape, targetData));
    }
}
