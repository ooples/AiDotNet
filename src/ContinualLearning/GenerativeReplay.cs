using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Generative Replay (also known as Deep Generative Replay) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Generative Replay uses a generative model (like a VAE or GAN)
/// to create pseudo-examples from previous tasks instead of storing real examples. This
/// enables rehearsal without storing actual data, which is useful for privacy-sensitive
/// applications or when memory is limited.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Train a generative model alongside the main model (called the "solver").</description></item>
/// <item><description>After each task, use the generator to create pseudo-examples of previous tasks.</description></item>
/// <item><description>When learning new tasks, mix real new data with generated pseudo-examples.</description></item>
/// <item><description>The generator is also trained on mixed data to maintain its ability to generate old examples.</description></item>
/// </list>
///
/// <para><b>Key Components:</b></para>
/// <list type="bullet">
/// <item><description><b>Solver:</b> The main model being trained on tasks.</description></item>
/// <item><description><b>Generator:</b> Produces synthetic examples of previous tasks.</description></item>
/// <item><description><b>Scholar:</b> Combined solver + generator system.</description></item>
/// </list>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>No need to store real examples (privacy-preserving).</description></item>
/// <item><description>Constant memory regardless of number of tasks.</description></item>
/// <item><description>Can generate unlimited pseudo-examples for rehearsal.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Shin, H., Lee, J.K., Kim, J., and Kim, J. "Continual Learning with
/// Deep Generative Replay" (2017). NeurIPS.</para>
/// </remarks>
public class GenerativeReplay<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private IGenerativeModel<T>? _generator;
    private readonly int _replayBatchSize;
    private readonly double _replayRatio;
    private double _lambda;
    private int _taskCount;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the GenerativeReplay class.
    /// </summary>
    /// <param name="replayBatchSize">Number of pseudo-examples to generate per batch (default: 32).</param>
    /// <param name="replayRatio">Ratio of replay samples to new samples (default: 0.5).</param>
    /// <param name="lambda">Weight for replay loss contribution (default: 1.0).</param>
    /// <param name="seed">Random seed for reproducibility (default: null for random).</param>
    public GenerativeReplay(
        int replayBatchSize = 32,
        double replayRatio = 0.5,
        double lambda = 1.0,
        int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _generator = null;
        _replayBatchSize = replayBatchSize;
        _replayRatio = replayRatio;
        _lambda = lambda;
        _taskCount = 0;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets the number of tasks processed.
    /// </summary>
    public int TaskCount => _taskCount;

    /// <summary>
    /// Gets the replay batch size.
    /// </summary>
    public int ReplayBatchSize => _replayBatchSize;

    /// <summary>
    /// Gets the replay ratio.
    /// </summary>
    public double ReplayRatio => _replayRatio;

    /// <summary>
    /// Sets the generative model used for replay.
    /// </summary>
    /// <param name="generator">The generative model (VAE, GAN, etc.).</param>
    public void SetGenerator(IGenerativeModel<T> generator)
    {
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
    }

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        // Generative replay doesn't need special setup before a task
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = taskData.inputs ?? throw new ArgumentNullException(nameof(taskData));
        _ = taskData.targets ?? throw new ArgumentNullException(nameof(taskData));

        // Update the generator with task data (if generator is set)
        // In practice, the generator should be trained alongside the solver
        // This hook allows updating internal state after task completion
        _taskCount++;
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        if (_generator == null || _taskCount == 0)
        {
            return _numOps.Zero;
        }

        // Generate pseudo-examples from the generator
        var (replayInputs, replayTargets) = GenerateReplaySamples();

        if (replayInputs == null || replayTargets == null)
        {
            return _numOps.Zero;
        }

        // Forward pass on generated samples
        var outputs = network.Predict(replayInputs);

        // Compute MSE loss on replay samples
        var loss = _numOps.Zero;
        for (int i = 0; i < outputs.Length; i++)
        {
            var diff = _numOps.Subtract(outputs[i], replayTargets[i]);
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
        // Generative replay works through loss computation, not gradient modification
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = gradients ?? throw new ArgumentNullException(nameof(gradients));
        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _taskCount = 0;
        // Note: Generator is not reset as it may be shared
    }

    /// <summary>
    /// Generates pseudo-examples for replay using the generator.
    /// </summary>
    /// <returns>Tuple of generated inputs and targets, or nulls if generator not available.</returns>
    public (Tensor<T>? inputs, Tensor<T>? targets) GenerateReplaySamples()
    {
        if (_generator == null)
        {
            return (null, null);
        }

        return _generator.GenerateSamples(_replayBatchSize);
    }

    /// <summary>
    /// Creates a mixed training batch combining new data with generated replay data.
    /// </summary>
    /// <param name="currentInputs">Current task inputs.</param>
    /// <param name="currentTargets">Current task targets.</param>
    /// <param name="batchSize">Total batch size.</param>
    /// <returns>Mixed batch of current and replay data.</returns>
    public (Tensor<T> inputs, Tensor<T> targets) CreateMixedBatch(
        Tensor<T> currentInputs,
        Tensor<T> currentTargets,
        int batchSize)
    {
        _ = currentInputs ?? throw new ArgumentNullException(nameof(currentInputs));
        _ = currentTargets ?? throw new ArgumentNullException(nameof(currentTargets));

        if (_generator == null || _taskCount == 0)
        {
            return (currentInputs, currentTargets);
        }

        var replayCount = (int)(batchSize * _replayRatio);
        var currentCount = batchSize - replayCount;

        // Sample from current task
        var currentBatchSize = currentInputs.Shape[0];
        var currentIndices = Enumerable.Range(0, currentBatchSize)
            .OrderBy(_ => _random.Next())
            .Take(Math.Min(currentCount, currentBatchSize))
            .ToArray();

        // Generate replay samples
        var (replayInputs, replayTargets) = _generator.GenerateSamples(replayCount);

        if (replayInputs == null || replayTargets == null)
        {
            return (SampleTensor(currentInputs, currentIndices),
                    SampleTensor(currentTargets, currentIndices));
        }

        // Combine current and replay samples
        return CombineTensors(
            SampleTensor(currentInputs, currentIndices),
            SampleTensor(currentTargets, currentIndices),
            replayInputs,
            replayTargets);
    }

    /// <summary>
    /// Samples specific indices from a tensor.
    /// </summary>
    private Tensor<T> SampleTensor(Tensor<T> tensor, int[] indices)
    {
        var newShape = new int[tensor.Shape.Length];
        newShape[0] = indices.Length;
        for (int i = 1; i < tensor.Shape.Length; i++)
        {
            newShape[i] = tensor.Shape[i];
        }

        var sampleSize = 1;
        for (int i = 1; i < tensor.Shape.Length; i++)
        {
            sampleSize *= tensor.Shape[i];
        }

        var data = new Vector<T>(indices.Length * sampleSize);
        for (int i = 0; i < indices.Length; i++)
        {
            var srcStart = indices[i] * sampleSize;
            var dstStart = i * sampleSize;
            for (int j = 0; j < sampleSize; j++)
            {
                data[dstStart + j] = tensor[srcStart + j];
            }
        }

        return new Tensor<T>(newShape, data);
    }

    /// <summary>
    /// Combines current and replay tensors into a single batch.
    /// </summary>
    private (Tensor<T> inputs, Tensor<T> targets) CombineTensors(
        Tensor<T> currentInputs,
        Tensor<T> currentTargets,
        Tensor<T> replayInputs,
        Tensor<T> replayTargets)
    {
        var totalBatchSize = currentInputs.Shape[0] + replayInputs.Shape[0];

        // Input tensor
        var inputShape = new int[currentInputs.Shape.Length];
        inputShape[0] = totalBatchSize;
        for (int i = 1; i < currentInputs.Shape.Length; i++)
        {
            inputShape[i] = currentInputs.Shape[i];
        }

        var inputSampleSize = 1;
        for (int i = 1; i < currentInputs.Shape.Length; i++)
        {
            inputSampleSize *= currentInputs.Shape[i];
        }

        var inputData = new Vector<T>(totalBatchSize * inputSampleSize);

        // Copy current inputs
        for (int i = 0; i < currentInputs.Length; i++)
        {
            inputData[i] = currentInputs[i];
        }

        // Copy replay inputs
        var offset = currentInputs.Length;
        for (int i = 0; i < replayInputs.Length; i++)
        {
            inputData[offset + i] = replayInputs[i];
        }

        // Target tensor
        var targetShape = new int[currentTargets.Shape.Length];
        targetShape[0] = totalBatchSize;
        for (int i = 1; i < currentTargets.Shape.Length; i++)
        {
            targetShape[i] = currentTargets.Shape[i];
        }

        var targetSampleSize = 1;
        for (int i = 1; i < currentTargets.Shape.Length; i++)
        {
            targetSampleSize *= currentTargets.Shape[i];
        }

        var targetData = new Vector<T>(totalBatchSize * targetSampleSize);

        // Copy current targets
        for (int i = 0; i < currentTargets.Length; i++)
        {
            targetData[i] = currentTargets[i];
        }

        // Copy replay targets
        offset = currentTargets.Length;
        for (int i = 0; i < replayTargets.Length; i++)
        {
            targetData[offset + i] = replayTargets[i];
        }

        return (new Tensor<T>(inputShape, inputData), new Tensor<T>(targetShape, targetData));
    }
}

/// <summary>
/// Interface for generative models used with GenerativeReplay.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface IGenerativeModel<T>
{
    /// <summary>
    /// Generates synthetic samples from the learned distribution.
    /// </summary>
    /// <param name="count">Number of samples to generate.</param>
    /// <returns>Tuple of generated inputs and corresponding targets.</returns>
    (Tensor<T>? inputs, Tensor<T>? targets) GenerateSamples(int count);

    /// <summary>
    /// Trains the generative model on the provided data.
    /// </summary>
    /// <param name="inputs">Training inputs.</param>
    /// <param name="targets">Training targets.</param>
    void Train(Tensor<T> inputs, Tensor<T> targets);
}
