using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Averaged Gradient Episodic Memory (A-GEM) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A-GEM is a more efficient version of GEM that uses a single
/// random sample from the episodic memory instead of checking constraints against all stored
/// examples. This makes it much faster while maintaining similar performance.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Store examples from each completed task in episodic memory.</description></item>
/// <item><description>For each gradient update, compute a reference gradient from a random batch
/// of ALL stored memories (not per-task).</description></item>
/// <item><description>If the proposed gradient would hurt past tasks (negative dot product with
/// reference), project it to be orthogonal.</description></item>
/// </list>
///
/// <para><b>Key Difference from GEM:</b></para>
/// <list type="bullet">
/// <item><description>GEM: Checks constraints for each task separately (expensive QP solver).</description></item>
/// <item><description>A-GEM: Single averaged constraint from random memory sample (simple projection).</description></item>
/// </list>
///
/// <para><b>Projection Formula:</b></para>
/// <para>If g · g_ref &lt; 0: g_proj = g - (g · g_ref / g_ref · g_ref) × g_ref</para>
///
/// <para><b>Reference:</b> Chaudhry, A. et al. "Efficient Lifelong Learning with A-GEM" (2019). ICLR.</para>
/// </remarks>
public class AveragedGEM<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<(Tensor<T> inputs, Tensor<T> targets)> _episodicMemory;
    private readonly int _memorySize;
    private readonly int _sampleSize;
    private double _lambda;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the AveragedGEM class.
    /// </summary>
    /// <param name="memorySize">Maximum samples to store per task (default: 256).</param>
    /// <param name="sampleSize">Batch size to sample from memory for reference gradient (default: 64).</param>
    /// <param name="lambda">Regularization strength for compatibility (default: 1.0).</param>
    /// <param name="seed">Random seed for reproducibility (default: null for random).</param>
    public AveragedGEM(int memorySize = 256, int sampleSize = 64, double lambda = 1.0, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _episodicMemory = [];
        _memorySize = memorySize;
        _sampleSize = sampleSize;
        _lambda = lambda;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets the number of tasks stored in episodic memory.
    /// </summary>
    public int TaskCount => _episodicMemory.Count;

    /// <summary>
    /// Gets the total number of samples in episodic memory.
    /// </summary>
    public int TotalMemorySize => _episodicMemory.Sum(m => m.inputs.Shape[0]);

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        // A-GEM doesn't need to do anything before a task
        Guard.NotNull(network);
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        Guard.NotNull(network);
        _ = taskData.inputs ?? throw new ArgumentNullException(nameof(taskData));
        _ = taskData.targets ?? throw new ArgumentNullException(nameof(taskData));

        // Sample and store examples from the completed task
        var sampledData = SampleMemory(taskData.inputs, taskData.targets);
        _episodicMemory.Add(sampledData);
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        // A-GEM uses gradient projection, not loss-based regularization
        return _numOps.Zero;
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        Guard.NotNull(network);
        Guard.NotNull(gradients);

        if (_episodicMemory.Count == 0)
        {
            return gradients;
        }

        // Sample a batch from episodic memory
        var (memInputs, memTargets) = SampleFromEpisodicMemory();

        // Compute reference gradient on memory sample
        var refGradient = ComputeReferenceGradient(network, memInputs, memTargets);

        // Check if gradient violates constraint
        var dotProduct = DotProduct(gradients, refGradient);

        if (_numOps.LessThan(dotProduct, _numOps.Zero))
        {
            // Project gradient: g_proj = g - (g · g_ref / g_ref · g_ref) × g_ref
            var refNormSq = DotProduct(refGradient, refGradient);

            if (_numOps.GreaterThan(refNormSq, _numOps.Zero))
            {
                var scale = _numOps.Divide(dotProduct, refNormSq);
                for (int i = 0; i < gradients.Length; i++)
                {
                    var adjustment = _numOps.Multiply(scale, refGradient[i]);
                    gradients[i] = _numOps.Subtract(gradients[i], adjustment);
                }
            }
        }

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _episodicMemory.Clear();
    }

    /// <summary>
    /// Samples a random batch from all episodic memories combined.
    /// </summary>
    private (Tensor<T> inputs, Tensor<T> targets) SampleFromEpisodicMemory()
    {
        // Collect all available samples
        var allIndices = new List<(int taskIdx, int sampleIdx)>();
        for (int t = 0; t < _episodicMemory.Count; t++)
        {
            var taskSize = _episodicMemory[t].inputs.Shape[0];
            for (int s = 0; s < taskSize; s++)
            {
                allIndices.Add((t, s));
            }
        }

        // Randomly sample
        var samplesToTake = Math.Min(_sampleSize, allIndices.Count);
        var sampledIndices = allIndices.OrderBy(_ => _random.Next()).Take(samplesToTake).ToList();

        // Build sampled tensors
        if (sampledIndices.Count == 0)
        {
            throw new InvalidOperationException("No samples available in episodic memory");
        }

        var firstTask = _episodicMemory[sampledIndices[0].taskIdx];
        var inputSampleSize = firstTask.inputs.Length / firstTask.inputs.Shape[0];
        var targetSampleSize = firstTask.targets.Length / firstTask.targets.Shape[0];

        var inputShape = new int[firstTask.inputs.Shape.Length];
        inputShape[0] = sampledIndices.Count;
        for (int i = 1; i < firstTask.inputs.Shape.Length; i++)
        {
            inputShape[i] = firstTask.inputs.Shape[i];
        }

        var targetShape = new int[firstTask.targets.Shape.Length];
        targetShape[0] = sampledIndices.Count;
        for (int i = 1; i < firstTask.targets.Shape.Length; i++)
        {
            targetShape[i] = firstTask.targets.Shape[i];
        }

        var inputData = new Vector<T>(sampledIndices.Count * inputSampleSize);
        var targetData = new Vector<T>(sampledIndices.Count * targetSampleSize);

        for (int i = 0; i < sampledIndices.Count; i++)
        {
            var (taskIdx, sampleIdx) = sampledIndices[i];
            var taskInputs = _episodicMemory[taskIdx].inputs;
            var taskTargets = _episodicMemory[taskIdx].targets;

            var srcInputStart = sampleIdx * inputSampleSize;
            var srcTargetStart = sampleIdx * targetSampleSize;
            var dstInputStart = i * inputSampleSize;
            var dstTargetStart = i * targetSampleSize;

            for (int j = 0; j < inputSampleSize; j++)
            {
                inputData[dstInputStart + j] = taskInputs[srcInputStart + j];
            }
            for (int j = 0; j < targetSampleSize; j++)
            {
                targetData[dstTargetStart + j] = taskTargets[srcTargetStart + j];
            }
        }

        return (new Tensor<T>(inputShape, inputData), new Tensor<T>(targetShape, targetData));
    }

    /// <summary>
    /// Samples examples from task data to store in memory.
    /// </summary>
    private (Tensor<T> inputs, Tensor<T> targets) SampleMemory(Tensor<T> inputs, Tensor<T> targets)
    {
        var batchSize = inputs.Shape[0];
        var samplesToKeep = Math.Min(_memorySize, batchSize);

        if (samplesToKeep >= batchSize)
        {
            return (inputs.Clone(), targets.Clone());
        }

        var indices = Enumerable.Range(0, batchSize)
            .OrderBy(_ => _random.Next())
            .Take(samplesToKeep)
            .ToArray();

        return (SampleTensor(inputs, indices), SampleTensor(targets, indices));
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
    /// Computes the reference gradient on memory samples.
    /// </summary>
    private Vector<T> ComputeReferenceGradient(INeuralNetwork<T> network, Tensor<T> inputs, Tensor<T> targets)
    {
        network.SetTrainingMode(true);

        var output = network.ForwardWithMemory(inputs);
        var outputGrad = ComputeLossGradient(output, targets);
        network.Backpropagate(outputGrad);
        var grads = network.GetParameterGradients();

        network.SetTrainingMode(false);
        return grads.Clone();
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        var sum = _numOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    /// <summary>
    /// Computes the gradient of the loss.
    /// </summary>
    private Tensor<T> ComputeLossGradient(Tensor<T> output, Tensor<T> target)
    {
        var gradData = new Vector<T>(output.Length);
        for (int i = 0; i < output.Length; i++)
        {
            gradData[i] = _numOps.Subtract(output[i], target[i]);
        }
        return new Tensor<T>(output.Shape, gradData);
    }
}
