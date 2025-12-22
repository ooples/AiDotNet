using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Gradient Episodic Memory (GEM) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Gradient Episodic Memory is like having a safety net that
/// catches you if you're about to forget something important. It stores examples from
/// previous tasks and checks each gradient update to make sure it won't hurt performance
/// on those stored examples.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After learning each task, GEM stores a subset of examples (episodic memory)
/// from that task.</description></item>
/// <item><description>When computing gradients for a new task, GEM first computes gradients
/// on the stored examples from previous tasks.</description></item>
/// <item><description>If the new gradient would increase the loss on previous tasks (negative
/// dot product with previous gradients), it projects the gradient to the closest one that
/// doesn't interfere.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Lopez-Paz and Ranzato, "Gradient Episodic Memory for Continual
/// Learning" (2017). NeurIPS.</para>
/// </remarks>
public class GradientEpisodicMemory<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<(Tensor<T> inputs, Tensor<T> targets)> _episodicMemory;
    private readonly List<Vector<T>> _referenceGradients;
    private readonly int _memorySize;
    private double _lambda;
    private double _margin;

    /// <summary>
    /// Initializes a new instance of the GradientEpisodicMemory class.
    /// </summary>
    /// <param name="memorySize">Maximum samples to store per task (default: 256).</param>
    /// <param name="margin">Margin for gradient constraint (default: 0.5).</param>
    /// <param name="lambda">Regularization strength for compatibility (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Memory size: How many examples to remember from each task.
    /// More examples = better protection but higher memory cost.</description></item>
    /// <item><description>Margin: How much "safety buffer" to keep. Higher margin means
    /// gradients are more constrained, reducing forgetting but potentially slowing learning.</description></item>
    /// </list>
    /// </remarks>
    public GradientEpisodicMemory(int memorySize = 256, double margin = 0.5, double lambda = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _episodicMemory = [];
        _referenceGradients = [];
        _memorySize = memorySize;
        _margin = margin;
        _lambda = lambda;
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets or sets the margin for the gradient constraint.
    /// </summary>
    public double Margin
    {
        get => _margin;
        set => _margin = value;
    }

    /// <summary>
    /// Gets the number of tasks currently stored in memory.
    /// </summary>
    public int TaskCount => _episodicMemory.Count;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        // Update reference gradients for all stored tasks before starting new task
        UpdateReferenceGradients(network);
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = taskData.inputs ?? throw new ArgumentNullException(nameof(taskData));
        _ = taskData.targets ?? throw new ArgumentNullException(nameof(taskData));

        // Sample and store examples from the completed task
        var sampledData = SampleMemory(taskData.inputs, taskData.targets);
        _episodicMemory.Add(sampledData);

        // Compute and store reference gradients for this task
        var refGrad = ComputeReferenceGradient(network, sampledData.inputs, sampledData.targets);
        _referenceGradients.Add(refGrad);
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        // GEM primarily works through gradient modification, not loss-based regularization.
        // However, we can optionally compute a surrogate loss.
        return _numOps.Zero;
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = gradients ?? throw new ArgumentNullException(nameof(gradients));

        if (_referenceGradients.Count == 0)
        {
            return gradients;
        }

        // Check if gradient violates any constraint
        var violations = new List<int>();
        for (int i = 0; i < _referenceGradients.Count; i++)
        {
            var dotProduct = DotProduct(gradients, _referenceGradients[i]);
            var marginT = _numOps.FromDouble(-_margin);
            if (_numOps.LessThan(dotProduct, marginT))
            {
                violations.Add(i);
            }
        }

        if (violations.Count == 0)
        {
            return gradients;
        }

        // Project gradient to satisfy constraints
        return ProjectGradient(gradients, violations);
    }

    /// <inheritdoc />
    public void Reset()
    {
        _episodicMemory.Clear();
        _referenceGradients.Clear();
    }

    /// <summary>
    /// Samples examples from task data to store in episodic memory.
    /// </summary>
    private (Tensor<T> inputs, Tensor<T> targets) SampleMemory(Tensor<T> inputs, Tensor<T> targets)
    {
        var batchSize = inputs.Shape[0];
        var samplesToKeep = Math.Min(_memorySize, batchSize);

        if (samplesToKeep >= batchSize)
        {
            return (inputs.Clone(), targets.Clone());
        }

        // Random sampling
        var indices = Enumerable.Range(0, batchSize)
            .OrderBy(_ => RandomHelper.Shared.Next())
            .Take(samplesToKeep)
            .ToArray();

        var sampledInputs = SampleTensor(inputs, indices);
        var sampledTargets = SampleTensor(targets, indices);

        return (sampledInputs, sampledTargets);
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
    /// Computes the reference gradient for stored examples.
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
    /// Updates reference gradients for all stored tasks.
    /// </summary>
    private void UpdateReferenceGradients(INeuralNetwork<T> network)
    {
        for (int i = 0; i < _episodicMemory.Count; i++)
        {
            var (inputs, targets) = _episodicMemory[i];
            _referenceGradients[i] = ComputeReferenceGradient(network, inputs, targets);
        }
    }

    /// <summary>
    /// Projects the gradient to satisfy constraints using a simplified QP solver.
    /// </summary>
    private Vector<T> ProjectGradient(Vector<T> gradient, List<int> violations)
    {
        // Simplified projection: subtract component along violating reference gradients
        var projected = gradient.Clone();

        foreach (var taskIdx in violations)
        {
            var refGrad = _referenceGradients[taskIdx];
            var dot = DotProduct(projected, refGrad);
            var refNormSq = DotProduct(refGrad, refGrad);

            // Project out the component that violates the constraint
            if (_numOps.GreaterThan(refNormSq, _numOps.Zero))
            {
                var scale = _numOps.Divide(dot, refNormSq);
                for (int i = 0; i < projected.Length; i++)
                {
                    var adjustment = _numOps.Multiply(scale, refGrad[i]);
                    projected[i] = _numOps.Subtract(projected[i], adjustment);
                }
            }
        }

        return projected;
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
    /// Computes the gradient of the loss with respect to output.
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
