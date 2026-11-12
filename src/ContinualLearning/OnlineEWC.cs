using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Online Elastic Weight Consolidation (Online EWC) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Online EWC is a memory-efficient variant of EWC that maintains
/// a single running approximation of the Fisher Information Matrix across all tasks, rather
/// than storing separate matrices for each task.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After each task, the Fisher matrix is merged with the running estimate
/// using exponential moving average.</description></item>
/// <item><description>The optimal parameters are also updated as a weighted combination.</description></item>
/// <item><description>This provides O(n) memory complexity instead of O(n × t) for t tasks.</description></item>
/// </list>
///
/// <para><b>Formula:</b></para>
/// <para>F̃ = γ * F_old + F_new</para>
/// <para>θ̃* = (γ * F_old * θ*_old + F_new * θ_new) / (γ * F_old + F_new)</para>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Constant memory regardless of number of tasks.</description></item>
/// <item><description>Suitable for long sequences of tasks.</description></item>
/// <item><description>More graceful degradation over many tasks.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Schwarz, J. et al. "Progress &amp; Compress: A scalable framework for
/// continual learning" (2018). ICML.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Progress & Compress: A Scalable Framework for Continual Learning", "https://arxiv.org/abs/1805.06370", Year = 2018, Authors = "Jonathan Schwarz, Wojciech Czarnecki, Jelena Luketina, Agnieszka Grabska-Barwinska, Yee Whye Teh, Razvan Pascanu, Raia Hadsell")]
[ComponentType(ComponentType.ContinualLearner)]
[PipelineStage(PipelineStage.Training)]
public class OnlineEWC<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private static IEngine Engine => AiDotNetEngine.Current;
    private Vector<T> _fisherDiagonal;         // Running Fisher estimate
    private Vector<T> _optimalParameters;      // Running optimal parameters
    private double _lambda;
    private readonly double _gamma;             // Decay factor for old Fisher
    private int _taskCount;

    /// <summary>
    /// Initializes a new instance of the OnlineEWC class.
    /// </summary>
    /// <param name="lambda">The regularization strength (default: 400).</param>
    /// <param name="gamma">Decay factor for old Fisher information (default: 1.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Lambda controls how strongly to penalize changes to important weights.</description></item>
    /// <item><description>Gamma controls how quickly older task importance decays. γ=1 gives equal
    /// weight to all tasks, γ&lt;1 emphasizes recent tasks more.</description></item>
    /// </list>
    /// </remarks>
    public OnlineEWC(double lambda = 400.0, double gamma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fisherDiagonal = new Vector<T>(0);
        _optimalParameters = new Vector<T>(0);
        _lambda = lambda;
        _gamma = gamma;
        _taskCount = 0;
    }

    /// <inheritdoc />
    public bool AccumulatesAcrossTasks => true;

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets the decay factor for old Fisher information.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the number of tasks processed.
    /// </summary>
    public int TaskCount => _taskCount;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        Guard.NotNull(network);

        // Initialize if this is the first task
        if (_fisherDiagonal.Length == 0)
        {
            int paramCount = checked((int)network.ParameterCount);
            _fisherDiagonal = new Vector<T>(paramCount);
            _optimalParameters = new Vector<T>(paramCount);
        }
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        Guard.NotNull(network);
        Guard.NotNull(taskData.inputs);
        Guard.NotNull(taskData.targets);

        var currentParams = network.GetParameters();

        // Compute Fisher diagonal via tape-based gradients
        var grads = network.ComputeGradients(taskData.inputs, taskData.targets);
        var newFisher = Engine.Multiply(grads, grads);

        if (_taskCount == 0)
        {
            // First task: just store the values
            _fisherDiagonal = newFisher.Clone();
            _optimalParameters = currentParams.Clone();
        }
        else
        {
            // Merge with running estimate
            MergeFisherAndParameters(currentParams, newFisher);
        }

        _taskCount++;
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        Guard.NotNull(network);

        if (_taskCount == 0)
        {
            return _numOps.Zero;
        }

        var currentParams = network.GetParameters();
        int n = currentParams.Length;

        // Online EWC loss: λ/2 * Σ F̃_i * (θ_i - θ̃*_i)²
        var diff = Engine.TensorSubtract(currentParams.AsTensor(), Head(_optimalParameters, n));
        var penalty = Engine.TensorMultiply(Head(_fisherDiagonal, n), Engine.TensorMultiply(diff, diff));
        var loss = Engine.TensorSum(penalty);

        var halfLambda = _numOps.FromDouble(_lambda / 2.0);
        return _numOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        Guard.NotNull(network);
        Guard.NotNull(gradients);

        if (_taskCount == 0)
        {
            return gradients;
        }

        var currentParams = network.GetParameters();
        var lambdaT = _numOps.FromDouble(_lambda);

        int n = gradients.Length;

        // Add Online EWC gradient: λ * F̃_i * (θ_i - θ̃*_i), in place like master
        var diff = Engine.TensorSubtract(Head(currentParams, n), Head(_optimalParameters, n));
        var ewcGrad = Engine.TensorMultiplyScalar(Engine.TensorMultiply(Head(_fisherDiagonal, n), diff), lambdaT);
        Engine.TensorAddInPlace(gradients.AsTensor(), ewcGrad);

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _fisherDiagonal = new Vector<T>(0);
        _optimalParameters = new Vector<T>(0);
        _taskCount = 0;
    }

    /// <summary>
    /// Merges new Fisher information and parameters with the running estimate.
    /// </summary>
    private void MergeFisherAndParameters(Vector<T> newParams, Vector<T> newFisher)
    {
        var gammaT = _numOps.FromDouble(_gamma);
        var epsilon = _numOps.FromDouble(1e-10);
        int n = _fisherDiagonal.Length;
        var fNew = Head(newFisher, n);

        // F̃ = γ * F_old + F_new
        var scaledOldFisher = Engine.TensorMultiplyScalar(_fisherDiagonal.AsTensor(), gammaT);
        var combinedFisher = Engine.TensorAdd(scaledOldFisher, fNew);

        // θ̃* = (γ * F_old * θ*_old + F_new * θ_new) / (γ * F_old + F_new)
        var oldContrib = Engine.TensorMultiply(scaledOldFisher, Head(_optimalParameters, n));
        var newContrib = Engine.TensorMultiply(fNew, Head(newParams, n));
        var totalContrib = Engine.TensorAdd(oldContrib, newContrib);

        // Avoid division by zero
        var denominator = Engine.TensorAddScalar(combinedFisher, epsilon);
        var merged = Engine.TensorDivide(totalContrib, denominator);

        // Write back in place, like master
        merged.AsSpan().CopyTo(_optimalParameters.AsWritableSpan());
        combinedFisher.AsSpan().CopyTo(_fisherDiagonal.AsWritableSpan());
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
        batch.Contiguous().AsSpan().Slice(startIdx, sampleSize).CopyTo(data.AsWritableSpan());

        return new Tensor<T>(newShape, data);
    }

    /// <summary>
    /// Zero-copy rank-1 tensor view over the first <paramref name="count"/> elements of a vector.
    /// </summary>
    private static Tensor<T> Head(Vector<T> vector, int count)
    {
        return count == vector.Length
            ? vector.AsTensor()
            : Tensor<T>.FromMemory(vector.AsWritableMemory().Slice(0, count), new[] { count });
    }

    /// <summary>
    /// Computes the gradient of the log-likelihood.
    /// </summary>
    private Tensor<T> ComputeLogLikelihoodGradient(Tensor<T> output, Tensor<T> target)
    {
        return Engine.TensorSubtract(output, target.Reshape(output._shape));
    }
}
