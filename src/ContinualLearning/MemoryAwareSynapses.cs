using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Memory Aware Synapses (MAS) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MAS is similar to EWC but estimates weight importance in an
/// unsupervised way using the sensitivity of the network output to each weight. This means
/// it doesn't need task labels to compute importance.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After each task, compute how much the output changes when each weight
/// is perturbed (gradient of output with respect to weights).</description></item>
/// <item><description>Weights that cause large output changes are important and should be protected.</description></item>
/// <item><description>When learning new tasks, penalize changes to important weights.</description></item>
/// </list>
///
/// <para><b>Key Formula:</b></para>
/// <para>Ω_i = 1/N × Σ_n |∂F(x_n)/∂θ_i|</para>
/// <para>where F is the network output and θ_i is weight i.</para>
///
/// <para><b>Advantages over EWC:</b></para>
/// <list type="bullet">
/// <item><description>Unsupervised - doesn't need task labels.</description></item>
/// <item><description>Can be computed on any unlabeled data.</description></item>
/// <item><description>Simpler computation than Fisher Information.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., and Tuytelaars, T.
/// "Memory Aware Synapses: Learning what (not) to forget" (2018). ECCV.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Memory Aware Synapses: Learning What (Not) to Forget", "https://arxiv.org/abs/1711.09601", Year = 2018, Authors = "Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, Tinne Tuytelaars")]
[ComponentType(ComponentType.ContinualLearner)]
[PipelineStage(PipelineStage.Training)]
public class MemoryAwareSynapses<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private static IEngine Engine => AiDotNetEngine.Current;
    private Vector<T> _omega;                    // Accumulated importance
    private Vector<T> _optimalParameters;        // Parameters at end of last task
    private double _lambda;
    private int _taskCount;

    /// <summary>
    /// Initializes a new instance of the MemoryAwareSynapses class.
    /// </summary>
    /// <param name="lambda">The regularization strength (default: 1.0).</param>
    public MemoryAwareSynapses(double lambda = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega = new Vector<T>(0);
        _optimalParameters = new Vector<T>(0);
        _lambda = lambda;
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
    /// Gets the number of tasks processed.
    /// </summary>
    public int TaskCount => _taskCount;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        Guard.NotNull(network);

        // Initialize if first task
        if (_omega.Length == 0)
        {
            int paramCount = checked((int)network.ParameterCount);
            _omega = new Vector<T>(paramCount);
            _optimalParameters = new Vector<T>(paramCount);
        }
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        Guard.NotNull(network);
        Guard.NotNull(taskData.inputs);

        // Compute importance using output sensitivity via tape-based gradients
        var grads = network.ComputeGradients(taskData.inputs, taskData.targets);
        var newOmega = Engine.Abs(grads); // Importance = |gradient|

        // Accumulate importance (in place)
        Engine.TensorAddInPlace(_omega.AsTensor(), Head(newOmega, _omega.Length));

        // Store current parameters as optimal
        _optimalParameters = network.GetParameters().Clone();
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

        // MAS loss: λ/2 * Σ Ω_i * (θ_i - θ*_i)²
        var diff = Engine.TensorSubtract(currentParams.AsTensor(), Head(_optimalParameters, n));
        var penalty = Engine.TensorMultiply(Head(_omega, n), Engine.TensorMultiply(diff, diff));
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

        // Add MAS gradient: λ * Ω_i * (θ_i - θ*_i), in place like master
        var diff = Engine.TensorSubtract(Head(currentParams, n), Head(_optimalParameters, n));
        var masGrad = Engine.TensorMultiplyScalar(Engine.TensorMultiply(Head(_omega, n), diff), lambdaT);
        Engine.TensorAddInPlace(gradients.AsTensor(), masGrad);

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _omega = new Vector<T>(0);
        _optimalParameters = new Vector<T>(0);
        _taskCount = 0;
    }

    /// <summary>
    /// Computes the gradient of the output L2 norm.
    /// </summary>
    private Tensor<T> ComputeOutputNormGradient(Tensor<T> output)
    {
        // Gradient of ||F||² = 2 * F
        var two = _numOps.FromDouble(2.0);
        return Engine.TensorMultiplyScalar(output, two);
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
}
