using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
public class MemoryAwareSynapses<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
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
        _ = network ?? throw new ArgumentNullException(nameof(network));

        // Initialize if first task
        if (_omega.Length == 0)
        {
            var paramCount = network.ParameterCount;
            _omega = new Vector<T>(paramCount);
            _optimalParameters = new Vector<T>(paramCount);
        }
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = taskData.inputs ?? throw new ArgumentNullException(nameof(taskData));

        // Compute importance using output sensitivity (unsupervised)
        var newOmega = ComputeOutputSensitivity(network, taskData.inputs);

        // Accumulate importance
        for (int i = 0; i < _omega.Length; i++)
        {
            _omega[i] = _numOps.Add(_omega[i], newOmega[i]);
        }

        // Store current parameters as optimal
        _optimalParameters = network.GetParameters().Clone();
        _taskCount++;
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        if (_taskCount == 0)
        {
            return _numOps.Zero;
        }

        var currentParams = network.GetParameters();
        var loss = _numOps.Zero;

        // MAS loss: λ/2 * Σ Ω_i * (θ_i - θ*_i)²
        for (int i = 0; i < currentParams.Length; i++)
        {
            var diff = _numOps.Subtract(currentParams[i], _optimalParameters[i]);
            var squaredDiff = _numOps.Multiply(diff, diff);
            var penalty = _numOps.Multiply(_omega[i], squaredDiff);
            loss = _numOps.Add(loss, penalty);
        }

        var halfLambda = _numOps.FromDouble(_lambda / 2.0);
        return _numOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = gradients ?? throw new ArgumentNullException(nameof(gradients));

        if (_taskCount == 0)
        {
            return gradients;
        }

        var currentParams = network.GetParameters();
        var lambdaT = _numOps.FromDouble(_lambda);

        // Add MAS gradient: λ * Ω_i * (θ_i - θ*_i)
        for (int i = 0; i < gradients.Length; i++)
        {
            var diff = _numOps.Subtract(currentParams[i], _optimalParameters[i]);
            var masGrad = _numOps.Multiply(_omega[i], diff);
            masGrad = _numOps.Multiply(lambdaT, masGrad);
            gradients[i] = _numOps.Add(gradients[i], masGrad);
        }

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
    /// Computes the importance of each parameter based on output sensitivity.
    /// </summary>
    /// <remarks>
    /// This is the key difference from EWC: we compute gradients of the output norm
    /// with respect to parameters, not gradients of the loss.
    /// </remarks>
    private Vector<T> ComputeOutputSensitivity(INeuralNetwork<T> network, Tensor<T> inputs)
    {
        var paramCount = network.ParameterCount;
        var omega = new Vector<T>(paramCount);
        var batchSize = inputs.Shape[0];

        network.SetTrainingMode(true);

        for (int i = 0; i < batchSize; i++)
        {
            var singleInput = ExtractSample(inputs, i);

            // Forward pass
            var output = network.ForwardWithMemory(singleInput);

            // Compute gradient of squared L2 norm of output with respect to parameters
            // d||F(x)||²/dθ = 2 * F(x) * dF(x)/dθ
            var outputGrad = ComputeOutputNormGradient(output);
            network.Backpropagate(outputGrad);

            // Get parameter gradients and accumulate absolute values
            var grads = network.GetParameterGradients();
            for (int j = 0; j < paramCount; j++)
            {
                var absGrad = _numOps.Abs(grads[j]);
                omega[j] = _numOps.Add(omega[j], absGrad);
            }
        }

        // Average over batch
        var batchSizeT = _numOps.FromDouble(batchSize);
        for (int j = 0; j < paramCount; j++)
        {
            omega[j] = _numOps.Divide(omega[j], batchSizeT);
        }

        network.SetTrainingMode(false);
        return omega;
    }

    /// <summary>
    /// Computes the gradient of the output L2 norm.
    /// </summary>
    private Tensor<T> ComputeOutputNormGradient(Tensor<T> output)
    {
        // Gradient of ||F||² = 2 * F
        var gradData = new Vector<T>(output.Length);
        var two = _numOps.FromDouble(2.0);

        for (int i = 0; i < output.Length; i++)
        {
            gradData[i] = _numOps.Multiply(two, output[i]);
        }

        return new Tensor<T>(output.Shape, gradData);
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
}
