using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;

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
public class OnlineEWC<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
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
        _ = network ?? throw new ArgumentNullException(nameof(network));

        // Initialize if this is the first task
        if (_fisherDiagonal.Length == 0)
        {
            var paramCount = network.ParameterCount;
            _fisherDiagonal = new Vector<T>(paramCount);
            _optimalParameters = new Vector<T>(paramCount);
        }
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = taskData.inputs ?? throw new ArgumentNullException(nameof(taskData));
        _ = taskData.targets ?? throw new ArgumentNullException(nameof(taskData));

        var currentParams = network.GetParameters();
        var newFisher = ComputeFisherDiagonal(network, taskData.inputs, taskData.targets);

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
        _ = network ?? throw new ArgumentNullException(nameof(network));

        if (_taskCount == 0)
        {
            return _numOps.Zero;
        }

        var currentParams = network.GetParameters();
        var loss = _numOps.Zero;

        // Online EWC loss: λ/2 * Σ F̃_i * (θ_i - θ̃*_i)²
        for (int i = 0; i < currentParams.Length; i++)
        {
            var diff = _numOps.Subtract(currentParams[i], _optimalParameters[i]);
            var squaredDiff = _numOps.Multiply(diff, diff);
            var penalty = _numOps.Multiply(_fisherDiagonal[i], squaredDiff);
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

        // Add Online EWC gradient: λ * F̃_i * (θ_i - θ̃*_i)
        for (int i = 0; i < gradients.Length; i++)
        {
            var diff = _numOps.Subtract(currentParams[i], _optimalParameters[i]);
            var ewcGrad = _numOps.Multiply(_fisherDiagonal[i], diff);
            ewcGrad = _numOps.Multiply(lambdaT, ewcGrad);
            gradients[i] = _numOps.Add(gradients[i], ewcGrad);
        }

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

        for (int i = 0; i < _fisherDiagonal.Length; i++)
        {
            // F̃ = γ * F_old + F_new
            var scaledOldFisher = _numOps.Multiply(gammaT, _fisherDiagonal[i]);
            var combinedFisher = _numOps.Add(scaledOldFisher, newFisher[i]);

            // θ̃* = (γ * F_old * θ*_old + F_new * θ_new) / (γ * F_old + F_new)
            var oldContrib = _numOps.Multiply(scaledOldFisher, _optimalParameters[i]);
            var newContrib = _numOps.Multiply(newFisher[i], newParams[i]);
            var totalContrib = _numOps.Add(oldContrib, newContrib);

            // Avoid division by zero
            var epsilon = _numOps.FromDouble(1e-10);
            var denominator = _numOps.Add(combinedFisher, epsilon);

            _optimalParameters[i] = _numOps.Divide(totalContrib, denominator);
            _fisherDiagonal[i] = combinedFisher;
        }
    }

    /// <summary>
    /// Computes the diagonal approximation of the Fisher Information Matrix.
    /// </summary>
    private Vector<T> ComputeFisherDiagonal(INeuralNetwork<T> network, Tensor<T> inputs, Tensor<T> targets)
    {
        var paramCount = network.ParameterCount;
        var fisherDiag = new Vector<T>(paramCount);
        var batchSize = inputs.Shape[0];

        network.SetTrainingMode(true);

        for (int i = 0; i < batchSize; i++)
        {
            var singleInput = ExtractSample(inputs, i);
            var singleTarget = ExtractSample(targets, i);

            var output = network.ForwardWithMemory(singleInput);
            var outputGrad = ComputeLogLikelihoodGradient(output, singleTarget);
            network.Backpropagate(outputGrad);

            var grads = network.GetParameterGradients();
            for (int j = 0; j < paramCount; j++)
            {
                var squaredGrad = _numOps.Multiply(grads[j], grads[j]);
                fisherDiag[j] = _numOps.Add(fisherDiag[j], squaredGrad);
            }
        }

        // Average over batch
        var batchSizeT = _numOps.FromDouble(batchSize);
        for (int j = 0; j < paramCount; j++)
        {
            fisherDiag[j] = _numOps.Divide(fisherDiag[j], batchSizeT);
        }

        network.SetTrainingMode(false);
        return fisherDiag;
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
    /// Computes the gradient of the log-likelihood.
    /// </summary>
    private Tensor<T> ComputeLogLikelihoodGradient(Tensor<T> output, Tensor<T> target)
    {
        var gradData = new Vector<T>(output.Length);
        for (int i = 0; i < output.Length; i++)
        {
            gradData[i] = _numOps.Subtract(output[i], target[i]);
        }
        return new Tensor<T>(output.Shape, gradData);
    }
}
