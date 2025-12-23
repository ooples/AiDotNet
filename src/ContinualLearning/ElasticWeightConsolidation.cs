using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Elastic Weight Consolidation (EWC) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Elastic Weight Consolidation is like putting rubber bands on
/// important parts of a neural network. When the network learns a new task, these rubber bands
/// pull the weights back toward their original values, preventing the network from forgetting
/// what it learned before.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After learning each task, EWC computes the Fisher Information Matrix (FIM),
/// which measures how important each weight is for that task.</description></item>
/// <item><description>When learning a new task, EWC adds a penalty to the loss function that
/// grows larger when important weights change from their optimal values.</description></item>
/// <item><description>The penalty is: λ/2 * Σᵢ Fᵢ(θᵢ - θ*ᵢ)² where F is the Fisher information,
/// θ is the current weight, θ* is the optimal weight for previous tasks, and λ controls the
/// strength of the penalty.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
/// networks" (2017). PNAS.</para>
/// </remarks>
public class ElasticWeightConsolidation<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<Vector<T>> _fisherDiagonals;
    private readonly List<Vector<T>> _optimalParameters;
    private double _lambda;

    /// <summary>
    /// Initializes a new instance of the ElasticWeightConsolidation class.
    /// </summary>
    /// <param name="lambda">The regularization strength (default: 400).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lambda controls the trade-off between learning new tasks
    /// and remembering old ones:</para>
    /// <list type="bullet">
    /// <item><description>Higher lambda (e.g., 1000-5000): Strong protection of old knowledge,
    /// but may struggle to learn new tasks.</description></item>
    /// <item><description>Lower lambda (e.g., 100-400): Easier to learn new tasks,
    /// but more forgetting of old tasks.</description></item>
    /// </list>
    /// </remarks>
    public ElasticWeightConsolidation(double lambda = 400.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fisherDiagonals = [];
        _optimalParameters = [];
        _lambda = lambda;
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        // EWC doesn't need to do anything else before a task starts.
        // All the work happens in AfterTask when we compute the Fisher information.
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = taskData.inputs ?? throw new ArgumentNullException(nameof(taskData));
        _ = taskData.targets ?? throw new ArgumentNullException(nameof(taskData));

        // Store the optimal parameters for this task
        var currentParams = network.GetParameters();
        _optimalParameters.Add(currentParams.Clone());

        // Compute the diagonal of the Fisher Information Matrix
        var fisherDiag = ComputeFisherDiagonal(network, taskData.inputs, taskData.targets);
        _fisherDiagonals.Add(fisherDiag);
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        if (_fisherDiagonals.Count == 0)
        {
            return _numOps.Zero;
        }

        var currentParams = network.GetParameters();
        var loss = _numOps.Zero;

        // Sum up the EWC penalty for all previous tasks
        for (int task = 0; task < _fisherDiagonals.Count; task++)
        {
            var fisher = _fisherDiagonals[task];
            var optimal = _optimalParameters[task];

            for (int i = 0; i < currentParams.Length; i++)
            {
                var diff = _numOps.Subtract(currentParams[i], optimal[i]);
                var squaredDiff = _numOps.Multiply(diff, diff);
                var penalty = _numOps.Multiply(fisher[i], squaredDiff);
                loss = _numOps.Add(loss, penalty);
            }
        }

        // Multiply by lambda/2
        var halfLambda = _numOps.FromDouble(_lambda / 2.0);
        return _numOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        // EWC uses loss-based regularization, not gradient modification.
        // The gradients from the EWC loss are added during backpropagation.
        // However, we can compute the EWC gradient here for convenience.

        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = gradients ?? throw new ArgumentNullException(nameof(gradients));

        if (_fisherDiagonals.Count == 0)
        {
            return gradients;
        }

        var currentParams = network.GetParameters();
        var ewcGrad = new Vector<T>(gradients.Length);

        // Compute gradient of EWC loss: λ * Σ_tasks F_i * (θ_i - θ*_i)
        for (int task = 0; task < _fisherDiagonals.Count; task++)
        {
            var fisher = _fisherDiagonals[task];
            var optimal = _optimalParameters[task];

            for (int i = 0; i < currentParams.Length; i++)
            {
                var diff = _numOps.Subtract(currentParams[i], optimal[i]);
                var grad = _numOps.Multiply(fisher[i], diff);
                ewcGrad[i] = _numOps.Add(ewcGrad[i], grad);
            }
        }

        // Scale by lambda and add to original gradients
        var lambdaT = _numOps.FromDouble(_lambda);
        for (int i = 0; i < gradients.Length; i++)
        {
            var scaledEwcGrad = _numOps.Multiply(lambdaT, ewcGrad[i]);
            gradients[i] = _numOps.Add(gradients[i], scaledEwcGrad);
        }

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _fisherDiagonals.Clear();
        _optimalParameters.Clear();
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

        // Accumulate squared gradients over the dataset
        for (int i = 0; i < batchSize; i++)
        {
            // Get single sample (simplified - assumes batch dimension is first)
            var singleInput = ExtractSample(inputs, i);
            var singleTarget = ExtractSample(targets, i);

            // Forward pass with memory
            var output = network.ForwardWithMemory(singleInput);

            // Compute gradient with respect to log-likelihood
            // For classification, this is typically the gradient of cross-entropy
            var outputGrad = ComputeLogLikelihoodGradient(output, singleTarget);
            network.Backpropagate(outputGrad);

            // Get parameter gradients and square them
            var grads = network.GetParameterGradients();
            for (int j = 0; j < paramCount; j++)
            {
                var squaredGrad = _numOps.Multiply(grads[j], grads[j]);
                fisherDiag[j] = _numOps.Add(fisherDiag[j], squaredGrad);
            }
        }

        // Average over the batch
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
    /// Computes the gradient of the log-likelihood (negative cross-entropy gradient).
    /// </summary>
    private Tensor<T> ComputeLogLikelihoodGradient(Tensor<T> output, Tensor<T> target)
    {
        // For softmax cross-entropy, the gradient is (output - target)
        var gradData = new Vector<T>(output.Length);
        for (int i = 0; i < output.Length; i++)
        {
            gradData[i] = _numOps.Subtract(output[i], target[i]);
        }

        return new Tensor<T>(output.Shape, gradData);
    }
}
