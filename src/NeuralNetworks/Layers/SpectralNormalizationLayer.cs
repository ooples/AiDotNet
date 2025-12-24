using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a spectral normalization layer that normalizes the weights of a layer by their spectral norm.
/// </summary>
/// <remarks>
/// <para>
/// Spectral normalization is a weight normalization technique that constrains the Lipschitz constant
/// of a neural network layer. It does this by dividing the weight matrix by its largest singular value
/// (spectral norm). This technique is particularly effective for stabilizing GAN training.
/// </para>
/// <para><b>For Beginners:</b> Spectral normalization keeps layer weights from getting too large.
///
/// Key benefits:
/// - Stabilizes GAN training by preventing extreme weight values
/// - Ensures the discriminator doesn't become too powerful too quickly
/// - Helps prevent mode collapse in GANs
/// - Computationally efficient compared to other normalization methods
///
/// How it works:
/// - Computes the largest singular value of the weight matrix
/// - Divides all weights by this value
/// - Keeps weights normalized throughout training
///
/// Reference: Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (2018)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SpectralNormalizationLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The underlying layer whose weights will be normalized.
    /// </summary>
    private readonly ILayer<T> _innerLayer;

    /// <summary>
    /// The left singular vector used for power iteration to compute the spectral norm.
    /// </summary>
    private Tensor<T> _u;

    /// <summary>
    /// The right singular vector used for power iteration.
    /// </summary>
    private Tensor<T> _v;

    /// <summary>
    /// The number of power iterations to perform when computing the spectral norm.
    /// </summary>
    private readonly int _powerIterations;

    /// <summary>
    /// Epsilon value for numerical stability.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// Cached input from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Original weights stored during Forward, to be restored after Backward.
    /// </summary>
    private Vector<T>? _originalParameters;

    /// <summary>
    /// Flag indicating that normalized weights are currently applied.
    /// </summary>
    private bool _normalizedWeightsApplied;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => _innerLayer.SupportsTraining;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpectralNormalizationLayer{T}"/> class.
    /// </summary>
    /// <param name="innerLayer">The layer whose weights will be spectrally normalized.</param>
    /// <param name="powerIterations">The number of power iterations to perform. Default is 1.</param>
    public SpectralNormalizationLayer(ILayer<T> innerLayer, int powerIterations = 1)
        : base(innerLayer.GetInputShape(), innerLayer.GetOutputShape())
    {
        _innerLayer = innerLayer;
        _powerIterations = powerIterations;
        _epsilon = NumOps.FromDouble(1e-12);

        // Initialize u and v vectors for power iteration
        // u has shape [outputSize], v has shape [inputSize]
        var inputShape = innerLayer.GetInputShape();
        var outputShape = innerLayer.GetOutputShape();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);

        // === Vectorized: Initialize u and v using TensorRandomUniformRange (Phase C: New IEngine methods) ===
        _u = Engine.TensorRandomUniformRange<T>([outputSize], NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));
        _v = Engine.TensorRandomUniformRange<T>([inputSize], NumOps.FromDouble(-1.0), NumOps.FromDouble(1.0));

        // Normalize u and v
        NormalizeVector(ref _u);
        NormalizeVector(ref _v);
    }

    /// <summary>
    /// Normalizes a vector tensor in-place using Engine operations.
    /// </summary>
    private void NormalizeVector(ref Tensor<T> vector)
    {
        // === Vectorized L2 normalization using IEngine (Phase B: US-GPU-015) ===
        var squared = Engine.TensorMultiply(vector, vector);
        T sumSquared = Engine.TensorSum(squared);
        T norm = NumOps.Sqrt(sumSquared);
        T normPlusEps = NumOps.Add(norm, _epsilon);

        // Vectorized division by scalar
        vector = Engine.TensorDivideScalar(vector, normPlusEps);
    }

    /// <summary>
    /// Computes the spectral norm using power iteration with vectorized operations.
    /// </summary>
    private T ComputeSpectralNorm(Tensor<T> weights)
    {
        // weights shape: [outputSize, inputSize]
        int outputSize = weights.Shape[0];
        int inputSize = weights.Shape[1];

        var u = _u;
        var v = _v;

        // Power iteration using vectorized matrix operations
        for (int iter = 0; iter < _powerIterations; iter++)
        {
            // v = W^T @ u, then normalize
            // W^T shape: [inputSize, outputSize]
            var wT = Engine.TensorTranspose(weights);

            // Reshape u for matrix multiplication: [outputSize] -> [outputSize, 1]
            var uReshaped = u.Reshape(outputSize, 1);

            // v_new = W^T @ u: [inputSize, outputSize] @ [outputSize, 1] -> [inputSize, 1]
            var vNew = Engine.TensorMatMul(wT, uReshaped);
            v = vNew.Reshape(inputSize);
            NormalizeVector(ref v);

            // u = W @ v, then normalize
            // Reshape v for matrix multiplication: [inputSize] -> [inputSize, 1]
            var vReshaped = v.Reshape(inputSize, 1);

            // u_new = W @ v: [outputSize, inputSize] @ [inputSize, 1] -> [outputSize, 1]
            var uNew = Engine.TensorMatMul(weights, vReshaped);
            u = uNew.Reshape(outputSize);
            NormalizeVector(ref u);
        }

        // Save updated u and v for next iteration
        _u = u;
        _v = v;

        // Compute spectral norm: u^T @ W @ v
        var vReshaped2 = v.Reshape(inputSize, 1);
        var Wv = Engine.TensorMatMul(weights, vReshaped2).Reshape(outputSize);

        // Dot product u^T @ Wv
        var dotProduct = u.Multiply(Wv);
        T spectralNorm = NumOps.Zero;
        for (int i = 0; i < dotProduct.Length; i++)
        {
            spectralNorm = NumOps.Add(spectralNorm, dotProduct[i]);
        }

        return spectralNorm;
    }

    /// <summary>
    /// Performs the forward pass through the layer with spectrally normalized weights.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Get weights from inner layer and reshape to matrix
        var parameters = _innerLayer.GetParameters();
        int outputSize = OutputShape.Aggregate(1, (a, b) => a * b);
        int inputSize = InputShape.Aggregate(1, (a, b) => a * b);
        int expectedWeightCount = outputSize * inputSize;

        // Validate that we have at least enough parameters for the weight matrix
        if (parameters.Length < expectedWeightCount)
        {
            throw new NotSupportedException(
                $"{nameof(SpectralNormalizationLayer<T>)} requires inner layer to have at least " +
                $"{expectedWeightCount} parameters for a {outputSize}x{inputSize} weight matrix. " +
                $"Got {parameters.Length} parameters.");
        }

        // Store original parameters to restore after Backward
        _originalParameters = parameters.Clone();

        // Create weight tensor [outputSize, inputSize]
        var weights = new Tensor<T>([outputSize, inputSize]);
        int paramIdx = 0;
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                weights[new int[] { i, j }] = parameters[paramIdx++];
            }
        }

        // Compute spectral norm
        T spectralNorm = ComputeSpectralNorm(weights);
        T normPlusEps = NumOps.Add(spectralNorm, _epsilon);

        // Normalize weights - element-wise division
        var normalizedWeights = new Tensor<T>([outputSize, inputSize]);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                T weightValue = weights[new int[] { i, j }];
                normalizedWeights[new int[] { i, j }] = NumOps.Divide(weightValue, normPlusEps);
            }
        }

        // Apply normalized weights to inner layer
        var normalizedParams = new Vector<T>(parameters.Length);
        paramIdx = 0;
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                normalizedParams[paramIdx] = normalizedWeights[new int[] { i, j }];
                paramIdx++;
            }
        }
        // Copy any remaining parameters (biases)
        for (; paramIdx < parameters.Length; paramIdx++)
        {
            normalizedParams[paramIdx] = parameters[paramIdx];
        }

        _innerLayer.SetParameters(normalizedParams);
        _normalizedWeightsApplied = true;

        try
        {
            // Forward through inner layer with normalized weights
            _lastOutput = _innerLayer.Forward(input);
            return _lastOutput;
        }
        catch
        {
            // Restore original weights on exception
            RestoreOriginalWeights();
            throw;
        }
    }

    /// <summary>
    /// Restores the original weights after Backward or on exception.
    /// </summary>
    private void RestoreOriginalWeights()
    {
        if (_normalizedWeightsApplied && _originalParameters != null)
        {
            _innerLayer.SetParameters(_originalParameters);
            _normalizedWeightsApplied = false;
            _originalParameters = null;
        }
    }

    /// <summary>
    /// Performs backpropagation through the layer.
    /// </summary>
    /// <remarks>
    /// Backpropagation uses the normalized weights (applied during Forward) to ensure
    /// gradients correspond to the actual weights used in the forward pass. After
    /// computing gradients, the original weights are restored.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        try
        {
            // Backpropagate through inner layer using normalized weights
            // Note: For simplicity, we pass gradients directly through
            // A more accurate implementation would compute the Jacobian of spectral normalization
            return _innerLayer.Backward(outputGradient);
        }
        finally
        {
            // Always restore original weights after Backward
            RestoreOriginalWeights();
        }
    }

    /// <summary>
    /// Updates the parameters of the inner layer.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        _innerLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets the parameters of the inner layer.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        return _innerLayer.GetParameters();
    }

    /// <summary>
    /// Sets the parameters of the inner layer.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        _innerLayer.SetParameters(parameters);
    }

    /// <summary>
    /// Gets the parameter gradients from the inner layer.
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        return _innerLayer.GetParameterGradients();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        RestoreOriginalWeights(); // Ensure weights are restored when resetting
        _innerLayer.ResetState();
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // Spectral normalization doesn't support JIT compilation due to dynamic weight normalization
        throw new NotSupportedException(
            "SpectralNormalizationLayer does not support JIT compilation. " +
            "The spectral norm is computed dynamically during each forward pass.");
    }
}
