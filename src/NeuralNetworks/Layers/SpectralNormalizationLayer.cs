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
    /// The vector used for power iteration to compute the spectral norm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector is used in the power iteration method to approximate the largest
    /// singular value of the weight matrix. It's updated during each forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> A helper vector used in the normalization calculation.
    ///
    /// - Updated each time the layer is used
    /// - Helps efficiently compute the spectral norm
    /// - Doesn't need to be perfect, approximation works well
    /// </para>
    /// </remarks>
    private Vector<T> _u;

    /// <summary>
    /// The number of power iterations to perform when computing the spectral norm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More iterations give a more accurate estimate of the spectral norm, but require more computation.
    /// Typically, 1 iteration is sufficient for training, as the vector u is carried across iterations.
    /// </para>
    /// <para><b>For Beginners:</b> How many times to refine the normalization calculation.
    ///
    /// - Default: 1 iteration (fast and usually sufficient)
    /// - Higher values: more accurate but slower
    /// - For training, 1 iteration works well because we do it repeatedly
    /// </para>
    /// </remarks>
    private readonly int _powerIterations;

    /// <summary>
    /// Epsilon value for numerical stability.
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpectralNormalizationLayer{T}"/> class.
    /// </summary>
    /// <param name="innerLayer">The layer whose weights will be spectrally normalized.</param>
    /// <param name="powerIterations">The number of power iterations to perform. Default is 1.</param>
    /// <remarks>
    /// <para>
    /// This constructor wraps an existing layer with spectral normalization. The inner layer's
    /// weights will be normalized by their spectral norm before each forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This wraps another layer to add spectral normalization.
    ///
    /// Example usage:
    /// - Create a convolutional layer
    /// - Wrap it with SpectralNormalizationLayer
    /// - The layer now has normalized weights automatically
    ///
    /// Parameters:
    /// - innerLayer: the layer to normalize (e.g., Conv2D, Dense)
    /// - powerIterations: how many refinement steps (1 is usually fine)
    /// </para>
    /// </remarks>
    public SpectralNormalizationLayer(ILayer<T> innerLayer, int powerIterations = 1)
        : base(innerLayer.InputSize, innerLayer.OutputSize)
    {
        _innerLayer = innerLayer;
        _powerIterations = powerIterations;
        _epsilon = NumOps.FromDouble(1e-12);

        // Initialize u vector randomly
        var random = new Random();
        _u = new Vector<T>(innerLayer.OutputSize);

        for (int i = 0; i < _u.Length; i++)
        {
            _u[i] = NumOps.FromDouble(random.NextDouble() * 2 - 1);
        }

        // Normalize u
        T norm = _u.L2Norm();
        for (int i = 0; i < _u.Length; i++)
        {
            _u[i] = NumOps.Divide(_u[i], NumOps.Add(norm, _epsilon));
        }
    }

    /// <summary>
    /// Computes the spectral norm of the weight matrix using power iteration.
    /// </summary>
    /// <param name="weights">The weight matrix to normalize.</param>
    /// <returns>The spectral norm (largest singular value) of the weight matrix.</returns>
    /// <remarks>
    /// <para>
    /// The power iteration method is an efficient way to approximate the largest singular value
    /// of a matrix. It iteratively updates vectors u and v to converge to the largest singular
    /// value and its corresponding singular vectors.
    /// </para>
    /// <para><b>For Beginners:</b> Calculates the "spectral norm" of the weights.
    ///
    /// What's the spectral norm:
    /// - It's the largest value that describes how much the weights can "stretch" the input
    /// - Mathematically, it's the largest singular value of the weight matrix
    /// - We want to keep this value under control for stable training
    ///
    /// The power iteration method:
    /// - Starts with a random vector
    /// - Repeatedly multiplies by the weight matrix and its transpose
    /// - Converges to the largest singular value
    /// - Very efficient, especially with just 1 iteration
    /// </para>
    /// </remarks>
    private T ComputeSpectralNorm(Matrix<T> weights)
    {
        Vector<T> u = _u;
        Vector<T> v = Vector<T>.Empty();

        // Perform power iterations
        for (int i = 0; i < _powerIterations; i++)
        {
            // v = W^T @ u
            v = weights.TransposeMultiply(u);

            // Normalize v
            T vNorm = v.L2Norm();
            for (int j = 0; j < v.Length; j++)
            {
                v[j] = NumOps.Divide(v[j], NumOps.Add(vNorm, _epsilon));
            }

            // u = W @ v
            u = weights.Multiply(v);

            // Normalize u
            T uNorm = u.L2Norm();
            for (int j = 0; j < u.Length; j++)
            {
                u[j] = NumOps.Divide(u[j], NumOps.Add(uNorm, _epsilon));
            }
        }

        // Update _u for next iteration
        _u = u;

        // Compute spectral norm: u^T @ W @ v
        Vector<T> Wv = weights.Multiply(v);
        T spectralNorm = NumOps.Zero;

        for (int i = 0; i < u.Length; i++)
        {
            spectralNorm = NumOps.Add(spectralNorm, NumOps.Multiply(u[i], Wv[i]));
        }

        return spectralNorm;
    }

    /// <summary>
    /// Performs a forward pass through the layer with spectrally normalized weights.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>The output vector after applying the spectrally normalized layer.</returns>
    /// <remarks>
    /// <para>
    /// This method normalizes the inner layer's weights by their spectral norm before
    /// performing the forward pass. This ensures the Lipschitz constant is constrained.
    /// </para>
    /// <para><b>For Beginners:</b> Processes input through the layer with normalized weights.
    ///
    /// Steps:
    /// 1. Get the current weights from the inner layer
    /// 2. Calculate the spectral norm
    /// 3. Divide all weights by the spectral norm
    /// 4. Apply the normalized weights to the input
    /// 5. Return the result
    ///
    /// This happens automatically every time data passes through the layer.
    /// </para>
    /// </remarks>
    public override Vector<T> Forward(Vector<T> input)
    {
        // Get weights from inner layer
        var weights = GetWeightMatrix();

        // Compute spectral norm
        T spectralNorm = ComputeSpectralNorm(weights);

        // Normalize weights
        var normalizedWeights = new Matrix<T>(weights.Rows, weights.Cols);
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Cols; j++)
            {
                normalizedWeights[i, j] = NumOps.Divide(
                    weights[i, j],
                    NumOps.Add(spectralNorm, _epsilon)
                );
            }
        }

        // Apply normalized weights to the inner layer (temporarily)
        ApplyWeightMatrix(normalizedWeights);

        // Forward pass through inner layer
        var output = _innerLayer.Forward(input);

        // Restore original weights (will be updated during backprop)
        ApplyWeightMatrix(weights);

        return output;
    }

    /// <summary>
    /// Performs a forward pass using a tensor input.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return _innerLayer.Forward(input);
    }

    /// <summary>
    /// Performs backpropagation through the layer.
    /// </summary>
    /// <param name="gradientOutput">The gradient from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// Backpropagation through spectral normalization requires computing the derivative
    /// of the normalization operation. However, in practice, many implementations simply
    /// backpropagate through the inner layer without explicitly computing this derivative,
    /// as the approximation works well and simplifies implementation.
    /// </para>
    /// <para><b>For Beginners:</b> Passes gradients back through the layer.
    ///
    /// During training:
    /// - Forward pass normalizes the weights
    /// - Backward pass updates the weights based on the error
    /// - The spectral normalization is recomputed in the next forward pass
    ///
    /// This simple approach works well in practice.
    /// </para>
    /// </remarks>
    public override Vector<T> Backward(Vector<T> gradientOutput)
    {
        return _innerLayer.Backward(gradientOutput);
    }

    /// <summary>
    /// Performs backpropagation using a tensor gradient.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> gradientOutput)
    {
        return _innerLayer.Backward(gradientOutput);
    }

    /// <summary>
    /// Updates the parameters of the inner layer.
    /// </summary>
    /// <param name="learningRate">The learning rate.</param>
    /// <remarks>
    /// <para>
    /// The weight updates are applied to the inner layer. The spectral normalization
    /// will be recomputed in the next forward pass with the updated weights.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(double learningRate)
    {
        _innerLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets the weight matrix from the inner layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extracts the weight matrix from the inner layer. The implementation
    /// depends on the type of inner layer (Dense, Convolutional, etc.).
    /// </para>
    /// </remarks>
    private Matrix<T> GetWeightMatrix()
    {
        var parameters = _innerLayer.GetParameters();

        // Assuming the inner layer's parameters can be reshaped into a matrix
        // The exact implementation depends on the layer type
        int rows = OutputSize;
        int cols = InputSize;

        var weights = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows && i * cols < parameters.Length; i++)
        {
            for (int j = 0; j < cols && i * cols + j < parameters.Length; j++)
            {
                weights[i, j] = parameters[i * cols + j];
            }
        }

        return weights;
    }

    /// <summary>
    /// Applies a weight matrix to the inner layer.
    /// </summary>
    private void ApplyWeightMatrix(Matrix<T> weights)
    {
        var parameters = new Vector<T>(weights.Rows * weights.Cols);

        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Cols; j++)
            {
                parameters[i * weights.Cols + j] = weights[i, j];
            }
        }

        _innerLayer.SetParameters(parameters);
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
    /// Gets metadata about this layer.
    /// </summary>
    public override LayerMetadata GetLayerMetadata()
    {
        return new LayerMetadata
        {
            LayerType = "SpectralNormalization",
            InputSize = InputSize,
            OutputSize = OutputSize,
            Parameters = new Dictionary<string, object>
            {
                { "PowerIterations", _powerIterations },
                { "InnerLayerType", _innerLayer.GetType().Name }
            }
        };
    }
}
