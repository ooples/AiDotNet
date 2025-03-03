namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements batch normalization for neural networks, which normalizes the inputs across a mini-batch.
/// </summary>
/// <remarks>
/// Batch normalization helps stabilize and accelerate training by normalizing layer inputs.
/// It works by normalizing each feature to have zero mean and unit variance across the batch,
/// then applying learnable scale (gamma) and shift (beta) parameters.
/// 
/// Benefits include:
/// - Faster training convergence
/// - Reduced sensitivity to weight initialization
/// - Ability to use higher learning rates
/// - Acts as a form of regularization
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
public class BatchNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private readonly T _momentum;
    private Vector<T> _gamma;
    private Vector<T> _beta;
    private Vector<T> _runningMean;
    private Vector<T> _runningVariance;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastNormalized;
    private Vector<T>? _lastMean;
    private Vector<T>? _lastVariance;
    private Vector<T>? _gammaGradient;
    private Vector<T>? _betaGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training mode.
    /// </summary>
    /// <remarks>
    /// Batch normalization behaves differently during training versus inference:
    /// - During training: Uses statistics from the current batch
    /// - During inference: Uses running statistics collected during training
    /// 
    /// This property always returns true because the layer needs to track its training state.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the BatchNormalizationLayer class.
    /// </summary>
    /// <param name="featureSize">The number of features (neurons) to normalize.</param>
    /// <param name="epsilon">A small constant added to the variance for numerical stability (default: 1e-5).</param>
    /// <param name="momentum">The momentum for updating running statistics (default: 0.9).</param>
    /// <remarks>
    /// The epsilon parameter prevents division by zero when normalizing features with very small variance.
    /// 
    /// The momentum parameter controls how much the running statistics are updated during training:
    /// - Values closer to 1.0 give more weight to past batches (slower updates)
    /// - Values closer to 0.0 give more weight to the current batch (faster updates)
    /// 
    /// A typical value is 0.9, which means each new batch contributes about 10% to the running statistics.
    /// </remarks>
    public BatchNormalizationLayer(int featureSize, double epsilon = 1e-5, double momentum = 0.9)
        : base([featureSize], [featureSize])
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _momentum = NumOps.FromDouble(momentum);
        _gamma = Vector<T>.CreateDefault(featureSize, NumOps.One);
        _beta = new Vector<T>(featureSize);
        _runningMean = new Vector<T>(featureSize);
        _runningVariance = Vector<T>.CreateDefault(featureSize, NumOps.One);
    }

    /// <summary>
    /// Performs the forward pass of batch normalization.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, featureSize].</param>
    /// <returns>The normalized, scaled, and shifted output tensor.</returns>
    /// <remarks>
    /// The forward pass performs these steps:
    /// 1. If in training mode:
    ///    - Compute mean and variance of the current batch
    ///    - Update running statistics for inference
    ///    - Normalize using batch statistics
    /// 2. If in inference mode:
    ///    - Normalize using running statistics collected during training
    /// 3. Apply scale (gamma) and shift (beta) parameters
    /// 
    /// The normalization formula is: y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];

        var output = new Tensor<T>(input.Shape);

        if (IsTrainingMode)
        {
            _lastMean = ComputeMean(input);
            _lastVariance = ComputeVariance(input, _lastMean);

            // Update running statistics
            _runningMean = UpdateRunningStatistic(_runningMean, _lastMean);
            _runningVariance = UpdateRunningStatistic(_runningVariance, _lastVariance);

            _lastNormalized = Normalize(input, _lastMean, _lastVariance);
        }
        else
        {
            _lastNormalized = Normalize(input, _runningMean, _runningVariance);
        }

        // Scale and shift
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                output[i, j] = NumOps.Add(NumOps.Multiply(_lastNormalized[i, j], _gamma[j]), _beta[j]);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of batch normalization.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// The backward pass computes three types of gradients:
    /// 1. Gradients for the input (to pass to previous layers)
    /// 2. Gradients for gamma (scale parameter)
    /// 3. Gradients for beta (shift parameter)
    /// 
    /// This is a complex calculation that accounts for how each input affects:
    /// - The normalized value directly
    /// - The batch mean
    /// - The batch variance
    /// 
    /// The implementation follows the chain rule of calculus to properly backpropagate
    /// through all operations in the forward pass.
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalized == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int featureSize = _lastInput.Shape[1];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _gammaGradient = new Vector<T>(featureSize);
        _betaGradient = new Vector<T>(featureSize);

        var varianceEpsilon = _lastVariance.Add(_epsilon);
        var invStd = varianceEpsilon.Transform(NumOps.Sqrt).Transform(x => MathHelper.Reciprocal(x));

        for (int j = 0; j < featureSize; j++)
        {
            T sumDy = NumOps.Zero;
            T sumDyXmu = NumOps.Zero;

            for (int i = 0; i < batchSize; i++)
            {
                T dy = outputGradient[i, j];
                T xmu = _lastNormalized[i, j];

                sumDy = NumOps.Add(sumDy, dy);
                sumDyXmu = NumOps.Add(sumDyXmu, NumOps.Multiply(dy, xmu));

                _gammaGradient[j] = NumOps.Add(_gammaGradient[j], NumOps.Multiply(dy, xmu));
                _betaGradient[j] = NumOps.Add(_betaGradient[j], dy);
            }

            T invN = NumOps.FromDouble(1.0 / batchSize);
            T invVar = invStd[j];

            for (int i = 0; i < batchSize; i++)
            {
                T xmu = _lastNormalized[i, j];
                T dy = outputGradient[i, j];

                inputGradient[i, j] = NumOps.Multiply(
                    _gamma[j],
                    NumOps.Multiply(
                        invN,
                        NumOps.Multiply(
                            invVar,
                            NumOps.Subtract(
                                NumOps.Multiply(NumOps.FromDouble(batchSize), dy),
                                NumOps.Add(sumDy, NumOps.Multiply(xmu, sumDyXmu))
                            )
                        )
                    )
                );
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters using the computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// This method updates the gamma (scale) and beta (shift) parameters using gradient descent:
    /// - gamma = gamma - learningRate * gammaGradient
    /// - beta = beta - learningRate * betaGradient
    /// 
    /// The gradients are computed during the backward pass and represent how much
    /// each parameter should change to reduce the loss function.
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = _gamma.Subtract(_gammaGradient.Multiply(learningRate));
        _beta = _beta.Subtract(_betaGradient.Multiply(learningRate));
    }

    private Vector<T> ComputeMean(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];
        var mean = new Vector<T>(featureSize);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                mean[j] = NumOps.Add(mean[j], input[i, j]);
            }
        }

        return mean.Divide(NumOps.FromDouble(batchSize));
    }

    private Vector<T> ComputeVariance(Tensor<T> input, Vector<T> mean)
    {
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];
        var variance = new Vector<T>(featureSize);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                T diff = NumOps.Subtract(input[i, j], mean[j]);
                variance[j] = NumOps.Add(variance[j], NumOps.Multiply(diff, diff));
            }
        }

        return variance.Divide(NumOps.FromDouble(batchSize));
    }

    private Tensor<T> Normalize(Tensor<T> input, Vector<T> mean, Vector<T> variance)
    {
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];
        var normalized = new Tensor<T>(input.Shape);

        var varianceEpsilon = variance.Add(_epsilon);
        var invStd = varianceEpsilon.Transform(NumOps.Sqrt).Transform(x => MathHelper.Reciprocal(x));

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < featureSize; j++)
            {
                normalized[i, j] = NumOps.Multiply(NumOps.Subtract(input[i, j], mean[j]), invStd[j]);
            }
        }

        return normalized;
    }

    private Vector<T> UpdateRunningStatistic(Vector<T> runningStatistic, Vector<T> batchStatistic)
    {
        return runningStatistic.Multiply(_momentum).Add(batchStatistic.Multiply(NumOps.Subtract(NumOps.One, _momentum)));
    }

        /// <summary>
    /// Gets all trainable parameters of the batch normalization layer.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (gamma and beta) concatenated together.</returns>
    /// <remarks>
    /// This method returns a single vector containing all trainable parameters of the layer:
    /// - First half: gamma (scale) parameters
    /// - Second half: beta (shift) parameters
    /// 
    /// This is useful for optimization algorithms that need access to all parameters at once,
    /// or for saving/loading model weights.
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Concatenate gamma and beta parameters
        int featureSize = InputShape[0];
        var parameters = new Vector<T>(featureSize * 2);
        
        for (int i = 0; i < featureSize; i++)
        {
            parameters[i] = _gamma[i];
            parameters[i + featureSize] = _beta[i];
        }
        
        return parameters;
    }

    /// <summary>
    /// Sets all trainable parameters of the batch normalization layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (gamma and beta) concatenated together.</param>
    /// <remarks>
    /// This method expects a single vector containing all trainable parameters:
    /// - First half: gamma (scale) parameters
    /// - Second half: beta (shift) parameters
    /// 
    /// The length of the parameters vector must be exactly twice the feature size.
    /// This method is useful for loading pre-trained weights or setting parameters
    /// after optimization.
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    public override void SetParameters(Vector<T> parameters)
    {
        int featureSize = InputShape[0];
        if (parameters.Length != featureSize * 2)
            throw new ArgumentException($"Expected {featureSize * 2} parameters, but got {parameters.Length}");
        
        for (int i = 0; i < featureSize; i++)
        {
            _gamma[i] = parameters[i];
            _beta[i] = parameters[i + featureSize];
        }
    }

    /// <summary>
    /// Resets the internal state of the batch normalization layer.
    /// </summary>
    /// <remarks>
    /// This method clears all cached values from the forward and backward passes,
    /// including:
    /// - Last input tensor
    /// - Last normalized values
    /// - Last batch mean and variance
    /// - Gradients for gamma and beta parameters
    /// 
    /// It does NOT reset the learned parameters (gamma and beta) or the running statistics
    /// (running mean and variance) used for inference.
    /// 
    /// This is typically called when starting a new training epoch or when switching
    /// between training and inference modes.
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastNormalized = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
    }
}