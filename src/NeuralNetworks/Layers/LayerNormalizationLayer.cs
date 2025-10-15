namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Layer Normalization layer that normalizes inputs across the feature dimension.
/// </summary>
/// <remarks>
/// <para>
/// Layer Normalization is a technique used to normalize the inputs to a layer, which can help improve
/// training stability and speed. Unlike Batch Normalization which normalizes across the batch dimension,
/// Layer Normalization normalizes across the feature dimension independently for each sample. This makes
/// it particularly useful for recurrent networks and when batch sizes are small. The layer learns scale
/// (gamma) and shift (beta) parameters to allow the network to recover the original representation if needed.
/// </para>
/// <para><b>For Beginners:</b> This layer helps stabilize and speed up training by standardizing the data.
/// 
/// Think of Layer Normalization like standardizing test scores:
/// - It makes each sample's features have a mean of 0 and standard deviation of 1
/// - It does this independently for each sample (unlike Batch Normalization)
/// - It applies this normalization along the feature dimension
/// - After normalizing, it scales and shifts the values using learnable parameters
/// 
/// For example, in a sentiment analysis task, some input sentences might use very positive words while 
/// others use more neutral language. Layer Normalization helps the network focus on the relative importance 
/// of features within each sample rather than their absolute values.
/// 
/// This is particularly useful for:
/// - Recurrent neural networks
/// - Cases where batch sizes are small
/// - Making training more stable and faster
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LayerNormalizationLayer<T> : LayerBase<T>
{
    /// <summary>
    /// A small value added to the variance for numerical stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value prevents division by zero when the variance is very small or zero. It ensures
    /// that the normalization remains numerically stable.
    /// </para>
    /// <para><b>For Beginners:</b> This is a tiny safety value to prevent division by zero.
    /// 
    /// The epsilon value:
    /// - Prevents errors when the variation between features is extremely small
    /// - Is added to the variance before taking the square root
    /// - Is typically a very small number like 0.00001 (1e-5)
    /// 
    /// Think of it as a small safety buffer that prevents mathematical errors
    /// when the data has very little variation.
    /// </para>
    /// </remarks>
    private readonly T _epsilon = default!;

    /// <summary>
    /// The scale parameters learned during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The gamma parameters control the scale of the normalized values. They are learnable parameters
    /// that allow the layer to recover the original representation if needed.
    /// </para>
    /// <para><b>For Beginners:</b> These are learnable scaling factors for each feature.
    /// 
    /// The gamma parameters:
    /// - Control how much to scale each feature after normalization
    /// - Are initialized to 1.0 for each feature
    /// - Allow the network to amplify or reduce the importance of specific features
    /// - Are adjusted during training to find the optimal scaling
    /// 
    /// If gamma is greater than 1, it amplifies the feature; if less than 1, it reduces its importance.
    /// </para>
    /// </remarks>
    private Vector<T> _gamma = default!;

    /// <summary>
    /// The shift parameters learned during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The beta parameters control the shift of the normalized values. They are learnable parameters
    /// that allow the layer to recover the original representation if needed.
    /// </para>
    /// <para><b>For Beginners:</b> These are learnable offset values for each feature.
    /// 
    /// The beta parameters:
    /// - Control how much to shift each feature after normalization and scaling
    /// - Are initialized to 0.0 for each feature
    /// - Allow the network to shift features up or down
    /// - Are adjusted during training to find the optimal offsets
    /// 
    /// These values let the network adjust the "baseline" for each feature
    /// after normalization has centered them around zero.
    /// </para>
    /// </remarks>
    private Vector<T> _beta = default!;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the normalized tensor (before scaling and shifting) from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastNormalized;

    /// <summary>
    /// Stores the mean values for each sample from the last forward pass.
    /// </summary>
    private Vector<T>? _lastMean;

    /// <summary>
    /// Stores the standard deviation values for each sample from the last forward pass.
    /// </summary>
    private Vector<T>? _lastStd;

    /// <summary>
    /// Stores the gradients for the gamma parameters calculated during the backward pass.
    /// </summary>
    private Vector<T>? _gammaGradient;

    /// <summary>
    /// Stores the gradients for the beta parameters calculated during the backward pass.
    /// </summary>
    private Vector<T>? _betaGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters (gamma and beta).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The LayerNormalizationLayer always returns true because it contains trainable scale and shift parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has parameters that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The Layer Normalization layer always supports training because it has gamma (scale)
    /// and beta (shift) parameters that are learned during training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="LayerNormalizationLayer{T}"/> class with the specified feature size and epsilon value.
    /// </summary>
    /// <param name="featureSize">The number of features in the input data.</param>
    /// <param name="epsilon">A small value added to the variance for numerical stability. Defaults to 1e-5.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Layer Normalization layer with the specified feature size and epsilon value.
    /// The gamma parameters are initialized to 1.0 and the beta parameters are initialized to 0.0.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Layer Normalization layer with specific settings.
    /// 
    /// When creating this layer, you specify:
    /// - featureSize: How many features each sample has (like dimensions in your data)
    /// - epsilon: A tiny safety value to prevent division by zero (usually you can use the default)
    /// 
    /// The layer automatically initializes with:
    /// - Gamma values of 1.0 for each feature (neutral scaling)
    /// - Beta values of 0.0 for each feature (no initial shifting)
    /// 
    /// For example, if your data has 128 features, you would use featureSize=128.
    /// </para>
    /// </remarks>
    public LayerNormalizationLayer(int featureSize, double epsilon = 1e-5)
        : base([featureSize], [featureSize])
    {
        _epsilon = NumOps.FromDouble(epsilon);
        _gamma = Vector<T>.CreateDefault(featureSize, NumOps.One);
        _beta = new Vector<T>(featureSize);
    }

    /// <summary>
    /// Performs the forward pass of the layer normalization layer.
    /// </summary>
    /// <param name="input">The input tensor to normalize. Shape should be [batchSize, featureSize].</param>
    /// <returns>The normalized tensor with the same shape as the input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the layer normalization. It normalizes each sample
    /// independently across the feature dimension by subtracting the mean and dividing by the standard
    /// deviation. It then scales and shifts the normalized values using the learned gamma and beta parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method normalizes your data as it passes through the layer.
    /// 
    /// During the forward pass:
    /// 1. For each sample in the batch:
    ///    - Calculate the mean of all features
    ///    - Calculate the standard deviation of all features
    ///    - Normalize each feature by subtracting the mean and dividing by the standard deviation
    /// 2. Apply scaling (gamma) and shifting (beta) to the normalized values
    /// 3. Save information needed for the backward pass
    /// 
    /// This normalization helps training by standardizing the distribution of inputs to the next layer,
    /// making the network less sensitive to the scale of the input features.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int featureSize = input.Shape[1];

        var output = new Tensor<T>(input.Shape);
        _lastMean = new Vector<T>(batchSize);
        _lastStd = new Vector<T>(batchSize);
        _lastNormalized = new Tensor<T>(input.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            var sample = new Vector<T>(featureSize);
            for (int j = 0; j < featureSize; j++)
            {
                sample[j] = input[i, j];
            }

            _lastMean[i] = sample.Mean();
            _lastStd[i] = NumOps.Sqrt(NumOps.Add(sample.Variance(), _epsilon));

            for (int j = 0; j < featureSize; j++)
            {
                _lastNormalized[i, j] = NumOps.Divide(NumOps.Subtract(input[i, j], _lastMean[i]), _lastStd[i]);
                output[i, j] = NumOps.Add(NumOps.Multiply(_lastNormalized[i, j], _gamma[j]), _beta[j]);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the layer normalization layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the layer normalization, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients for the gamma and
    /// beta parameters, and returns the gradient with respect to the input for further backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output contributed to errors
    /// 2. It calculates how the gamma and beta parameters should change to reduce errors
    /// 3. It calculates how the input should change, which will be used by earlier layers
    /// 
    /// This backward computation is complex because changing the mean and standard deviation
    /// of a sample affects all features, creating interdependencies in the gradients.
    /// 
    /// The method will throw an error if you try to run it before performing a forward pass.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastNormalized == null || _lastMean == null || _lastStd == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int featureSize = _lastInput.Shape[1];

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        _gammaGradient = new Vector<T>(featureSize);
        _betaGradient = new Vector<T>(featureSize);

        for (int i = 0; i < batchSize; i++)
        {
            var dxhat = new Vector<T>(featureSize);
            T dvariance = NumOps.Zero;
            T dmean = NumOps.Zero;

            for (int j = 0; j < featureSize; j++)
            {
                T dy = outputGradient[i, j];
                dxhat[j] = NumOps.Multiply(dy, _gamma[j]);
                _gammaGradient[j] = NumOps.Add(_gammaGradient[j], NumOps.Multiply(dy, _lastNormalized[i, j]));
                _betaGradient[j] = NumOps.Add(_betaGradient[j], dy);
            }

            for (int j = 0; j < featureSize; j++)
            {
                T xhat = _lastNormalized[i, j];
                dvariance = NumOps.Add(dvariance, NumOps.Multiply(dxhat[j], NumOps.Multiply(NumOps.Subtract(_lastInput[i, j], _lastMean[i]), NumOps.FromDouble(-0.5 / Math.Pow(Convert.ToDouble(_lastStd[i]), 3)))));
                dmean = NumOps.Add(dmean, NumOps.Multiply(dxhat[j], NumOps.FromDouble(-1.0 / Convert.ToDouble(_lastStd[i]))));
            }

            T sumDiff = NumOps.Zero;
            for (int j = 0; j < featureSize; j++)
            {
                sumDiff = NumOps.Add(sumDiff, NumOps.Subtract(_lastInput[i, j], _lastMean[i]));
            }

            dmean = NumOps.Add(dmean, NumOps.Multiply(NumOps.Multiply(dvariance, NumOps.FromDouble(-2.0 / featureSize)), sumDiff));

            for (int j = 0; j < featureSize; j++)
            {
                T dx = NumOps.Add(
                    NumOps.Divide(dxhat[j], _lastStd[i]),
                    NumOps.Add(
                        NumOps.Multiply(dvariance, NumOps.Divide(NumOps.FromDouble(2), NumOps.Multiply(NumOps.FromDouble(featureSize), NumOps.Subtract(_lastInput[i, j], _lastMean[i])))),
                        NumOps.Divide(NumOps.FromDouble(1.0), NumOps.Multiply(NumOps.FromDouble(featureSize), dmean))
                    )
                );
                inputGradient[i, j] = dx;
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the gamma and beta parameters of the layer based on the gradients calculated
    /// during the backward pass. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - The gamma (scaling) and beta (shifting) values are adjusted to reduce prediction errors
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually improving
    /// its ability to normalize inputs in the most helpful way for the network.
    /// 
    /// The method will throw an error if you try to run it before performing a backward pass.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = _gamma.Subtract(_gammaGradient.Multiply(learningRate));
        _beta = _beta.Subtract(_betaGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (gamma and beta) and combines them into a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving and loading
    /// model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include gamma (scaling) and beta (shifting) values
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _gamma.Length + _beta.Length;

        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Copy gamma parameters
        for (int i = 0; i < _gamma.Length; i++)
        {
            parameters[index++] = _gamma[i];
        }

        // Copy beta parameters
        for (int i = 0; i < _beta.Length; i++)
        {
            parameters[index++] = _beta[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the gamma and beta parameters of the layer from a single vector of parameters.
    /// The vector must have the correct length to match the total number of parameters in the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The first half of the values are used for gamma (scaling)
    /// - The second half of the values are used for beta (shifting)
    /// - Throws an error if the input doesn't match the expected number of parameters
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Setting specific parameter values for testing
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _gamma.Length + _beta.Length;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int index = 0;

        // Set gamma parameters
        for (int i = 0; i < _gamma.Length; i++)
        {
            _gamma[i] = parameters[index++];
        }

        // Set beta parameters
        for (int i = 0; i < _beta.Length; i++)
        {
            _beta[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// This includes the last input, normalized values, mean, standard deviation, and gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - All stored information about previous inputs is removed
    /// - All calculated statistics (mean, standard deviation) are cleared
    /// - All gradient information is cleared
    /// - The layer is ready for new data without being influenced by previous data
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastNormalized = null;
        _lastMean = null;
        _lastStd = null;
        _gammaGradient = null;
        _betaGradient = null;
    }

    /// <summary>
    /// Copies the parameters from another layer normalization layer.
    /// </summary>
    /// <param name="source">The source layer to copy parameters from.</param>
    public void CopyParameters(LayerNormalizationLayer<T> source)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (source._gamma.Length != _gamma.Length || source._beta.Length != _beta.Length)
        {
            throw new ArgumentException("Source layer has incompatible dimensions");
        }

        _gamma = (Vector<T>)source._gamma.Clone();
        _beta = (Vector<T>)source._beta.Clone();
    }

    /// <summary>
    /// Performs a soft update of the parameters using Polyak averaging.
    /// </summary>
    /// <param name="source">The source layer to update from.</param>
    /// <param name="tau">The interpolation parameter (0 < tau <= 1).</param>
    public void SoftUpdate(LayerNormalizationLayer<T> source, T tau)
    {
        if (source == null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        if (source._gamma.Length != _gamma.Length || source._beta.Length != _beta.Length)
        {
            throw new ArgumentException("Source layer has incompatible dimensions");
        }

        // Soft update: θ' = (1 - τ) * θ' + τ * θ
        for (int i = 0; i < _gamma.Length; i++)
        {
            _gamma[i] = NumOps.Add(
                NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), _gamma[i]),
                NumOps.Multiply(tau, source._gamma[i])
            );
            _beta[i] = NumOps.Add(
                NumOps.Multiply(NumOps.Subtract(NumOps.One, tau), _beta[i]),
                NumOps.Multiply(tau, source._beta[i])
            );
        }
    }
}