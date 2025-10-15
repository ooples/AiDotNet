namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements batch normalization for neural networks, which normalizes the inputs across a mini-batch.
/// </summary>
/// <remarks>
/// <para>
/// Batch normalization helps stabilize and accelerate training by normalizing layer inputs.
/// It works by normalizing each feature to have zero mean and unit variance across the batch,
/// then applying learnable scale (gamma) and shift (beta) parameters.
/// </para>
/// <para>
/// Benefits include:
/// - Faster training convergence
/// - Reduced sensitivity to weight initialization
/// - Ability to use higher learning rates
/// - Acts as a form of regularization
/// </para>
/// <para><b>For Beginners:</b> Batch normalization is like standardizing test scores in a classroom.
/// 
/// Imagine a class where each student (input) has a raw test score. Batch normalization:
/// 1. Calculates the average score and how spread out the scores are
/// 2. Converts each score to show how many standard deviations it is from the average
/// 3. Applies adjustable scaling and shifting to the standardized scores
/// 
/// This helps neural networks learn more efficiently by:
/// - Keeping input values in a consistent range
/// - Reducing the "internal covariate shift" problem
/// - Making the network less sensitive to poor weight initialization
/// - Allowing higher learning rates without divergence
/// 
/// In practice, this means your network will typically train faster and perform better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
public class BatchNormalizationLayer<T> : LayerBase<T>
{
    /// <summary>
    /// A small constant added to the variance for numerical stability.
    /// </summary>
    /// <remarks>
    /// This prevents division by zero when normalizing features with very small variance.
    /// Typical values are around 1e-5 to 1e-3.
    /// </remarks>
    private readonly T _epsilon = default!;
    
    /// <summary>
    /// The momentum for updating running statistics.
    /// </summary>
    /// <remarks>
    /// Controls how much weight is given to the current batch versus previous batches
    /// when updating running statistics. Values closer to 1.0 give more weight to past
    /// statistics (slower updates).
    /// </remarks>
    private readonly T _momentum = default!;
    
    /// <summary>
    /// The scale parameter applied after normalization.
    /// </summary>
    /// <remarks>
    /// Also known as gamma. This learnable parameter allows the network to scale
    /// each normalized feature. Initialized to ones.
    /// </remarks>
    private Vector<T> _gamma = default!;
    
    /// <summary>
    /// The shift parameter applied after normalization.
    /// </summary>
    /// <remarks>
    /// Also known as beta. This learnable parameter allows the network to shift
    /// each normalized feature. Initialized to zeros.
    /// </remarks>
    private Vector<T> _beta = default!;
    
    /// <summary>
    /// The running mean used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to zeros.
    /// </remarks>
    private Vector<T> _runningMean = default!;
    
    /// <summary>
    /// The running variance used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to ones.
    /// </remarks>
    private Vector<T> _runningVariance = default!;
    
    /// <summary>
    /// The input from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    private Tensor<T>? _lastInput;
    
    /// <summary>
    /// The normalized values from the last forward pass.
    /// </summary>
    /// <remarks>
    /// These are the values after normalization but before scaling and shifting.
    /// Stored for use in the backward pass.
    /// </remarks>
    private Tensor<T>? _lastNormalized;
    
    /// <summary>
    /// The batch mean from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    private Vector<T>? _lastMean;
    
    /// <summary>
    /// The batch variance from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    private Vector<T>? _lastVariance;
    
    /// <summary>
    /// The gradient of the loss with respect to gamma.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update gamma.
    /// </remarks>
    private Vector<T>? _gammaGradient;
    
    /// <summary>
    /// The gradient of the loss with respect to beta.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update beta.
    /// </remarks>
    private Vector<T>? _betaGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Batch normalization behaves differently during training versus inference:
    /// - During training: Uses statistics from the current batch
    /// - During inference: Uses running statistics collected during training
    /// </para>
    /// <para>
    /// This property always returns true because the layer needs to track its training state.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the network that this layer behaves differently during training versus testing.
    /// 
    /// During training, batch normalization uses statistics (mean and variance) calculated from
    /// the current batch of data. During testing or inference, it uses the average statistics
    /// collected during training.
    /// 
    /// This property being true means:
    /// - The layer needs to know whether it's in training or inference mode
    /// - The layer has parameters that can be updated during training
    /// - The layer's behavior will change depending on the mode
    /// 
    /// This is important because it affects how the network processes data and how
    /// the layer's internal statistics are updated.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the BatchNormalizationLayer class.
    /// </summary>
    /// <param name="featureSize">The number of features (neurons) to normalize.</param>
    /// <param name="epsilon">A small constant added to the variance for numerical stability (default: 1e-5).</param>
    /// <param name="momentum">The momentum for updating running statistics (default: 0.9).</param>
    /// <remarks>
    /// <para>
    /// The epsilon parameter prevents division by zero when normalizing features with very small variance.
    /// </para>
    /// <para>
    /// The momentum parameter controls how much the running statistics are updated during training:
    /// - Values closer to 1.0 give more weight to past batches (slower updates)
    /// - Values closer to 0.0 give more weight to the current batch (faster updates)
    /// </para>
    /// <para>
    /// A typical value is 0.9, which means each new batch contributes about 10% to the running statistics.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a batch normalization layer with the specified settings.
    /// 
    /// When creating a BatchNormalizationLayer:
    /// - featureSize: How many features (neurons) this layer will normalize
    /// - epsilon: A small number (like 0.00001) to prevent division by zero
    /// - momentum: How quickly running statistics are updated (0.9 means 90% old + 10% new)
    /// 
    /// For example, in a neural network for image classification:
    /// ```csharp
    /// // Create a batch normalization layer for 128 features
    /// var batchNormLayer = new BatchNormalizationLayer<float>(128);
    /// ```
    /// 
    /// The layer initializes with:
    /// - Scale parameters (gamma) set to 1.0
    /// - Shift parameters (beta) set to 0.0
    /// - Running statistics (mean and variance) initialized to 0.0 and 1.0
    /// </para>
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
    /// <para>
    /// The forward pass performs these steps:
    /// 1. If in training mode:
    ///    - Compute mean and variance of the current batch
    ///    - Update running statistics for inference
    ///    - Normalize using batch statistics
    /// 2. If in inference mode:
    ///    - Normalize using running statistics collected during training
    /// 3. Apply scale (gamma) and shift (beta) parameters
    /// </para>
    /// <para>
    /// The normalization formula is: y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
    /// </para>
    /// <para><b>For Beginners:</b> This method normalizes the input data and applies learned scaling and shifting.
    /// 
    /// During the forward pass, this method:
    /// 
    /// 1. Saves the input for later use in backpropagation
    /// 2. If in training mode:
    ///    - Calculates the mean and variance of each feature across the batch
    ///    - Updates the running statistics for use during inference
    ///    - Normalizes the data using the batch statistics
    /// 3. If in inference/testing mode:
    ///    - Uses the running statistics collected during training
    /// 4. Applies the learned scale (gamma) and shift (beta) parameters
    /// 
    /// The normalization makes each feature have approximately zero mean and unit variance,
    /// while the scale and shift parameters allow the network to learn the optimal
    /// distribution for each feature.
    /// </para>
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
    /// <para>
    /// The backward pass computes three types of gradients:
    /// 1. Gradients for the input (to pass to previous layers)
    /// 2. Gradients for gamma (scale parameter)
    /// 3. Gradients for beta (shift parameter)
    /// </para>
    /// <para>
    /// This is a complex calculation that accounts for how each input affects:
    /// - The normalized value directly
    /// - The batch mean
    /// - The batch variance
    /// </para>
    /// <para>
    /// The implementation follows the chain rule of calculus to properly backpropagate
    /// through all operations in the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the error gradients flow backward through this layer.
    /// 
    /// During backpropagation, this method:
    /// 
    /// 1. Checks that Forward() was called first
    /// 2. Creates tensors to hold the gradients for inputs and parameters
    /// 3. Calculates the inverse standard deviation (1/sqrt(variance + epsilon))
    /// 4. For each feature:
    ///    - Sums the output gradients across the batch
    ///    - Sums the product of output gradients and normalized values
    ///    - Calculates gradients for gamma and beta parameters
    ///    - Calculates gradients for each input value
    /// 
    /// The calculation is complex because in batch normalization, each input affects:
    /// - Its own normalized value directly
    /// - The mean of the batch (which affects all normalized values)
    /// - The variance of the batch (which affects all normalized values)
    /// 
    /// The formula accounts for all these dependencies using the chain rule of calculus.
    /// 
    /// This method stores the gradients for gamma and beta to use during parameter updates,
    /// and returns the gradient for the input to pass to previous layers.
    /// </para>
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
    /// Computes the mean of each feature across the batch.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, featureSize].</param>
    /// <returns>A vector containing the mean of each feature.</returns>
    /// <remarks>
    /// <para>
    /// This private helper method calculates the mean of each feature across all samples in the batch.
    /// For each feature, it sums the values across all samples and then divides by the batch size.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the average value of each feature across the batch.
    /// 
    /// For example, if we have a batch of 4 samples with 3 features each:
    /// ```
    /// [
    ///   [1.0, 2.0, 3.0],
    ///   [4.0, 5.0, 6.0],
    ///   [7.0, 8.0, 9.0],
    ///   [10.0, 11.0, 12.0]
    /// ]
    /// ```
    /// 
    /// The mean would be:
    /// ```
    /// [5.5, 6.5, 7.5]
    /// ```
    /// 
    /// This is calculated by:
    /// 1. Summing each column: [22.0, 26.0, 30.0]
    /// 2. Dividing by the batch size (4): [5.5, 6.5, 7.5]
    /// 
    /// These mean values are used in the normalization process to center the data
    /// around zero.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Computes the variance of each feature across the batch.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, featureSize].</param>
    /// <param name="mean">The mean of each feature across the batch.</param>
    /// <returns>A vector containing the variance of each feature.</returns>
    /// <remarks>
    /// <para>
    /// This private helper method calculates the variance of each feature across all samples in the batch.
    /// For each feature, it sums the squared differences from the mean across all samples and then divides
    /// by the batch size.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how spread out the values are for each feature.
    /// 
    /// Variance measures how far each value is from the mean, on average. To calculate it:
    /// 1. For each value, find the difference from the mean
    /// 2. Square that difference (to make all values positive)
    /// 3. Average all these squared differences
    /// 
    /// Using the same example as before, for the first feature with mean 5.5:
    /// - (1.0 - 5.5)� = (-4.5)� = 20.25
    /// - (4.0 - 5.5)� = (-1.5)� = 2.25
    /// - (7.0 - 5.5)� = (1.5)� = 2.25
    /// - (10.0 - 5.5)� = (4.5)� = 20.25
    /// - Average: (20.25 + 2.25 + 2.25 + 20.25) / 4 = 11.25
    /// 
    /// The variance is used in the normalization process to scale the data to have
    /// unit variance (standard deviation of 1).
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Normalizes the input tensor using the provided mean and variance.
    /// </summary>
    /// <param name="input">The input tensor with shape [batchSize, featureSize].</param>
    /// <param name="mean">The mean of each feature.</param>
    /// <param name="variance">The variance of each feature.</param>
    /// <returns>The normalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// This private helper method normalizes the input tensor by subtracting the mean and dividing by the
    /// standard deviation (square root of variance plus epsilon) for each feature. This transforms the data
    /// to have approximately zero mean and unit variance.
    /// </para>
    /// <para><b>For Beginners:</b> This method standardizes the input data to have zero mean and unit variance.
    /// 
    /// For each value in the input, this method:
    /// 1. Subtracts the mean (to center the data around zero)
    /// 2. Divides by the standard deviation (to scale the data to have unit variance)
    /// 
    /// The formula is: normalized = (input - mean) / sqrt(variance + epsilon)
    /// 
    /// The epsilon value is a small constant added for numerical stability to prevent
    /// division by zero when the variance is very small.
    /// 
    /// This standardization makes the data more consistent and helps the network
    /// learn more efficiently.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Updates a running statistic with a new batch statistic.
    /// </summary>
    /// <param name="runningStatistic">The current running statistic (mean or variance).</param>
    /// <param name="batchStatistic">The statistic from the current batch.</param>
    /// <returns>The updated running statistic.</returns>
    /// <remarks>
    /// <para>
    /// This private helper method updates a running statistic (either mean or variance) using the
    /// exponential moving average formula:
    /// 
    /// runningStatistic = momentum * runningStatistic + (1 - momentum) * batchStatistic
    /// </para>
    /// <para>
    /// This gives more weight to past statistics (controlled by the momentum parameter) and less
    /// weight to the current batch, resulting in a more stable estimate over time.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the running average of statistics used during inference.
    /// 
    /// During training, we calculate statistics (mean and variance) for each batch.
    /// But during inference, we need stable statistics that represent the entire dataset.
    /// 
    /// This method implements an exponential moving average that gradually updates
    /// the running statistics with each new batch:
    /// 
    /// newRunningValue = momentum * oldRunningValue + (1 - momentum) * currentBatchValue
    /// 
    /// For example, with momentum = 0.9:
    /// - 90% of the old running value is kept
    /// - 10% of the current batch value is added
    /// 
    /// This creates a smoothed estimate that becomes more stable as training progresses,
    /// which is then used during inference instead of batch-specific statistics.
    /// </para>
    /// </remarks>
    private Vector<T> UpdateRunningStatistic(Vector<T> runningStatistic, Vector<T> batchStatistic)
    {
        return runningStatistic.Multiply(_momentum).Add(batchStatistic.Multiply(NumOps.Subtract(NumOps.One, _momentum)));
    }

        /// <summary>
    /// Gets all trainable parameters of the batch normalization layer.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (gamma and beta) concatenated together.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a single vector containing all trainable parameters of the layer:
    /// - First half: gamma (scale) parameters
    /// - Second half: beta (shift) parameters
    /// </para>
    /// <para>
    /// This is useful for optimization algorithms that need access to all parameters at once,
    /// or for saving/loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the learnable parameters as a single vector.
    /// 
    /// Batch normalization has two sets of learnable parameters:
    /// - Gamma (scale): Controls how much to stretch or compress the normalized data
    /// - Beta (shift): Controls how much to move the normalized data up or down
    /// 
    /// This method combines both sets into a single vector, with gamma values first,
    /// followed by beta values. For example, with 3 features:
    /// 
    /// [gamma1, gamma2, gamma3, beta1, beta2, beta3]
    /// 
    /// This format is useful for:
    /// - Saving and loading models
    /// - Advanced optimization algorithms that work with all parameters at once
    /// - Regularization techniques that need to access all parameters
    /// 
    /// The total length of the returned vector is twice the number of features,
    /// since there's one gamma and one beta parameter per feature.
    /// </para>
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
    /// <para>
    /// This method expects a single vector containing all trainable parameters:
    /// - First half: gamma (scale) parameters
    /// - Second half: beta (shift) parameters
    /// </para>
    /// <para>
    /// The length of the parameters vector must be exactly twice the feature size.
    /// This method is useful for loading pre-trained weights or setting parameters
    /// after optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads parameters into the layer from a single vector.
    /// 
    /// This is the counterpart to GetParameters() - it takes a vector containing
    /// all parameters and sets them in the layer. The vector must have the format:
    /// 
    /// [gamma1, gamma2, ..., gammaN, beta1, beta2, ..., betaN]
    /// 
    /// Where N is the number of features. The total length must be exactly 2*N.
    /// 
    /// This method is commonly used for:
    /// - Loading pre-trained models
    /// - Setting parameters after external optimization
    /// - Implementing transfer learning
    /// - Testing different parameter configurations
    /// 
    /// If the vector doesn't have the expected length, the method will throw an
    /// exception to prevent incorrect parameter assignments.
    /// </para>
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
    /// Updates the layer's parameters using the computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the gamma (scale) and beta (shift) parameters using gradient descent:
    /// - gamma = gamma - learningRate * gammaGradient
    /// - beta = beta - learningRate * betaGradient
    /// </para>
    /// <para>
    /// The gradients are computed during the backward pass and represent how much
    /// each parameter should change to reduce the loss function.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's learnable parameters during training.
    /// 
    /// After the backward pass calculates how each parameter affects the error,
    /// this method adjusts those parameters to reduce the error:
    /// 
    /// 1. It checks that the backward pass has been called first
    /// 2. It updates the gamma (scale) parameters:
    ///    gamma = gamma - learningRate * gammaGradient
    /// 3. It updates the beta (shift) parameters:
    ///    beta = beta - learningRate * betaGradient
    /// 
    /// The learning rate controls how big the updates are:
    /// - A larger learning rate means bigger changes (faster learning but potentially unstable)
    /// - A smaller learning rate means smaller changes (slower but more stable learning)
    /// 
    /// For example, if a particular gamma value is causing high error, its gradient
    /// will be large, and this method will adjust that parameter more significantly
    /// to reduce the error in the next forward pass.
    /// 
    /// This is the step where actual "learning" happens in the neural network.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = _gamma.Subtract(_gammaGradient.Multiply(learningRate));
        _beta = _beta.Subtract(_betaGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Resets the internal state of the batch normalization layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all cached values from the forward and backward passes,
    /// including:
    /// - Last input tensor
    /// - Last normalized values
    /// - Last batch mean and variance
    /// - Gradients for gamma and beta parameters
    /// </para>
    /// <para>
    /// It does NOT reset the learned parameters (gamma and beta) or the running statistics
    /// (running mean and variance) used for inference.
    /// </para>
    /// <para>
    /// This is typically called when starting a new training epoch or when switching
    /// between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory of previous calculations.
    /// 
    /// During training, the batch normalization layer keeps track of:
    /// - The last input it processed
    /// - The normalized values it calculated
    /// - The mean and variance of the last batch
    /// - The gradients for its parameters
    /// 
    /// This method clears all of these temporary values, which is useful when:
    /// - Starting a new training epoch
    /// - Switching between training and testing modes
    /// - Ensuring the layer behaves deterministically
    /// 
    /// Important: This does NOT reset the learned parameters (gamma and beta) or
    /// the running statistics (running mean and variance) that are used during inference.
    /// It only clears temporary calculation values.
    /// 
    /// Think of it as clearing the layer's short-term memory while preserving its
    /// long-term learning.
    /// </para>
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