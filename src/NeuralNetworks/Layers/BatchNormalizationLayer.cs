using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

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
    private readonly T _epsilon;

    /// <summary>
    /// The momentum for updating running statistics.
    /// </summary>
    /// <remarks>
    /// Controls how much weight is given to the current batch versus previous batches
    /// when updating running statistics. Values closer to 1.0 give more weight to past
    /// statistics (slower updates).
    /// </remarks>
    private readonly T _momentum;

    /// <summary>
    /// The scale parameter applied after normalization.
    /// </summary>
    /// <remarks>
    /// Also known as gamma. This learnable parameter allows the network to scale
    /// each normalized feature. Initialized to ones.
    /// </remarks>
    private Tensor<T> _gamma;

    /// <summary>
    /// The shift parameter applied after normalization.
    /// </summary>
    /// <remarks>
    /// Also known as beta. This learnable parameter allows the network to shift
    /// each normalized feature. Initialized to zeros.
    /// </remarks>
    private Tensor<T> _beta;

    /// <summary>
    /// The running mean used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to zeros.
    /// </remarks>
    private Tensor<T> _runningMean;

    /// <summary>
    /// The running variance used during inference.
    /// </summary>
    /// <remarks>
    /// This is updated during training and used for normalization during inference.
    /// Initialized to ones.
    /// </remarks>
    private Tensor<T> _runningVariance;

    /// <summary>
    /// The input from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The batch mean from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    private Tensor<T>? _lastMean;

    /// <summary>
    /// The batch variance from the last forward pass.
    /// </summary>
    /// <remarks>
    /// Stored for use in the backward pass.
    /// </remarks>
    private Tensor<T>? _lastVariance;

    /// <summary>
    /// The gradient of the loss with respect to gamma.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update gamma.
    /// </remarks>
    private Tensor<T>? _gammaGradient;

    /// <summary>
    /// The gradient of the loss with respect to beta.
    /// </summary>
    /// <remarks>
    /// Computed during the backward pass and used to update beta.
    /// </remarks>
    private Tensor<T>? _betaGradient;

    // GPU-resident cached tensors for GPU training pipeline
    private IGpuTensor<T>? _lastInputGpu;

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
    /// <summary>
    /// Gets the gamma (scale) parameters of the batch normalization layer.
    /// </summary>
    /// <returns>The gamma tensor used for scaling normalized values.</returns>
    public Tensor<T> GetGamma()
    {
        return _gamma;
    }

    /// <summary>
    /// Gets the beta (shift) parameters of the batch normalization layer.
    /// </summary>
    /// <returns>The beta tensor used for shifting scaled values.</returns>
    public Tensor<T> GetBeta()
    {
        return _beta;
    }

    /// <summary>
    /// Initializes gamma (scale) parameters to zero.
    /// </summary>
    /// <remarks>
    /// This is used for zero-init residual in ResNet, where the last BatchNorm in each
    /// residual block has gamma initialized to zero. This makes the residual blocks
    /// start as identity mappings, which can improve training.
    /// </remarks>
    public void ZeroInitGamma()
    {
        int featureSize = InputShape[0];
        _gamma = Tensor<T>.CreateDefault([featureSize], NumOps.Zero);
    }

    /// <summary>
    /// Gets the running mean of the batch normalization layer.
    /// </summary>
    /// <returns>The running mean tensor used during inference.</returns>
    public Tensor<T> GetRunningMean()
    {
        return _runningMean;
    }

    /// <summary>
    /// Gets the running variance of the batch normalization layer.
    /// </summary>
    /// <returns>The running variance tensor used during inference.</returns>
    public Tensor<T> GetRunningVariance()
    {
        return _runningVariance;
    }
    /// <summary>
    /// Gets the epsilon value used for numerical stability.
    /// </summary>
    /// <returns>The epsilon value.</returns>
    public T GetEpsilon()
    {
        return _epsilon;
    }

    /// <summary>
    /// Gets the momentum value for running statistics.
    /// </summary>
    /// <returns>The momentum value.</returns>
    public T GetMomentum()
    {
        return _momentum;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        return new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["Epsilon"] = NumOps.ToDouble(_epsilon).ToString("R", System.Globalization.CultureInfo.InvariantCulture),
            ["Momentum"] = NumOps.ToDouble(_momentum).ToString("R", System.Globalization.CultureInfo.InvariantCulture)
        };
    }

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the BatchNormalizationLayer class.
    /// </summary>
    /// <param name="numFeatures">The number of features (neurons) to normalize.</param>
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
    /// - numFeatures: How many features (neurons) this layer will normalize
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
    public BatchNormalizationLayer(int numFeatures, double epsilon = NumericalStabilityHelper.LargeEpsilon, double momentum = 0.9)
        : base([numFeatures], [numFeatures])
    {
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        _momentum = NumOps.FromDouble(momentum);
        _gamma = Tensor<T>.CreateDefault([numFeatures], NumOps.One);
        _beta = new Tensor<T>([numFeatures]);
        _runningMean = new Tensor<T>([numFeatures]);
        _runningVariance = Tensor<T>.CreateDefault([numFeatures], NumOps.One);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
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

        if (IsTrainingMode)
        {
            // Training: Use Engine.BatchNorm to compute batch stats and normalize
            // This is fully GPU accelerated
            var output = Engine.BatchNorm(input, _gamma, _beta, NumOps.ToDouble(_epsilon), out var batchMean, out var batchVariance);

            _lastMean = batchMean;
            _lastVariance = batchVariance;

            // Update running statistics using Exponential Moving Average (Vectorized)
            // running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            T oneMinusMomentum = NumOps.Subtract(NumOps.One, _momentum);

            var momentumRunningMean = Engine.TensorMultiplyScalar(_runningMean, _momentum);
            var scaledBatchMean = Engine.TensorMultiplyScalar(batchMean, oneMinusMomentum);
            _runningMean = Engine.TensorAdd(momentumRunningMean, scaledBatchMean);

            var momentumRunningVar = Engine.TensorMultiplyScalar(_runningVariance, _momentum);
            var scaledBatchVar = Engine.TensorMultiplyScalar(batchVariance, oneMinusMomentum);
            _runningVariance = Engine.TensorAdd(momentumRunningVar, scaledBatchVar);

            return output;
        }
        else
        {
            // Inference: Use running statistics
            // output = gamma * (input - runningMean) / sqrt(runningVar + epsilon) + beta

            // Calculate scale and shift terms
            var epsilonVec = Tensor<T>.CreateDefault(_runningVariance.Shape, _epsilon);
            var variancePlusEps = Engine.TensorAdd(_runningVariance, epsilonVec);
            var stdDev = Engine.TensorSqrt(variancePlusEps);

            var scale = Engine.TensorDivide(_gamma, stdDev);
            var term2 = Engine.TensorDivide(Engine.TensorMultiply(_gamma, _runningMean), stdDev);
            var shift = Engine.TensorSubtract(_beta, term2);

            // Handle any tensor rank (2D, 3D, 4D, 5D, etc.)
            // Dimension 0 is batch, dimension 1 is features/channels
            // Dimensions 2+ are spatial dimensions
            return ApplyInferenceAnyRank(input, scale, shift);
        }
    }

    /// <summary>
    /// Applies batch normalization inference for tensors of any rank.
    /// </summary>
    /// <remarks>
    /// Supports any tensor rank >= 2. Dimension 0 is batch, dimension 1 is features/channels,
    /// and dimensions 2+ are spatial dimensions that are processed element-wise.
    /// </remarks>
    private Tensor<T> ApplyInferenceAnyRank(Tensor<T> input, Tensor<T> scale, Tensor<T> shift)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];

        // Calculate total spatial size (product of all dimensions after batch and channels)
        int spatialSize = 1;
        for (int d = 2; d < input.Shape.Length; d++)
        {
            spatialSize *= input.Shape[d];
        }

        var inputData = input.Data.Span;
        var scaleData = scale.Data.Span;
        var shiftData = shift.Data.Span;
        var outputData = new T[inputData.Length];

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                int batchOffset = n * channels * spatialSize;
                int channelOffset = c * spatialSize;
                T scaleC = scaleData[c];
                T shiftC = shiftData[c];

                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = batchOffset + channelOffset + s;
                    outputData[idx] = NumOps.Add(NumOps.Multiply(inputData[idx], scaleC), shiftC);
                }
            }
        }

        return new Tensor<T>(input.Shape, new Vector<T>(outputData));
    }

    /// <summary>
    /// Gets whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs GPU-resident batch normalization forward pass.
    /// </summary>
    /// <param name="input">GPU-resident input tensor with shape [batch, features] or [batch, channels, H, W].</param>
    /// <returns>GPU-resident output tensor with same shape as input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when GPU engine is not available.</exception>
    /// <remarks>
    /// <para>
    /// This method performs batch normalization entirely on GPU, avoiding CPU round-trips.
    /// The input and output tensors remain GPU-resident for chained GPU operations.
    /// </para>
    /// <para>
    /// During training mode, running statistics (mean and variance) are updated on GPU
    /// and then downloaded back to CPU for persistence.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];

        // Store input shape for backward pass
        _lastInput = null; // GPU path doesn't store CPU tensor

        double epsilonDouble = NumOps.ToDouble(_epsilon);
        double momentumDouble = NumOps.ToDouble(_momentum);

        // Call GPU-resident batch norm
        var (output, saveMean, saveVar) = gpuEngine.FusedBatchNormGpu(
            input,
            _gamma,
            _beta,
            ref _runningMean,
            ref _runningVariance,
            epsilonDouble,
            momentumDouble,
            IsTrainingMode);

        // Store saved values for backward pass (if training)
        if (IsTrainingMode && saveMean is not null && saveVar is not null)
        {
            _lastInputGpu = input;
            _lastMean = saveMean;
            _lastVariance = saveVar;
        }

        return output;
    }

    /// <summary>
    /// Performs GPU-resident backward pass for the batch normalization layer.
    /// Computes gradients for input, gamma, and beta entirely on GPU.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient from the next layer.</param>
    /// <returns>GPU-resident gradient to pass to the previous layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown if ForwardGpu was not called first.</exception>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine");

        if (_lastInputGpu == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu");

        float epsilon = (float)NumOps.ToDouble(_epsilon);

        // Upload saved mean and variance to GPU
        var saveMeanGpu = gpuEngine.UploadToGpu(_lastMean, GpuTensorRole.Intermediate);
        var saveVarGpu = gpuEngine.UploadToGpu(_lastVariance, GpuTensorRole.Intermediate);

        // Compute backward using GPU-resident operation
        var (gradInput, gradGamma, gradBeta) = gpuEngine.BatchNormBackwardGpu<T>(
            outputGradient,
            _lastInputGpu,
            _gamma,
            saveMeanGpu,
            saveVarGpu,
            epsilon);

        // Download gradients for parameter update
        _gammaGradient = gradGamma.ToTensor();
        _betaGradient = gradBeta.ToTensor();

        return gradInput;
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
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Use Engine for GPU/CPU accelerated Batch Normalization Backward
        var inputGradient = Engine.BatchNormBackward(
            outputGradient,
            _lastInput,
            _gamma,
            _lastMean,
            _lastVariance,
            NumOps.ToDouble(_epsilon),
            out var gradGamma,
            out var gradBeta);

        _gammaGradient = gradGamma;
        _betaGradient = gradBeta;

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It recreates the forward
    /// computation graph for normalization, scaling, and shifting, then propagates gradients through it.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Convert to computation nodes
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Use gamma and beta tensors directly
        var gammaNode = Autodiff.TensorOperations<T>.Variable(_gamma, "gamma", requiresGradient: true);
        var betaNode = Autodiff.TensorOperations<T>.Variable(_beta, "beta", requiresGradient: true);

        // Forward pass using autodiff BatchNorm operation
        var outputNode = Autodiff.TensorOperations<T>.BatchNorm(
            inputNode,
            gammaNode,
            betaNode,
            _runningMean,
            _runningVariance,
            IsTrainingMode,
            NumOps.ToDouble(_epsilon)
        );

        // Set output gradient
        outputNode.Gradient = outputGradient;

        // Inline topological sort
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients directly
        _gammaGradient = gammaNode.Gradient ?? throw new InvalidOperationException("Gamma gradient is null.");
        _betaGradient = betaNode.Gradient ?? throw new InvalidOperationException("Beta gradient is null.");

        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
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
    public override int ParameterCount => _gamma.Length + _beta.Length;

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Production-grade: Use Vector.Concatenate instead of manual loops
        return Vector<T>.Concatenate(_gamma.ToVector(), _beta.ToVector());
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
            throw new ArgumentException($"Expected {featureSize * 2} parameters, but got {parameters.Length}", nameof(parameters));

        // Production-grade: Use Tensor.FromVector instead of manual loops
        var gammaVec = parameters.Slice(0, featureSize);
        var betaVec = parameters.Slice(featureSize, featureSize);

        _gamma = Tensor<T>.FromVector(gammaVec, [featureSize]);
        _beta = Tensor<T>.FromVector(betaVec, [featureSize]);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
    }

    private Tensor<T>? _gammaVelocity;
    private Tensor<T>? _betaVelocity;

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

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            if (_gammaVelocity == null)
            {
                _gammaVelocity = new Tensor<T>(_gamma.Shape);
                _gammaVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_gammaVelocity, PersistentTensorRole.OptimizerState);
            }
            if (_betaVelocity == null)
            {
                _betaVelocity = new Tensor<T>(_beta.Shape);
                _betaVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_betaVelocity, PersistentTensorRole.OptimizerState);
            }

            gpuEngine.SgdMomentumUpdateGpu(_gamma, _gammaGradient, _gammaVelocity, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_beta, _betaGradient, _betaVelocity, lr, 0.0f, 0.0f);
        }
        else
        {
            // Production-grade: Use Engine operations instead of manual loops
            _gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
            _beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));

            // Notify GPU that tensor data has changed
            Engine.InvalidatePersistentTensor(_gamma);
            Engine.InvalidatePersistentTensor(_beta);
        }
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
        // Clear CPU cached values
        _lastInput = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;

        // Clear GPU cached tensors
        _lastInputGpu = null;
    }

    /// <summary>
    /// Exports the batch normalization layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the batch normalization operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node with shape [batch=1, features]
    /// 2. Creates constant nodes for gamma (scale) and beta (shift) parameters
    /// 3. Uses running statistics (mean and variance) for inference mode
    /// 4. Applies the batch normalization operation: gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of batch normalization for JIT.
    ///
    /// JIT compilation converts the batch normalization operation into optimized native code.
    /// During inference (prediction), batch normalization uses:
    /// - Running mean and variance collected during training (not batch statistics)
    /// - Learned scale (gamma) and shift (beta) parameters
    ///
    /// The symbolic graph allows the JIT compiler to:
    /// - Optimize the normalization formula: (x - mean) / sqrt(variance + epsilon)
    /// - Fuse the scale and shift operations: result * gamma + beta
    /// - Generate SIMD-optimized code for better performance
    ///
    /// This typically provides 5-10x speedup compared to interpreted execution.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer shape or parameters are not initialized.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured. Call InitializeWeights() or Forward() first.");

        if (_gamma == null || _beta == null)
            throw new InvalidOperationException("Layer parameters not initialized. Gamma and beta must be initialized before JIT compilation.");

        if (_runningMean == null || _runningVariance == null)
            throw new InvalidOperationException("Running statistics not initialized. Train the model first before using JIT compilation.");

        // Create symbolic input node (shape definition only, batch size adapts at runtime)
        // BatchNormalizationLayer expects input shape: [featureSize]
        // BatchNorm expects: [batch, features]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create constant nodes for gamma (scale) and beta (shift) parameters
        // Use tensors directly - no conversion needed
        var gammaNode = TensorOperations<T>.Constant(_gamma, "gamma");
        var betaNode = TensorOperations<T>.Constant(_beta, "beta");

        // Convert epsilon from T to double for BatchNorm call
        var epsilonDouble = NumOps.ToDouble(_epsilon);

        // Apply BatchNorm operation (inference mode with running statistics)
        // Use running statistics tensors directly - no conversion needed
        var batchNormNode = TensorOperations<T>.BatchNorm(
            inputNode,
            gamma: gammaNode,
            beta: betaNode,
            runningMean: _runningMean,
            runningVar: _runningVariance,
            training: false,  // Inference mode for JIT compilation
            epsilon: epsilonDouble);

        return batchNormNode;
    }

    /// <summary>
    /// Gets whether this batch normalization layer supports JIT compilation.
    /// </summary>
    /// <value>True if the layer parameters and running statistics are initialized.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. The layer supports JIT if:
    /// - Gamma (scale) and beta (shift) parameters are initialized
    /// - Running mean and variance statistics are initialized (from training)
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this layer can use JIT compilation for faster inference.
    ///
    /// The layer can be JIT compiled if:
    /// - The layer has been initialized with learnable parameters (gamma and beta)
    /// - The model has been trained, so running statistics are available
    ///
    /// Batch normalization during inference requires running statistics collected during training,
    /// so JIT compilation is only supported after the model has been trained at least once.
    ///
    /// Once these conditions are met, JIT compilation can provide significant speedup (5-10x)
    /// by optimizing the normalization, scaling, and shifting operations.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // BatchNormalization supports JIT if parameters and running statistics are initialized
            return _gamma != null && _beta != null &&
                   _runningMean != null && _runningVariance != null;
        }
    }
}
