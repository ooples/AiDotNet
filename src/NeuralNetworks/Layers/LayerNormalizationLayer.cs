

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
    private readonly T _epsilon;

    /// <summary>
    /// The scale parameters learned during training.
    /// </summary>
    private Tensor<T> _gamma;

    /// <summary>
    /// The shift parameters learned during training.
    /// </summary>
    private Tensor<T> _beta;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the mean values for each sample from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastMean;

    /// <summary>
    /// Stores the variance values for each sample from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastVariance;

    /// <summary>
    /// Stores the gradients for the gamma parameters calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _gammaGradient;

    /// <summary>
    /// Stores the gradients for the beta parameters calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _betaGradient;

    /// <summary>
    /// Returns layer-specific metadata required for cloning/serialization.
    /// </summary>
    /// <remarks>
    /// Layer normalization requires its epsilon value to be preserved to reconstruct numerically stable behavior
    /// during serialization-based cloning.
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Epsilon"] = Convert.ToDouble(_epsilon, System.Globalization.CultureInfo.InvariantCulture)
            .ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

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
    /// Gets the gamma (scale) parameters of the layer normalization layer.
    /// </summary>
    /// <returns>The gamma vector used for scaling normalized values.</returns>
    public Vector<T> GetGamma()
    {
        return _gamma.ToVector();
    }

    /// <summary>
    /// Gets the gamma tensor for JIT compilation and internal use.
    /// </summary>
    public Tensor<T> GetGammaTensor() => _gamma;

    /// <summary>
    /// Gets the beta (shift) parameters of the layer normalization layer.
    /// </summary>
    /// <returns>The beta vector used for shifting scaled values.</returns>
    public Vector<T> GetBeta()
    {
        return _beta.ToVector();
    }

    /// <summary>
    /// Gets the beta tensor for JIT compilation and internal use.
    /// </summary>
    public Tensor<T> GetBetaTensor() => _beta;

    /// <summary>
    /// Gets the normalized shape (feature size) of the layer.
    /// </summary>
    /// <returns>The normalized shape array.</returns>
    public int[] GetNormalizedShape()
    {
        return OutputShape;
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
    public LayerNormalizationLayer(int featureSize, double epsilon = NumericalStabilityHelper.LargeEpsilon)
        : base([featureSize], [featureSize])
    {
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        _gamma = Tensor<T>.CreateDefault([featureSize], NumOps.One);
        _beta = Tensor<T>.CreateDefault([featureSize], NumOps.Zero);
    }

    /// <summary>
    /// Performs the forward pass of the layer normalization layer.
    /// </summary>
    /// <param name="input">The input tensor to normalize. Shape should be [batchSize, featureSize].</param>
    /// <returns>The normalized tensor with the same shape as the input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the layer normalization. It uses the Engine's accelerated
    /// LayerNorm operation to normalize each sample independently across the feature dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method normalizes your data as it passes through the layer.
    /// 
    /// During the forward pass:
    /// 1. The layer calculates mean and variance for each sample using GPU acceleration
    /// 2. It normalizes, scales, and shifts the data in a single optimized operation
    /// 3. It stores the statistics for the backward pass
    /// 
    /// This is much faster than doing it manually for each sample.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Use Engine for GPU/CPU accelerated Layer Normalization
        // This replaces the manual loop over batch items
        var output = Engine.LayerNorm(
            input,
            _gamma,
            _beta,
            NumOps.ToDouble(_epsilon),
            out var mean,
            out var variance);

        _lastMean = mean;
        _lastVariance = variance;

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

        // Use Engine for GPU/CPU accelerated Layer Normalization Backward
        var inputGradient = Engine.LayerNormBackward(
            outputGradient,
            _lastInput,
            _gamma,
            _lastMean,
            _lastVariance,
            NumOps.ToDouble(_epsilon),
            out var gammaGradient,
            out var betaGradient);

        _gammaGradient = gammaGradient;
        _betaGradient = betaGradient;

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
    /// computation graph and propagates gradients through it.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int featureSize = _lastInput.Shape[1];

        // Convert to computation nodes
        var input = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Gamma and beta are already tensors
        var gammaNode = Autodiff.TensorOperations<T>.Variable(_gamma, "gamma", requiresGradient: true);
        var betaNode = Autodiff.TensorOperations<T>.Variable(_beta, "beta", requiresGradient: true);

        // Use LayerNorm operation for full gradient computation
        var normalizedShape = new int[] { featureSize };
        var output = Autodiff.TensorOperations<T>.LayerNorm(input, normalizedShape, gammaNode, betaNode, NumOps.ToDouble(_epsilon));

        // Set the gradient at the output
        output.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((output, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

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

        // Extract gradients from the computation graph
        if (gammaNode.Gradient != null)
        {
            _gammaGradient = gammaNode.Gradient;
        }

        if (betaNode.Gradient != null)
        {
            _betaGradient = betaNode.Gradient;
        }

        return input.Gradient ?? throw new InvalidOperationException("Input gradient was not computed during backward pass.");
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

        _gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
        _beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));
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
    public override int ParameterCount => _gamma.Length + _beta.Length;

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(_gamma.ToVector(), _beta.ToVector());
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _gamma.Length + _beta.Length;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        var gammaVec = parameters.Slice(0, _gamma.Length);
        var betaVec = parameters.Slice(_gamma.Length, _beta.Length);

        _gamma = Tensor<T>.FromVector(gammaVec, _gamma.Shape);
        _beta = Tensor<T>.FromVector(betaVec, _beta.Shape);
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
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
    }

    /// <summary>
    /// Exports the layer normalization layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the layer normalization operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node with shape [batch=1, features]
    /// 2. Creates constant nodes for gamma (scale) and beta (shift) parameters
    /// 3. Applies the layer normalization operation: gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
    /// 4. Unlike batch normalization, layer norm computes statistics per sample (no running statistics needed)
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of layer normalization for JIT.
    ///
    /// JIT compilation converts the layer normalization operation into optimized native code.
    /// Layer normalization:
    /// - Computes mean and variance for each sample independently across features
    /// - Normalizes: (x - mean) / sqrt(variance + epsilon)
    /// - Scales and shifts: result * gamma + beta
    /// - Works identically during training and inference (no batch dependency)
    ///
    /// The symbolic graph allows the JIT compiler to:
    /// - Optimize the per-sample normalization formula
    /// - Fuse the scale and shift operations
    /// - Generate SIMD-optimized code for better performance
    ///
    /// This is particularly important for Transformers and RNNs where layer norm is critical.
    /// Typically provides 5-10x speedup compared to interpreted execution.
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

        // Create symbolic input node (shape definition only, batch size adapts at runtime)
        // LayerNormalizationLayer expects input shape: [featureSize]
        // LayerNorm expects: [batch, features]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create constant nodes for gamma (scale) and beta (shift) parameters
        var gammaNode = TensorOperations<T>.Constant(_gamma, "gamma");
        var betaNode = TensorOperations<T>.Constant(_beta, "beta");

        // Convert epsilon from T to double for LayerNorm call
        var epsilonDouble = NumOps.ToDouble(_epsilon);

        // Apply LayerNorm operation
        // normalizedShape specifies the dimensions to normalize over (the feature dimension)
        var normalizedShape = new int[] { InputShape[0] };
        var layerNormNode = TensorOperations<T>.LayerNorm(
            inputNode,
            normalizedShape: normalizedShape,
            gamma: gammaNode,
            beta: betaNode,
            epsilon: epsilonDouble);

        return layerNormNode;
    }

    /// <summary>
    /// Gets whether this layer normalization layer supports JIT compilation.
    /// </summary>
    /// <value>True if the layer parameters are initialized.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. The layer supports JIT if:
    /// - Gamma (scale) and beta (shift) parameters are initialized
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this layer can use JIT compilation for faster inference.
    ///
    /// The layer can be JIT compiled if:
    /// - The layer has been initialized with learnable parameters (gamma and beta)
    ///
    /// Unlike batch normalization, layer normalization doesn't require running statistics,
    /// so it can be JIT compiled immediately after initialization. It works the same way
    /// during training and inference, computing mean and variance on the fly for each sample.
    ///
    /// Once initialized, JIT compilation can provide significant speedup (5-10x)
    /// by optimizing the per-sample normalization, scaling, and shifting operations.
    ///
    /// This is especially important for Transformers where layer norm is used extensively
    /// in every encoder and decoder block.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // LayerNormalization supports JIT if parameters are initialized
            // No running statistics needed (unlike BatchNorm)
            return _gamma != null && _beta != null;
        }
    }
}
