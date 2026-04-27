using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.ActivationNormalization)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, HasTrainingMode = false, TestInputShape = "1, 4", TestConstructorArgs = "")]
public partial class LayerNormalizationLayer<T> : LayerBase<T>
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
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]

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

    private Tensor<T>? _gammaVelocity;
    private Tensor<T>? _betaVelocity;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuLastInput;
    private Tensor<T>? _gpuSaveMean;
    private Tensor<T>? _gpuSaveInvVar;

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
    /// Indicates whether this layer supports GPU-resident execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

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
    public LayerNormalizationLayer(double epsilon = NumericalStabilityHelper.LargeEpsilon)
        : base(new[] { -1 }, new[] { -1 })
    {
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        // Lazy init: feature dim resolved from input.Shape[^1] on first forward.
        _gamma = new Tensor<T>([0]);
        _beta = new Tensor<T>([0]);
    }

    /// <summary>
    /// Resolves <c>featureSize</c> from <c>input.Shape[^1]</c> (last dim) on the first forward call,
    /// allocates gamma/beta tensors, and registers them as trainable parameters. Per the
    /// LayerNorm contract (Ba et al. 2016), normalization happens along the feature axis.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int featureSize = input.Shape[input.Shape.Length - 1];
        if (featureSize <= 0)
        {
            throw new ArgumentException(
                $"LayerNormalizationLayer cannot resolve featureSize: input.Shape[^1] = {featureSize}.",
                nameof(input));
        }

        _gamma = Tensor<T>.CreateDefault([featureSize], NumOps.One);
        _beta = Tensor<T>.CreateDefault([featureSize], NumOps.Zero);

        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);

        ResolveShapes(new[] { featureSize }, new[] { featureSize });
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
        EnsureInitializedFromInput(input);

        if (IsTrainingMode)
        {
            _lastInput = input;
        }

        // Use Engine for GPU/CPU accelerated Layer Normalization
        // This replaces the manual loop over batch items
        var output = Engine.LayerNorm(
            input,
            _gamma,
            _beta,
            NumOps.ToDouble(_epsilon),
            out var mean,
            out var variance);

        if (IsTrainingMode)
        {
            _lastMean = mean;
            _lastVariance = variance;
        }

        return output;
    }

    /// <summary>
    /// GPU-resident forward pass for layer normalization.
    /// Normalizes input across the feature dimension entirely on GPU.
    /// </summary>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>GPU-resident normalized output tensor.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];
        double epsilonDouble = NumOps.ToDouble(_epsilon);

        var (output, saveMean, saveInvVar) = gpuEngine.LayerNormGpu(
            input, _gamma, _beta, epsilonDouble);

        // Cache state for backward pass only during training
        // Skip this expensive download during inference (50% overhead reduction)
        if (IsTrainingMode)
        {
            _gpuLastInput = input;
            _gpuSaveMean = saveMean;
            _gpuSaveInvVar = saveInvVar;
            _lastInput = input;
            _lastMean = saveMean;
            _lastVariance = saveInvVar;
        }

        return output;
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

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            if (_gammaVelocity == null)
            {
                _gammaVelocity = new Tensor<T>(_gamma._shape);
                _gammaVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_gammaVelocity, PersistentTensorRole.OptimizerState);
            }
            if (_betaVelocity == null)
            {
                _betaVelocity = new Tensor<T>(_beta._shape);
                _betaVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_betaVelocity, PersistentTensorRole.OptimizerState);
            }

            gpuEngine.SgdMomentumUpdateGpu(_gamma, _gammaGradient, _gammaVelocity, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_beta, _betaGradient, _betaVelocity, lr, 0.0f, 0.0f);
        }
        else
        {
            // Update in-place to preserve GPU-registered tensor references
            var updGamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
            var updBeta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));
            for (int i = 0; i < _gamma.Length; i++) _gamma[i] = updGamma[i];
            for (int i = 0; i < _beta.Length; i++) _beta[i] = updBeta[i];

            Engine.InvalidatePersistentTensor(_gamma);
            Engine.InvalidatePersistentTensor(_beta);
        }
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

        // Write in-place to preserve engine persistent tensor references
        var gSpan = _gamma.Data.Span;
        for (int i = 0; i < _gamma.Length; i++) gSpan[i] = parameters[i];
        var bSpan = _beta.Data.Span;
        for (int i = 0; i < _beta.Length; i++) bSpan[i] = parameters[_gamma.Length + i];

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
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
    public override Vector<T> GetParameterGradients()
    {
        if (_gammaGradient == null || _betaGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(_gammaGradient.ToVector(), _betaGradient.ToVector());
    }

    public override void ClearGradients() { base.ClearGradients(); _gammaGradient = null; _betaGradient = null; }

    public override void ResetState()
    {
        // Clear GPU cached values
        _gpuLastInput = null;
        _gpuSaveMean = null;
        _gpuSaveInvVar = null;

        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
    }
}
