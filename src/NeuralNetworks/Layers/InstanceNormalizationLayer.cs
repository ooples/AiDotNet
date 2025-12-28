namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an Instance Normalization layer that normalizes each channel independently across spatial dimensions.
/// </summary>
/// <remarks>
/// <para>
/// Instance Normalization normalizes each channel independently for each sample in the batch.
/// Unlike Batch Normalization which computes statistics across the batch dimension,
/// Instance Normalization computes statistics independently for each sample and each channel.
/// This is essentially Group Normalization with numGroups = numChannels.
/// </para>
/// <para><b>For Beginners:</b> This layer helps stabilize training, especially for style transfer and image generation.
///
/// Think of Instance Normalization like adjusting the contrast of each color channel independently:
/// - Each channel (e.g., red, green, blue) is normalized on its own
/// - Each image in the batch is treated independently
/// - This removes instance-specific contrast information
///
/// Key advantages:
/// - Works well for style transfer and image generation
/// - Independent of batch size (works with batch size of 1)
/// - Removes instance-specific style information, making it ideal for style transfer
///
/// Common usage:
/// - Style transfer networks (separates content from style)
/// - GANs (Generative Adversarial Networks)
/// - Image-to-image translation
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class InstanceNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private readonly int _numChannels;
    private readonly bool _affine;
    private Tensor<T> _gamma;
    private Tensor<T> _beta;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastMean;
    private Tensor<T>? _lastVariance;
    private Tensor<T>? _gammaGradient;
    private Tensor<T>? _betaGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets metadata for serialization.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["NumChannels"] = _numChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Affine"] = _affine.ToString();
        metadata["Epsilon"] = Convert.ToDouble(_epsilon, System.Globalization.CultureInfo.InvariantCulture)
            .ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <summary>
    /// Gets the number of channels this layer normalizes.
    /// </summary>
    public int NumChannels => _numChannels;

    /// <summary>
    /// Gets whether affine transformation (learnable gamma and beta) is enabled.
    /// </summary>
    public bool Affine => _affine;

    /// <summary>
    /// Gets the gamma (scale) parameters.
    /// </summary>
    public Vector<T> GetGamma() => _gamma.ToVector();

    /// <summary>
    /// Gets the gamma (scale) parameters as a tensor.
    /// </summary>
    public Tensor<T> GetGammaTensor() => _gamma;

    /// <summary>
    /// Gets the beta (shift) parameters.
    /// </summary>
    public Vector<T> GetBeta() => _beta.ToVector();

    /// <summary>
    /// Gets the beta (shift) parameters as a tensor.
    /// </summary>
    public Tensor<T> GetBetaTensor() => _beta;

    /// <summary>
    /// Gets the epsilon value used for numerical stability.
    /// </summary>
    public T GetEpsilon() => _epsilon;

    /// <summary>
    /// Initializes a new instance of the InstanceNormalizationLayer.
    /// </summary>
    /// <param name="numChannels">Number of channels (features) to normalize.</param>
    /// <param name="epsilon">Small constant for numerical stability. Defaults to 1e-5.</param>
    /// <param name="affine">Whether to include learnable affine parameters (gamma and beta). Defaults to true.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when numChannels is not positive.</exception>
    public InstanceNormalizationLayer(int numChannels, double epsilon = NumericalStabilityHelper.LargeEpsilon, bool affine = true)
        : base([numChannels], [numChannels])
    {
        if (numChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be positive.");

        _numChannels = numChannels;
        _affine = affine;
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);

        // Initialize gamma to 1 and beta to 0
        _gamma = Tensor<T>.CreateDefault([numChannels], NumOps.One);
        _beta = Tensor<T>.CreateDefault([numChannels], NumOps.Zero);
    }

    /// <summary>
    /// Performs the forward pass of instance normalization.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, channels, ...spatial].</param>
    /// <returns>Normalized output tensor with the same shape as input.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        var shape = input.Shape;
        int channels = shape.Length > 1 ? shape[1] : 1;

        if (channels != _numChannels)
            throw new ArgumentException($"Input has {channels} channels but layer expects {_numChannels} channels.");

        // Instance Normalization is Group Normalization with numGroups = numChannels
        // This means each channel is normalized independently
        var output = Engine.GroupNorm(
            input,
            _numChannels, // numGroups = numChannels for instance norm
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
    /// Performs the backward pass of instance normalization.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient with respect to the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Use Engine for GPU/CPU accelerated backward pass
        var inputGradient = Engine.GroupNormBackward(
            outputGradient,
            _lastInput,
            _numChannels, // numGroups = numChannels for instance norm
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
    /// Updates the layer's parameters using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (!_affine)
            return; // No learnable parameters when affine is false

        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
        _beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing gamma and beta parameters (if affine) or empty vector.</returns>
    public override Vector<T> GetParameters()
    {
        if (!_affine)
            return new Vector<T>(0);

        return Vector<T>.Concatenate(_gamma.ToVector(), _beta.ToVector());
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (!_affine)
        {
            if (parameters.Length != 0)
                throw new ArgumentException("Non-affine InstanceNorm has no parameters, but received " + parameters.Length);
            return;
        }

        int totalParams = _gamma.Length + _beta.Length;

        if (parameters.Length != totalParams)
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");

        var gammaVec = parameters.Slice(0, _gamma.Length);
        var betaVec = parameters.Slice(_gamma.Length, _beta.Length);

        _gamma = Tensor<T>.FromVector(gammaVec, _gamma.Shape);
        _beta = Tensor<T>.FromVector(betaVec, _beta.Shape);
    }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => _affine ? _numChannels * 2 : 0;

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the InstanceNormalization operation.</returns>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        throw new NotSupportedException(
            "InstanceNormalization JIT compilation is not yet implemented. " +
            "Use the layer in interpreted mode by setting SupportsJitCompilation = false.");
    }
}
