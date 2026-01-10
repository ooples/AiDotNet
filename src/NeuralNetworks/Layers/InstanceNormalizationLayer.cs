using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
    private int[] _originalInputShape = [];

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuInput;
    private IGpuBuffer? _gpuMean;
    private IGpuBuffer? _gpuInvVar;
    private int _gpuBatch;
    private int _gpuChannels;
    private int _gpuSpatialSize;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

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

        // Register trainable parameters for GPU memory optimization (only if affine)
        if (_affine)
        {
            RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
            RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
        }
    }

    /// <summary>
    /// Performs the forward pass of instance normalization.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, channels, ...spatial].</param>
    /// <returns>Normalized output tensor with the same shape as input.</returns>
    /// <remarks>
    /// Supports any-rank tensors:
    /// - 2D [batch, channels] - normalizes each channel
    /// - 3D [batch, channels, length] - normalizes over length dimension
    /// - 4D [batch, channels, height, width] - normalizes over spatial dimensions
    /// - 5D [batch, channels, D, H, W] - normalizes over all spatial dimensions
    /// - ND - generalizes to any number of dimensions
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;

        // Instance Norm expects input [batch, channels, ...spatial]
        // We flatten to 4D and use GroupNorm with numGroups = numChannels
        Tensor<T> flattenedInput;
        int channels;

        if (input.Rank == 2)
        {
            // 2D [batch, channels] -> [batch, channels, 1, 1]
            channels = input.Shape[1];
            flattenedInput = input.Reshape(new int[] { input.Shape[0], channels, 1, 1 });
        }
        else if (input.Rank == 3)
        {
            // 3D [batch, channels, length] -> [batch, channels, length, 1]
            channels = input.Shape[1];
            flattenedInput = input.Reshape(new int[] { input.Shape[0], channels, input.Shape[2], 1 });
        }
        else if (input.Rank == 4)
        {
            // 4D [batch, channels, height, width] -> use directly
            channels = input.Shape[1];
            flattenedInput = input;
        }
        else if (input.Rank == 5)
        {
            // 5D [batch, channels, D, H, W] -> flatten spatial dims
            channels = input.Shape[1];
            int spatialDim = input.Shape[2] * input.Shape[3] * input.Shape[4];
            flattenedInput = input.Reshape(new int[] { input.Shape[0], channels, spatialDim, 1 });
        }
        else
        {
            // ND -> flatten all but batch and channel dimensions
            int batch = input.Shape[0];
            channels = input.Shape[1];
            int spatialDim = 1;
            for (int i = 2; i < input.Rank; i++)
            {
                spatialDim *= input.Shape[i];
            }
            flattenedInput = input.Reshape(new int[] { batch, channels, spatialDim, 1 });
        }

        if (channels != _numChannels)
            throw new ArgumentException($"Input has {channels} channels but layer expects {_numChannels} channels.");

        _lastInput = flattenedInput;

        // Instance Normalization is Group Normalization with numGroups = numChannels
        // This means each channel is normalized independently
        var output = Engine.GroupNorm(
            flattenedInput,
            _numChannels, // numGroups = numChannels for instance norm
            _gamma,
            _beta,
            NumOps.ToDouble(_epsilon),
            out var mean,
            out var variance);

        _lastMean = mean;
        _lastVariance = variance;

        // Reshape output back to original shape
        return output.Reshape(_originalInputShape);
    }

    /// <summary>
    /// Performs the forward pass of instance normalization on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after normalization.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the native GPU InstanceNorm operation for efficient
    /// normalization where each channel is normalized independently.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input.Shape;
        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Store original shape for reshaping output
        _originalInputShape = shape;

        // Parse input dimensions - InstanceNorm expects [batch, channels, spatial]
        int batch, channels, spatialSize;

        if (shape.Length == 2)
        {
            // [batch, channels] - spatial size is 1
            batch = shape[0];
            channels = shape[1];
            spatialSize = 1;
        }
        else if (shape.Length == 3)
        {
            // [batch, channels, length]
            batch = shape[0];
            channels = shape[1];
            spatialSize = shape[2];
        }
        else if (shape.Length == 4)
        {
            // [batch, channels, height, width]
            batch = shape[0];
            channels = shape[1];
            spatialSize = shape[2] * shape[3];
        }
        else if (shape.Length >= 5)
        {
            // [batch, channels, D, H, W, ...] - flatten all spatial dims
            batch = shape[0];
            channels = shape[1];
            spatialSize = 1;
            for (int i = 2; i < shape.Length; i++)
                spatialSize *= shape[i];
        }
        else
        {
            throw new ArgumentException($"InstanceNorm requires at least 2D input, got {shape.Length}D.");
        }

        if (channels != _numChannels)
            throw new ArgumentException($"Input has {channels} channels but layer expects {_numChannels} channels.");

        // Allocate output buffer and save buffers for mean/variance
        int totalSize = batch * channels * spatialSize;
        int statsSize = batch * channels; // Mean and variance per sample per channel
        var outputBuffer = backend.AllocateBuffer(totalSize);
        var saveMeanBuffer = backend.AllocateBuffer(statsSize);
        var saveInvVarBuffer = backend.AllocateBuffer(statsSize);

        // Upload gamma and beta parameters to GPU
        var gammaData = DirectGpuEngine.ToFloatArray<T>(_gamma.Data);
        var betaData = DirectGpuEngine.ToFloatArray<T>(_beta.Data);
        using var gammaBuffer = backend.AllocateBuffer(gammaData);
        using var betaBuffer = backend.AllocateBuffer(betaData);

        // Use native GPU InstanceNorm operation
        float epsilon = (float)NumOps.ToDouble(_epsilon);
        backend.InstanceNorm(input.Buffer, outputBuffer, gammaBuffer, betaBuffer,
            saveMeanBuffer, saveInvVarBuffer, batch, channels, spatialSize, epsilon);

        // Cache mean/variance for backward pass if training
        if (IsTrainingMode)
        {
            _gpuInput = input;
            _gpuMean = saveMeanBuffer;
            _gpuInvVar = saveInvVarBuffer;
            _gpuBatch = batch;
            _gpuChannels = channels;
            _gpuSpatialSize = spatialSize;

            // Also cache for CPU backward compatibility
            _lastInput = input.ToTensor();
            var meanData = new float[statsSize];
            var varData = new float[statsSize];
            backend.DownloadBuffer(saveMeanBuffer, meanData);
            backend.DownloadBuffer(saveInvVarBuffer, varData);
            _lastMean = new Tensor<T>([batch, channels], new Vector<T>(DirectGpuEngine.FromFloatArray<T>(meanData)));
            _lastVariance = new Tensor<T>([batch, channels], new Vector<T>(DirectGpuEngine.FromFloatArray<T>(varData)));
        }
        else
        {
            // Dispose temp buffers when not training
            saveMeanBuffer.Dispose();
            saveInvVarBuffer.Dispose();
        }

        return new GpuTensor<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the backward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient tensor.</param>
    /// <returns>GPU-resident input gradient tensor.</returns>
    public IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        if (_gpuInput == null || _gpuMean == null || _gpuInvVar == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu");

        int batch = _gpuBatch;
        int channels = _gpuChannels;
        int spatialSize = _gpuSpatialSize;
        int totalSize = batch * channels * spatialSize;

        // Upload gamma to GPU
        var gammaData = DirectGpuEngine.ToFloatArray<T>(_gamma.Data);
        using var gammaBuffer = backend.AllocateBuffer(gammaData);

        // Allocate output buffers
        var gradInputBuffer = backend.AllocateBuffer(totalSize);
        var gradGammaBuffer = backend.AllocateBuffer(channels);
        var gradBetaBuffer = backend.AllocateBuffer(channels);

        // Use GPU InstanceNormBackward kernel
        float epsilon = (float)NumOps.ToDouble(_epsilon);
        backend.InstanceNormBackward(
            outputGradient.Buffer,
            _gpuInput.Buffer,
            gammaBuffer,
            _gpuMean,
            _gpuInvVar,
            gradInputBuffer,
            gradGammaBuffer,
            gradBetaBuffer,
            batch, channels, spatialSize, epsilon);

        // Download gradGamma and gradBeta for parameter updates
        var gradGammaData = backend.DownloadBuffer(gradGammaBuffer);
        var gradBetaData = backend.DownloadBuffer(gradBetaBuffer);
        gradGammaBuffer.Dispose();
        gradBetaBuffer.Dispose();

        _gammaGradient = new Tensor<T>([channels], new Vector<T>(DirectGpuEngine.FromFloatArray<T>(gradGammaData)));
        _betaGradient = new Tensor<T>([channels], new Vector<T>(DirectGpuEngine.FromFloatArray<T>(gradBetaData)));

        // Return input gradient as GPU tensor
        return new GpuTensor<T>(backend, gradInputBuffer, outputGradient.Shape, GpuTensorRole.Gradient, ownsBuffer: true);
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

        // Flatten gradient to 4D for processing (matching forward pass)
        Tensor<T> flattenedGradient;
        int channels;

        if (outputGradient.Rank == 2)
        {
            channels = outputGradient.Shape[1];
            flattenedGradient = outputGradient.Reshape(new int[] { outputGradient.Shape[0], channels, 1, 1 });
        }
        else if (outputGradient.Rank == 3)
        {
            channels = outputGradient.Shape[1];
            flattenedGradient = outputGradient.Reshape(new int[] { outputGradient.Shape[0], channels, outputGradient.Shape[2], 1 });
        }
        else if (outputGradient.Rank == 4)
        {
            flattenedGradient = outputGradient;
        }
        else if (outputGradient.Rank == 5)
        {
            channels = outputGradient.Shape[1];
            int spatialDim = outputGradient.Shape[2] * outputGradient.Shape[3] * outputGradient.Shape[4];
            flattenedGradient = outputGradient.Reshape(new int[] { outputGradient.Shape[0], channels, spatialDim, 1 });
        }
        else
        {
            int batch = outputGradient.Shape[0];
            channels = outputGradient.Shape[1];
            int spatialDim = 1;
            for (int i = 2; i < outputGradient.Rank; i++)
            {
                spatialDim *= outputGradient.Shape[i];
            }
            flattenedGradient = outputGradient.Reshape(new int[] { batch, channels, spatialDim, 1 });
        }

        // Use Engine for GPU/CPU accelerated backward pass
        var inputGradient = Engine.GroupNormBackward(
            flattenedGradient,
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

        // Reshape gradient back to original input shape
        return inputGradient.Reshape(_originalInputShape);
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

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
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

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
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

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuMean?.Dispose();
        _gpuMean = null;
        _gpuInvVar?.Dispose();
        _gpuInvVar = null;
        _gpuBatch = 0;
        _gpuChannels = 0;
        _gpuSpatialSize = 0;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <remarks>
    /// Instance normalization supports JIT compilation by leveraging GroupNorm with numGroups = numChannels.
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the InstanceNormalization operation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representing the Instance Normalization layer.
    /// Instance normalization is implemented as GroupNorm with numGroups = numChannels,
    /// meaning each channel is normalized independently across spatial dimensions.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape is null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create gamma and beta parameter nodes
        var gammaNode = TensorOperations<T>.Constant(_gamma, "gamma");
        var betaNode = TensorOperations<T>.Constant(_beta, "beta");

        // Apply GroupNorm operation with numGroups = numChannels (instance norm)
        var outputNode = TensorOperations<T>.GroupNorm(
            inputNode,
            _numChannels, // numGroups = numChannels for instance norm
            gammaNode,
            betaNode,
            NumOps.ToDouble(_epsilon));

        return outputNode;
    }
}
