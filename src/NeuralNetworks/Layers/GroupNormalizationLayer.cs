using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Group Normalization layer that normalizes inputs across groups of channels.
/// </summary>
/// <remarks>
/// <para>
/// Group Normalization divides channels into groups and normalizes the features within each group.
/// This makes it invariant to batch size, making it suitable for small batch sizes or applications
/// where batch statistics are not reliable (like VAEs and generative models).
/// </para>
/// <para><b>For Beginners:</b> This layer helps stabilize training for convolutional networks.
///
/// Think of Group Normalization like organizing students into study groups:
/// - Each group (of channels) studies together and normalizes their behavior
/// - It works the same regardless of class size (batch size)
/// - This is especially useful for generative models like VAEs where batch sizes may be small
///
/// Key advantages:
/// - Works well with small batch sizes (even batch size of 1)
/// - More stable than Batch Normalization for generative models
/// - Used extensively in modern architectures like Stable Diffusion VAE
///
/// Typical usage:
/// - numGroups=32 for 256+ channels
/// - numGroups=16 for 128 channels
/// - numGroups=8 for 64 channels
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GroupNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private readonly int _numGroups;
    private readonly int _numChannels;
    private Tensor<T> _gamma;
    private Tensor<T> _beta;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastMean;
    private Tensor<T>? _lastVariance;
    private Tensor<T>? _gammaGradient;
    private Tensor<T>? _betaGradient;

    #region GPU Training Fields

    // Cached GPU tensors for GPU-resident training
    private IGpuTensor<T>? _gpuLastInput;

    // GPU weight buffers
    private GpuTensor<T>? _gpuGamma;
    private GpuTensor<T>? _gpuBeta;

    // GPU gradient buffers
    private IGpuTensor<T>? _gpuGammaGradient;
    private IGpuTensor<T>? _gpuBetaGradient;

    // GPU optimizer state buffers (velocity/momentum)
    private GpuTensor<T>? _gpuGammaVelocity;
    private GpuTensor<T>? _gpuBetaVelocity;

    // GPU optimizer state buffers (first moment for Adam)
    private GpuTensor<T>? _gpuGammaM;
    private GpuTensor<T>? _gpuBetaM;

    // GPU optimizer state buffers (second moment for Adam)
    private GpuTensor<T>? _gpuGammaV;
    private GpuTensor<T>? _gpuBetaV;

    #endregion

    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-resident training.
    /// </summary>
    public override bool SupportsGpuTraining => true;

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["NumGroups"] = _numGroups.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumChannels"] = _numChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Epsilon"] = Convert.ToDouble(_epsilon, System.Globalization.CultureInfo.InvariantCulture)
            .ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public int NumGroups => _numGroups;
    public int NumChannels => _numChannels;
    public Vector<T> GetGamma() => _gamma.ToVector();
    public Tensor<T> GetGammaTensor() => _gamma;
    public Vector<T> GetBeta() => _beta.ToVector();
    public Tensor<T> GetBetaTensor() => _beta;
    public T GetEpsilon() => _epsilon;

    public GroupNormalizationLayer(int numGroups, int numChannels, double epsilon = NumericalStabilityHelper.LargeEpsilon)
        : base([numChannels], [numChannels])
    {
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");
        if (numChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be positive.");
        if (numChannels % numGroups != 0)
            throw new ArgumentException($"Number of channels ({numChannels}) must be divisible by number of groups ({numGroups}).");

        _numGroups = numGroups;
        _numChannels = numChannels;
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        _gamma = Tensor<T>.CreateDefault([numChannels], NumOps.One);
        _beta = Tensor<T>.CreateDefault([numChannels], NumOps.Zero);

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
    }

    /// <summary>
    /// Tracks whether we added a batch dimension to a 3D input.
    /// </summary>
    private bool _addedBatchDimension;

    /// <summary>
    /// Original input shape for restoring higher-rank tensors after processing.
    /// </summary>
    private int[]? _originalInputShape;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        _originalInputShape = input.Shape;

        var shape = input.Shape;

        // Support any rank >= 2. Last 3 dims are [C, H, W] for 3D+, or [N, C] for 2D.
        if (shape.Length < 2)
            throw new ArgumentException($"GroupNormalization requires at least 2D input. Got rank {shape.Length}.");

        int channels;
        Tensor<T> input4D;

        if (shape.Length == 4)
        {
            channels = shape[1];
            input4D = input;
            _addedBatchDimension = false;
        }
        else if (shape.Length == 3)
        {
            channels = shape[0];
            input4D = input.Reshape(1, shape[0], shape[1], shape[2]);
            _addedBatchDimension = true;
        }
        else if (shape.Length == 2)
        {
            channels = shape[1];
            input4D = input;
            _addedBatchDimension = false;
        }
        else
        {
            // Higher rank (>= 5): flatten leading dimensions into batch, keep last 3 as [C, H, W]
            _addedBatchDimension = false;
            int flatBatch = 1;
            for (int d = 0; d < shape.Length - 3; d++)
                flatBatch *= shape[d];
            channels = shape[shape.Length - 3];
            input4D = input.Reshape(flatBatch, shape[shape.Length - 3], shape[shape.Length - 2], shape[shape.Length - 1]);
        }

        if (channels != _numChannels)
            throw new ArgumentException($"Input has {channels} channels but layer expects {_numChannels} channels.");

        // Use Engine for GPU/CPU accelerated Group Normalization
        var output = Engine.GroupNorm(
            input4D,
            _numGroups,
            _gamma,
            _beta,
            NumOps.ToDouble(_epsilon),
            out var mean,
            out var variance);

        _lastMean = mean;
        _lastVariance = variance;

        // Restore original tensor rank
        if (_originalInputShape.Length > 4)
        {
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 3] = output.Shape[1];
            outputShape[_originalInputShape.Length - 2] = output.Shape[2];
            outputShape[_originalInputShape.Length - 1] = output.Shape[3];
            return output.Reshape(outputShape);
        }
        return _addedBatchDimension
            ? output.Reshape(output.Shape[1], output.Shape[2], output.Shape[3])
            : output;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="inputs">The GPU-resident input tensors.</param>
    /// <returns>A GPU-resident output tensor after group normalization.</returns>
    /// <remarks>
    /// <para>
    /// This method performs group normalization entirely on the GPU without downloading
    /// intermediate results to CPU. Uses native GroupNorm GPU kernel for maximum performance.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input.Shape;

        // Support any rank >= 2
        if (shape.Length < 2)
            throw new ArgumentException($"GroupNormalization requires at least 2D input. Got rank {shape.Length}.");

        int batch, channels, spatialSize;
        int[] outputShape;
        bool addedBatch = false;
        _originalInputShape = shape;

        if (shape.Length == 4)
        {
            batch = shape[0];
            channels = shape[1];
            spatialSize = shape[2] * shape[3];
            outputShape = shape;
        }
        else if (shape.Length == 3)
        {
            batch = 1;
            channels = shape[0];
            spatialSize = shape[1] * shape[2];
            outputShape = new[] { 1, shape[0], shape[1], shape[2] };
            addedBatch = true;
        }
        else if (shape.Length == 2)
        {
            batch = shape[0];
            channels = shape[1];
            spatialSize = 1;
            outputShape = shape;
        }
        else
        {
            // Higher rank (>= 5): flatten leading dimensions into batch, keep last 3 as [C, H, W]
            batch = 1;
            for (int d = 0; d < shape.Length - 3; d++)
                batch *= shape[d];
            channels = shape[shape.Length - 3];
            spatialSize = shape[shape.Length - 2] * shape[shape.Length - 1];
            outputShape = new[] { batch, channels, shape[shape.Length - 2], shape[shape.Length - 1] };
        }

        if (channels != _numChannels)
            throw new ArgumentException($"Input has {channels} channels but layer expects {_numChannels} channels.");

        // Upload gamma and beta to GPU
        float[] gammaData = DirectGpuEngine.ToFloatArray<T>(_gamma.Data.ToArray());
        float[] betaData = DirectGpuEngine.ToFloatArray<T>(_beta.Data.ToArray());
        using var gammaBuffer = backend.AllocateBuffer(gammaData);
        using var betaBuffer = backend.AllocateBuffer(betaData);

        // Allocate output and statistics buffers
        int totalSize = batch * channels * spatialSize;
        var outputBuffer = backend.AllocateBuffer(totalSize);

        // Allocate buffers for mean and variance (for backward pass during training)
        int statsSize = batch * _numGroups;
        var meanBuffer = backend.AllocateBuffer(statsSize);
        var invVarBuffer = backend.AllocateBuffer(statsSize);

        // Execute GPU GroupNorm kernel
        backend.GroupNorm(
            input.Buffer,
            outputBuffer,
            gammaBuffer,
            betaBuffer,
            meanBuffer,
            invVarBuffer,
            batch,
            _numGroups,
            channels,
            spatialSize,
            (float)NumOps.ToDouble(_epsilon));

        // Cache statistics for backward pass during training
        if (IsTrainingMode)
        {
            // Cache GPU tensor for GPU-resident training
            _gpuLastInput = input;

            // Also download to CPU for backward compatibility with CPU backward pass
            _lastInput = input.ToTensor();
            _lastMean = new Tensor<T>(new[] { batch, _numGroups },
                new Vector<T>(DirectGpuEngine.FromFloatArray<T>(backend.DownloadBuffer(meanBuffer))));
            _lastVariance = new Tensor<T>(new[] { batch, _numGroups },
                new Vector<T>(DirectGpuEngine.FromFloatArray<T>(backend.DownloadBuffer(invVarBuffer))));
            _addedBatchDimension = addedBatch;
        }

        // Dispose statistics buffers if not needed
        meanBuffer.Dispose();
        invVarBuffer.Dispose();

        // Create output tensor with correct shape
        var result = new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);

        // Restore original tensor rank
        if (_originalInputShape != null && _originalInputShape.Length > 4)
        {
            return gpuEngine.ReshapeGpu(result, _originalInputShape);
        }
        if (addedBatch)
        {
            return gpuEngine.ReshapeGpu(result, new[] { shape[0], shape[1], shape[2] });
        }

        return result;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Flatten gradient to 4D the same way forward flattened input
        int rank = outputGradient.Shape.Length;
        Tensor<T> grad4D;

        if (rank == 3)
        {
            grad4D = outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2]);
        }
        else if (rank <= 4)
        {
            grad4D = outputGradient;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= outputGradient.Shape[d];
            grad4D = outputGradient.Reshape(flatBatch, outputGradient.Shape[rank - 3], outputGradient.Shape[rank - 2], outputGradient.Shape[rank - 1]);
        }

        // Get input with batch dimension for backward pass
        Tensor<T> input4D;
        if (_lastInput.Shape.Length == 3)
        {
            input4D = _lastInput.Reshape(1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2]);
        }
        else if (_lastInput.Shape.Length <= 4)
        {
            input4D = _lastInput;
        }
        else
        {
            int flatBatch = 1;
            for (int d = 0; d < _lastInput.Shape.Length - 3; d++)
                flatBatch *= _lastInput.Shape[d];
            input4D = _lastInput.Reshape(flatBatch, _lastInput.Shape[_lastInput.Shape.Length - 3], _lastInput.Shape[_lastInput.Shape.Length - 2], _lastInput.Shape[_lastInput.Shape.Length - 1]);
        }

        // Use Engine for GPU/CPU accelerated backward pass
        var inputGradient = Engine.GroupNormBackward(
            grad4D,
            input4D,
            _numGroups,
            _gamma,
            _lastMean,
            _lastVariance,
            NumOps.ToDouble(_epsilon),
            out var gradGamma,
            out var gradBeta);

        _gammaGradient = gradGamma;
        _betaGradient = gradBeta;

        // Restore to original input shape
        if (_originalInputShape != null && _originalInputShape.Length != inputGradient.Shape.Length)
        {
            return inputGradient.Reshape(_originalInputShape);
        }
        return _addedBatchDimension
            ? inputGradient.Reshape(inputGradient.Shape[1], inputGradient.Shape[2], inputGradient.Shape[3])
            : inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
        _beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
    }

    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(_gamma.ToVector(), _beta.ToVector());
    }

    public override void SetParameters(Vector<T> parameters)
    {
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

    public override void ResetState()
    {
        _lastInput = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
        _addedBatchDimension = false;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the GroupNormalization operation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representing the GroupNormalization layer.
    /// The graph divides channels into groups and normalizes within each group,
    /// then applies learned scale (gamma) and shift (beta) parameters per channel.
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

        // Apply GroupNorm operation
        var outputNode = TensorOperations<T>.GroupNorm(
            inputNode,
            _numGroups,
            gammaNode,
            betaNode,
            NumOps.ToDouble(_epsilon));

        return outputNode;
    }

    /// <summary>
    /// GPU-resident backward pass for group normalization layer.
    /// Computes gradients for gamma and beta parameters.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output (GPU tensor).</param>
    /// <returns>The gradient of the loss with respect to the layer's input (GPU tensor).</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_gpuLastInput == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Use CPU backward pass for gradient computation as fallback
        var outputGradCpu = outputGradient.ToTensor();

        // Clear existing gradients
        _gammaGradient = null;
        _betaGradient = null;

        // Perform CPU backward to compute gradients
        var inputGradCpu = Backward(outputGradCpu);

        // Upload gradients to GPU
        if (_gammaGradient != null)
            _gpuGammaGradient = new GpuTensor<T>(backend, _gammaGradient, GpuTensorRole.Gradient);
        if (_betaGradient != null)
            _gpuBetaGradient = new GpuTensor<T>(backend, _betaGradient, GpuTensorRole.Gradient);

        return new GpuTensor<T>(backend, inputGradCpu, GpuTensorRole.Gradient);
    }

    /// <summary>
    /// GPU-resident parameter update using the provided optimizer configuration.
    /// Updates gamma and beta parameters directly on GPU.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        if (_gpuGammaGradient == null || _gpuBetaGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure GPU weight tensors exist
        _gpuGamma ??= new GpuTensor<T>(backend, _gamma, GpuTensorRole.Weight);
        _gpuBeta ??= new GpuTensor<T>(backend, _beta, GpuTensorRole.Weight);

        // Ensure optimizer state exists
        EnsureGroupNormOptimizerState(config, backend);

        // Build optimizer state for each parameter
        var gammaState = BuildGroupNormOptimizerState("gamma");
        var betaState = BuildGroupNormOptimizerState("beta");

        // Apply optimizer updates on GPU
        config.ApplyUpdate(backend, _gpuGamma.Buffer, _gpuGammaGradient.Buffer, gammaState, _gamma.Length);
        config.ApplyUpdate(backend, _gpuBeta.Buffer, _gpuBetaGradient.Buffer, betaState, _beta.Length);

        // Download updated weights to CPU for backward compatibility
        _gamma = _gpuGamma.ToTensor();
        _beta = _gpuBeta.ToTensor();

        // Notify engine that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
    }

    /// <summary>
    /// Ensures optimizer state buffers are allocated for the optimizer type.
    /// </summary>
    private void EnsureGroupNormOptimizerState(IGpuOptimizerConfig config, IDirectGpuBackend backend)
    {
        var optimizerType = config.OptimizerType;

        // SGD, NAG, LARS only need velocity
        if (optimizerType == GpuOptimizerType.Sgd ||
            optimizerType == GpuOptimizerType.Nag ||
            optimizerType == GpuOptimizerType.Lars)
        {
            _gpuGammaVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_gamma.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBetaVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_beta.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
        else if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW)
        {
            // Adam, AdamW need both M (first moment) and V (second moment)
            _gpuGammaM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_gamma.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBetaM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_beta.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuGammaV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_gamma.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBetaV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_beta.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
    }

    /// <summary>
    /// Builds optimizer state for a specific parameter tensor.
    /// </summary>
    private GpuOptimizerState BuildGroupNormOptimizerState(string paramName)
    {
        return paramName switch
        {
            "gamma" => new GpuOptimizerState { Velocity = _gpuGammaVelocity?.Buffer, M = _gpuGammaM?.Buffer, V = _gpuGammaV?.Buffer },
            "beta" => new GpuOptimizerState { Velocity = _gpuBetaVelocity?.Buffer, M = _gpuBetaM?.Buffer, V = _gpuBetaV?.Buffer },
            _ => new GpuOptimizerState()
        };
    }
}
