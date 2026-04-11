#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

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
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.ActivationNormalization)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, TestInputShape = "1, 4", TestConstructorArgs = "2, 4")]
public partial class GroupNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private readonly int _numGroups;
    private readonly int _numChannels;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]

    private Tensor<T> _gamma;
    private Tensor<T> _beta;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastMean;
    private Tensor<T>? _lastVariance;
    private Tensor<T>? _gammaGradient;
    private Tensor<T>? _betaGradient;

    #region GPU Training Fields

    // Cached GPU tensors for GPU-resident training
    private Tensor<T>? _gpuLastInput;

    // GPU weight buffers
    private Tensor<T>? _gpuGamma;
    private Tensor<T>? _gpuBeta;

    // GPU gradient buffers
    private Tensor<T>? _gpuGammaGradient;
    private Tensor<T>? _gpuBetaGradient;

    // GPU optimizer state buffers (velocity/momentum)
    private Tensor<T>? _gpuGammaVelocity;
    private Tensor<T>? _gpuBetaVelocity;

    // GPU optimizer state buffers (first moment for Adam)
    private Tensor<T>? _gpuGammaM;
    private Tensor<T>? _gpuBetaM;

    // GPU optimizer state buffers (second moment for Adam)
    private Tensor<T>? _gpuGammaV;
    private Tensor<T>? _gpuBetaV;

    #endregion

    public override int ParameterCount => _gamma.Length + _beta.Length;
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
        _originalInputShape = input._shape;

        var shape = input._shape;

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
            // Must go through Engine so the gradient tape records the reshape —
            // direct Tensor<T>.Reshape bypasses the tape and snaps the gradient
            // chain right at the GroupNorm entry, which was the root cause of
            // diffusion tape training returning all-zero gradients.
            input4D = Engine.Reshape(input, new[] { 1, shape[0], shape[1], shape[2] });
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
            input4D = Engine.Reshape(input, new[]
            {
                flatBatch,
                shape[shape.Length - 3],
                shape[shape.Length - 2],
                shape[shape.Length - 1]
            });
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

        // Restore original tensor rank — via Engine so tape records the restore.
        if (_originalInputShape.Length > 4)
        {
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 3] = output.Shape[1];
            outputShape[_originalInputShape.Length - 2] = output.Shape[2];
            outputShape[_originalInputShape.Length - 1] = output.Shape[3];
            return Engine.Reshape(output, outputShape);
        }
        return _addedBatchDimension
            ? Engine.Reshape(output, new[] { output.Shape[1], output.Shape[2], output.Shape[3] })
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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input._shape;

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
            _lastInput = input;
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
        var result = GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);

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

    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Update in-place to preserve GPU-registered tensor references
        var updGamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
        var updBeta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));
        for (int i = 0; i < _gamma.Length; i++) _gamma[i] = updGamma[i];
        for (int i = 0; i < _beta.Length; i++) _beta[i] = updBeta[i];

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

        // Write in-place to preserve engine persistent tensor references
        var gSpan = _gamma.Data.Span;
        for (int i = 0; i < _gamma.Length; i++) gSpan[i] = parameters[i];
        var bSpan = _beta.Data.Span;
        for (int i = 0; i < _beta.Length; i++) bSpan[i] = parameters[_gamma.Length + i];

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
    }

    public override Vector<T> GetParameterGradients()
    {
        if (_gammaGradient == null || _betaGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(_gammaGradient.ToVector(), _betaGradient.ToVector());
    }

    public override void ClearGradients() { base.ClearGradients(); _gammaGradient = null; _betaGradient = null; }

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
        _gpuGamma ??= GpuTensorHelper.UploadToGpu<T>(backend, _gamma, GpuTensorRole.Weight);
        _gpuBeta ??= GpuTensorHelper.UploadToGpu<T>(backend, _beta, GpuTensorRole.Weight);

        // Ensure optimizer state exists
        EnsureGroupNormOptimizerState(config, backend);

        // Build optimizer state for each parameter
        var gammaState = BuildGroupNormOptimizerState("gamma");
        var betaState = BuildGroupNormOptimizerState("beta");

        // Apply optimizer updates on GPU
        config.ApplyUpdate(backend, _gpuGamma.Buffer, _gpuGammaGradient.Buffer, gammaState, _gamma.Length);
        config.ApplyUpdate(backend, _gpuBeta.Buffer, _gpuBetaGradient.Buffer, betaState, _beta.Length);

        // Download updated weights to CPU for backward compatibility
        _gamma = _gpuGamma;
        _beta = _gpuBeta;

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
            _gpuGammaVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_gamma.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBetaVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_beta.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
        else if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW)
        {
            // Adam, AdamW need both M (first moment) and V (second moment)
            _gpuGammaM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_gamma.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBetaM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_beta.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuGammaV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_gamma.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBetaV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_beta.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
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
