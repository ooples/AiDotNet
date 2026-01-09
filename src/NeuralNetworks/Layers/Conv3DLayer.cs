using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a 3D convolutional layer for processing volumetric data like voxel grids.
/// </summary>
/// <remarks>
/// <para>
/// A 3D convolutional layer applies learnable filters to volumetric input data to extract
/// spatial features across all three dimensions. This is essential for processing 3D data
/// such as voxelized point clouds, medical imaging (CT/MRI), or video sequences.
/// </para>
/// <para><b>For Beginners:</b> A 3D convolutional layer is like a 2D convolution but extended
/// to work with volumetric data.
///
/// Think of it like examining a 3D cube of data:
/// - A 2D convolution slides a filter across height and width
/// - A 3D convolution slides a filter across depth, height, and width
///
/// This is useful for:
/// - Recognizing 3D shapes from voxel grids (like ModelNet40)
/// - Analyzing medical scans (CT, MRI)
/// - Processing video frames as a 3D volume
///
/// The layer learns to detect 3D patterns like edges, surfaces, and volumes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class Conv3DLayer<T> : LayerBase<T>
{
    #region Properties

    /// <summary>
    /// Gets the number of input channels expected by this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input channels represent the depth of the input volume in the channel dimension.
    /// For raw voxel data, this is typically 1 (occupancy). For multi-feature voxels,
    /// this could be higher (e.g., density, color, normals).
    /// </para>
    /// </remarks>
    public int InputChannels { get; private set; }

    /// <summary>
    /// Gets the number of output channels (filters) produced by this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each output channel corresponds to one learned 3D filter that detects a specific
    /// volumetric pattern. More output channels allow the layer to learn more diverse features
    /// but increase computational cost.
    /// </para>
    /// </remarks>
    public int OutputChannels { get; private set; }

    /// <summary>
    /// Gets the size of the 3D convolution kernel (same for depth, height, width).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The kernel size determines the receptive field of each convolution operation.
    /// Typical values are 3 (most common), 5, or 7. Larger kernels capture more context
    /// but are more computationally expensive.
    /// </para>
    /// </remarks>
    public int KernelSize { get; private set; }

    /// <summary>
    /// Gets the stride of the convolution (step size when sliding the kernel).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stride controls how much the kernel moves between positions. A stride of 1 produces
    /// the largest output. Stride of 2 halves each spatial dimension (downsampling).
    /// </para>
    /// </remarks>
    public int Stride { get; private set; }

    /// <summary>
    /// Gets the zero-padding applied to all sides of the input volume.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Padding adds zeros around the input volume to control the output size.
    /// With padding = (kernel_size - 1) / 2, the output has the same spatial dimensions
    /// as the input (when stride = 1).
    /// </para>
    /// </remarks>
    public int Padding { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training (backpropagation).
    /// </summary>
    /// <value>Always <c>true</c> for Conv3DLayer as it has learnable parameters.</value>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation for accelerated execution.
    /// </summary>
    /// <value><c>true</c> if kernels and biases are initialized and activation can be JIT compiled.</value>
    public override bool SupportsJitCompilation => _kernels != null && _biases != null && CanActivationBeJitted();

    #endregion

    #region Private Fields

    /// <summary>
    /// The learnable convolution kernels with shape [OutputChannels, InputChannels, KernelSize, KernelSize, KernelSize].
    /// </summary>
    private Tensor<T> _kernels;

    /// <summary>
    /// The learnable bias values with shape [OutputChannels], one per output channel.
    /// </summary>
    private Tensor<T> _biases;

    /// <summary>
    /// Cached gradient for kernels computed during backward pass.
    /// </summary>
    private Tensor<T>? _kernelsGradient;

    /// <summary>
    /// Cached gradient for biases computed during backward pass.
    /// </summary>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Cached input from the last forward pass, needed for backward computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output from the last forward pass (before activation), needed for backward computation.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Cached output from the last forward pass (after activation).
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Depth of input volume (cached for shape calculations).
    /// </summary>
    private int _inputDepth;

    /// <summary>
    /// Height of input volume (cached for shape calculations).
    /// </summary>
    private int _inputHeight;

    /// <summary>
    /// Width of input volume (cached for shape calculations).
    /// </summary>
    private int _inputWidth;

    #endregion

    #region GPU Training Fields
    private IGpuTensor<T>? _gpuLastInput;
    private IGpuTensor<T>? _gpuLastOutput;

    // GPU weight buffers
    private GpuTensor<T>? _gpuKernels;
    private GpuTensor<T>? _gpuBiases;

    // GPU gradient buffers
    private GpuTensor<T>? _gpuKernelsGradient;
    private GpuTensor<T>? _gpuBiasesGradient;

    // GPU velocity buffers (SGD momentum)
    private GpuTensor<T>? _gpuKernelsVelocity;
    private GpuTensor<T>? _gpuBiasesVelocity;

    // GPU Adam first moment buffers
    private GpuTensor<T>? _gpuKernelsM;
    private GpuTensor<T>? _gpuBiasesM;

    // GPU Adam second moment buffers
    private GpuTensor<T>? _gpuKernelsV;
    private GpuTensor<T>? _gpuBiasesV;
    #endregion

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-resident training.
    /// </summary>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="Conv3DLayer{T}"/> class with specified parameters.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels (filters).</param>
    /// <param name="kernelSize">Size of the 3D convolution kernel.</param>
    /// <param name="inputDepth">Depth of the input volume.</param>
    /// <param name="inputHeight">Height of the input volume.</param>
    /// <param name="inputWidth">Width of the input volume.</param>
    /// <param name="stride">Stride of the convolution. Defaults to 1.</param>
    /// <param name="padding">Zero-padding added to all sides. Defaults to 0.</param>
    /// <param name="activation">The activation function to apply. Defaults to ReLU.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any dimension parameter is non-positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a 3D convolutional layer that processes volumetric data.</para>
    /// <para>
    /// The layer will:
    /// 1. Apply 3D convolution with the specified kernel size
    /// 2. Add learned biases
    /// 3. Apply the activation function (ReLU by default)
    /// </para>
    /// </remarks>
    public Conv3DLayer(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int inputDepth,
        int inputHeight,
        int inputWidth,
        int stride = 1,
        int padding = 0,
        IActivationFunction<T>? activationFunction = null)
        : base(
            CalculateInputShape(inputChannels, inputDepth, inputHeight, inputWidth),
            CalculateOutputShape(outputChannels, inputDepth, inputHeight, inputWidth, kernelSize, stride, padding),
            activationFunction ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, kernelSize, inputDepth, inputHeight, inputWidth, stride);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;
        _inputDepth = inputDepth;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;

        _kernels = new Tensor<T>([outputChannels, inputChannels, kernelSize, kernelSize, kernelSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Conv3DLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels (filters).</param>
    /// <param name="kernelSize">Size of the 3D convolution kernel.</param>
    /// <param name="inputDepth">Depth of the input volume.</param>
    /// <param name="inputHeight">Height of the input volume.</param>
    /// <param name="inputWidth">Width of the input volume.</param>
    /// <param name="stride">Stride of the convolution. Defaults to 1.</param>
    /// <param name="padding">Zero-padding added to all sides. Defaults to 0.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply. Defaults to ReLU.</param>
    /// <remarks>
    /// <para>
    /// Vector activation functions operate on entire vectors at once, which can be more efficient
    /// for certain operations like Softmax that need to consider all elements together.
    /// </para>
    /// </remarks>
    public Conv3DLayer(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int inputDepth,
        int inputHeight,
        int inputWidth,
        int stride = 1,
        int padding = 0,
        IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(
            CalculateInputShape(inputChannels, inputDepth, inputHeight, inputWidth),
            CalculateOutputShape(outputChannels, inputDepth, inputHeight, inputWidth, kernelSize, stride, padding),
            vectorActivationFunction ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, kernelSize, inputDepth, inputHeight, inputWidth, stride);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;
        _inputDepth = inputDepth;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;

        _kernels = new Tensor<T>([outputChannels, inputChannels, kernelSize, kernelSize, kernelSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    #endregion

    #region Static Helper Methods

    /// <summary>
    /// Calculates the input shape array for the layer.
    /// </summary>
    /// <param name="channels">Number of input channels.</param>
    /// <param name="depth">Depth of the input volume.</param>
    /// <param name="height">Height of the input volume.</param>
    /// <param name="width">Width of the input volume.</param>
    /// <returns>An array representing [channels, depth, height, width].</returns>
    private static int[] CalculateInputShape(int channels, int depth, int height, int width)
    {
        return [channels, depth, height, width];
    }

    /// <summary>
    /// Calculates the output shape array based on convolution parameters.
    /// </summary>
    /// <param name="channels">Number of output channels.</param>
    /// <param name="inputDepth">Depth of the input volume.</param>
    /// <param name="inputHeight">Height of the input volume.</param>
    /// <param name="inputWidth">Width of the input volume.</param>
    /// <param name="kernelSize">Size of the convolution kernel.</param>
    /// <param name="stride">Stride of the convolution.</param>
    /// <param name="padding">Padding applied to the input.</param>
    /// <returns>An array representing [channels, outputDepth, outputHeight, outputWidth].</returns>
    private static int[] CalculateOutputShape(int channels, int inputDepth, int inputHeight, int inputWidth,
        int kernelSize, int stride, int padding)
    {
        int outputDepth = (inputDepth + 2 * padding - kernelSize) / stride + 1;
        int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException(
                $"Kernel size {kernelSize} with stride {stride} and padding {padding} produces invalid output dimensions " +
                $"[{outputDepth}, {outputHeight}, {outputWidth}] for input [{inputDepth}, {inputHeight}, {inputWidth}].");

        return [channels, outputDepth, outputHeight, outputWidth];
    }

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels.</param>
    /// <param name="kernelSize">Kernel size.</param>
    /// <param name="inputDepth">Input depth.</param>
    /// <param name="inputHeight">Input height.</param>
    /// <param name="inputWidth">Input width.</param>
    /// <param name="stride">Stride value.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is invalid.</exception>
    private static void ValidateParameters(int inputChannels, int outputChannels, int kernelSize,
        int inputDepth, int inputHeight, int inputWidth, int stride)
    {
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be positive.");
        if (outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels), "Output channels must be positive.");
        if (kernelSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), "Kernel size must be positive.");
        if (inputDepth <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputDepth), "Input depth must be positive.");
        if (inputHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputHeight), "Input height must be positive.");
        if (inputWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be positive.");
        if (stride <= 0)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be positive.");
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the kernel weights using He (Kaiming) initialization and biases to zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// He initialization scales the initial weights based on the fan-in (number of input connections)
    /// to prevent vanishing or exploding gradients during training. This is particularly effective
    /// for ReLU activation functions.
    /// </para>
    /// <para>
    /// Formula: weight ~ N(0, sqrt(2 / fan_in)) where fan_in = InputChannels * KernelSize^3
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        // === Vectorized: He initialization using TensorRandomUniformRange (Phase C: New IEngine methods) ===
        int fanIn = InputChannels * KernelSize * KernelSize * KernelSize;
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(
            NumOps.FromDouble(2.0),
            NumOps.FromDouble(fanIn)));

        // Initialize kernels in [-scale, scale] range
        _kernels = Engine.TensorRandomUniformRange<T>(_kernels.Shape, NumOps.Negate(scale), scale);

        // Initialize biases to zero
        _biases = new Tensor<T>(_biases.Shape);
        Engine.TensorFill(_biases, NumOps.Zero);
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs the forward pass of the 3D convolution operation.
    /// </summary>
    /// <param name="input">
    /// The input tensor with shape [batch, channels, depth, height, width] or [channels, depth, height, width].
    /// </param>
    /// <returns>
    /// The output tensor after convolution, bias addition, and activation.
    /// Shape: [batch, OutputChannels, outD, outH, outW] or [OutputChannels, outD, outH, outW].
    /// </returns>
    /// <exception cref="ArgumentException">Thrown when input tensor has invalid rank or dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the vectorized IEngine.Conv3D operation for CPU/GPU acceleration.
    /// The computation flow is:
    /// 1. Reshape input to 5D if needed (add batch dimension)
    /// 2. Perform 3D convolution using Engine.Conv3D
    /// 3. Add biases using Engine.TensorBroadcastAdd
    /// 4. Apply activation function
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        bool hasBatch = input.Rank == 5;
        Tensor<T> batchedInput;

        if (hasBatch)
        {
            batchedInput = input;
        }
        else if (input.Rank == 4)
        {
            batchedInput = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]);
        }
        else
        {
            throw new ArgumentException(
                $"Conv3DLayer expects 4D [C,D,H,W] or 5D [N,C,D,H,W] input, got {input.Rank}D.", nameof(input));
        }

        // Get the fused activation type for optimal GPU/CPU performance
        var fusedActivation = GetFusedActivationType();

        Tensor<T> activated;
        if (fusedActivation != FusedActivationType.None)
        {
            // Use FusedConv3D for optimal GPU kernel fusion (conv3d + bias + activation)
            activated = Engine.FusedConv3D(
                batchedInput, _kernels, _biases,
                Stride, Stride, Stride,
                Padding, Padding, Padding,
                1, 1, 1,  // dilation
                fusedActivation);

            // Store for backward pass (activated output is also pre-activation for supported activations)
            _lastPreActivation = activated;
            _lastOutput = activated;
        }
        else
        {
            // Fallback for unsupported activations: use separate operations
            var convOutput = Engine.Conv3D(
                batchedInput,
                _kernels,
                [Stride, Stride, Stride],
                [Padding, Padding, Padding],
                [1, 1, 1]);

            var withBias = AddBiases(convOutput);
            _lastPreActivation = withBias;

            activated = ApplyActivation(withBias);
            _lastOutput = activated;
        }

        if (!hasBatch && activated.Rank == 5 && activated.Shape[0] == 1)
        {
            return activated.Reshape(activated.Shape[1], activated.Shape[2], activated.Shape[3], activated.Shape[4]);
        }

        return activated;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors, keeping all data on GPU.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensor [batch, inChannels, inDepth, inHeight, inWidth] in NCDHW format.</param>
    /// <returns>GPU-resident output tensor [batch, outChannels, outDepth, outHeight, outWidth] in NCDHW format.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of the Forward method.
    /// All data stays on the GPU throughout the computation, avoiding expensive CPU-GPU transfers.</para>
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

        // Validate input shape - GPU uses NCDHW format [batch, channels, depth, height, width]
        if (input.Shape.Length < 4)
        {
            throw new ArgumentException(
                $"Conv3D input requires at least 4D tensor [C, D, H, W]. Got rank {input.Shape.Length}.");
        }

        int rank = input.Shape.Length;

        // Reshape input to 5D NCDHW [B, C, D, H, W] for 3D convolution
        IGpuTensor<T> input5D;
        bool addedBatchDimension = false;
        if (rank == 4)
        {
            // 4D [C, D, H, W] -> 5D [1, C, D, H, W]
            addedBatchDimension = true;
            input5D = input.CreateView(0, [1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]]);
        }
        else if (rank == 5)
        {
            // 5D [B, C, D, H, W] - no reshaping needed
            input5D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 4; d++)
            {
                flatBatch *= input.Shape[d];
            }
            input5D = input.CreateView(0, [flatBatch, input.Shape[rank - 4], input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        // Validate input channels
        int actualInputChannels = input5D.Shape[1];
        if (actualInputChannels != InputChannels)
        {
            throw new ArgumentException(
                $"Expected input channels {InputChannels}, but got {actualInputChannels}.");
        }

        // Map activation function to FusedActivationType
        var fusedActivation = GetFusedActivationType();

        // Execute GPU-fused Conv3D + Bias + Activation
        var result = gpuEngine.FusedConv3DGpu(
            input5D,
            _kernels,
            _biases,
            Stride, Stride, Stride,      // strideD, strideH, strideW
            Padding, Padding, Padding,    // padD, padH, padW
            1, 1, 1,                       // dilationD, dilationH, dilationW
            fusedActivation);

        // Cache for GPU-resident training
        if (IsTrainingMode)
        {
            _gpuLastInput = input;
            _gpuLastOutput = result;
        }

        // Restore original shape if needed
        if (addedBatchDimension)
        {
            // Input was 4D [C, D, H, W], output should also be 4D [OutC, OutD, OutH, OutW]
            return result.CreateView(0, [OutputChannels, result.Shape[2], result.Shape[3], result.Shape[4]]);
        }

        return result;
    }

    /// <summary>
    /// Adds bias values to each output channel using vectorized operations.
    /// </summary>
    /// <param name="convOutput">The convolution output tensor [batch, channels, depth, height, width].</param>
    /// <returns>Tensor with biases added to each channel.</returns>
    private Tensor<T> AddBiases(Tensor<T> convOutput)
    {
        int batch = convOutput.Shape[0];
        int channels = convOutput.Shape[1];
        int depth = convOutput.Shape[2];
        int height = convOutput.Shape[3];
        int width = convOutput.Shape[4];

        var biasExpanded = _biases.Reshape(1, channels, 1, 1, 1);
        return Engine.TensorBroadcastAdd(convOutput, biasExpanded);
    }

    #endregion

    #region Backward Pass

    /// <summary>
    /// Performs the backward pass to compute gradients for training.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called.</exception>
    /// <remarks>
    /// <para>
    /// The backward pass routes to either manual or autodiff implementation based on the
    /// <see cref="LayerBase{T}.UseAutodiff"/> property.
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
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes:
    /// 1. Activation derivative applied to output gradient
    /// 2. Input gradient using Engine.Conv3DBackwardInput
    /// 3. Kernel gradient using Engine.Conv3DBackwardKernel
    /// 4. Bias gradient by summing over batch and spatial dimensions
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastPreActivation == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Handle any-rank gradient: reshape to match _lastOutput rank before ApplyActivationDerivative
        bool hasBatch = _lastInput.Rank == 5;
        Tensor<T> outGrad5D = (outputGradient.Rank == 4 && _lastOutput.Rank == 5)
            ? outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2], outputGradient.Shape[3])
            : outputGradient;

        var delta = ApplyActivationDerivative(_lastOutput, outGrad5D);

        Tensor<T> batchedDelta = hasBatch
            ? delta
            : (delta.Rank == 4
                ? delta.Reshape(1, delta.Shape[0], delta.Shape[1], delta.Shape[2], delta.Shape[3])
                : delta);

        Tensor<T> batchedInput = hasBatch
            ? _lastInput
            : _lastInput.Reshape(1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2], _lastInput.Shape[3]);

        var inputGrad = Engine.Conv3DBackwardInput(
            batchedDelta,
            _kernels,
            batchedInput.Shape,
            [Stride, Stride, Stride],
            [Padding, Padding, Padding],
            [1, 1, 1]);

        _kernelsGradient = Engine.Conv3DBackwardKernel(
            batchedDelta,
            batchedInput,
            _kernels.Shape,
            [Stride, Stride, Stride],
            [Padding, Padding, Padding],
            [1, 1, 1]);

        _biasesGradient = ComputeBiasGradient(batchedDelta);

        if (!hasBatch && inputGrad.Rank == 5 && inputGrad.Shape[0] == 1)
        {
            return inputGrad.Reshape(inputGrad.Shape[1], inputGrad.Shape[2], inputGrad.Shape[3], inputGrad.Shape[4]);
        }

        return inputGrad;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It's useful for:
    /// - Verifying gradient correctness
    /// - Rapid prototyping with custom modifications
    /// - Research and experimentation
    /// </para>
    /// <para>
    /// Currently falls back to manual implementation as Conv3D autodiff is pending integration.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // Conv3D autodiff operations are pending full integration
        // Fall back to manual implementation for now
        return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Computes the bias gradient by summing gradients over batch and spatial dimensions.
    /// </summary>
    /// <param name="delta">The gradient tensor [batch, channels, depth, height, width].</param>
    /// <returns>Bias gradient tensor [channels].</returns>
    private Tensor<T> ComputeBiasGradient(Tensor<T> delta)
    {
        int channels = delta.Shape[1];
        var biasGrad = new T[channels];

        var sumOverBatchAndSpatial = Engine.ReduceSum(delta, [0, 2, 3, 4], keepDims: false);

        for (int c = 0; c < channels; c++)
        {
            biasGrad[c] = sumOverBatchAndSpatial[c];
        }

        return new Tensor<T>(biasGrad, [channels]);
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates the layer parameters using the computed gradients and learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for gradient descent.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called.</exception>
    public override void UpdateParameters(T learningRate)
    {
        if (_kernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        var scaledKernelGrad = Engine.TensorMultiplyScalar(_kernelsGradient, learningRate);
        _kernels = Engine.TensorSubtract(_kernels, scaledKernelGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasesGradient, learningRate);
        _biases = Engine.TensorSubtract(_biases, scaledBiasGrad);

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all kernel and bias parameters.</returns>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            new Vector<T>(_kernels.ToArray()),
            new Vector<T>(_biases.ToArray()));
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters (kernels followed by biases).</param>
    /// <exception cref="ArgumentException">Thrown when parameter count does not match expected.</exception>
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = _kernels.Length + _biases.Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, but got {parameters.Length}");

        int index = 0;
        _kernels = new Tensor<T>(_kernels.Shape, parameters.Slice(index, _kernels.Length));
        index += _kernels.Length;
        _biases = new Tensor<T>(_biases.Shape, parameters.Slice(index, _biases.Length));

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Gets the kernel weights tensor.
    /// </summary>
    /// <returns>The kernel tensor with shape [OutputChannels, InputChannels, KernelSize, KernelSize, KernelSize].</returns>
    public override Tensor<T> GetWeights() => _kernels;

    /// <summary>
    /// Gets the bias tensor.
    /// </summary>
    /// <returns>The bias tensor with shape [OutputChannels].</returns>
    public override Tensor<T> GetBiases() => _biases;

    /// <summary>
    /// Gets the convolution filter kernels.
    /// </summary>
    /// <returns>The kernel tensor.</returns>
    public Tensor<T> GetFilters() => _kernels;

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <value>
    /// The sum of the number of kernel weights and biases.
    /// </value>
    /// <remarks>
    /// <para>
    /// This equals: OutputChannels * InputChannels * KernelSize^3 + OutputChannels
    /// </para>
    /// </remarks>
    public override int ParameterCount => _kernels.Length + _biases.Length;

    /// <summary>
    /// Creates a deep copy of the layer with the same configuration and parameters.
    /// </summary>
    /// <returns>A new instance of the <see cref="Conv3DLayer{T}"/> with identical configuration and parameters.</returns>
    /// <remarks>
    /// <para>
    /// The clone is completely independent from the original layer. Changes to one
    /// will not affect the other.
    /// </para>
    /// </remarks>
    public override LayerBase<T> Clone()
    {
        Conv3DLayer<T> copy;

        if (UsingVectorActivation)
        {
            copy = new Conv3DLayer<T>(
                InputChannels,
                OutputChannels,
                KernelSize,
                _inputDepth,
                _inputHeight,
                _inputWidth,
                Stride,
                Padding,
                VectorActivation);
        }
        else
        {
            copy = new Conv3DLayer<T>(
                InputChannels,
                OutputChannels,
                KernelSize,
                _inputDepth,
                _inputHeight,
                _inputWidth,
                Stride,
                Padding,
                ScalarActivation);
        }

        copy.SetParameters(GetParameters());
        return copy;
    }

    #endregion

    #region State Management

    /// <summary>
    /// Resets the cached state from forward/backward passes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this method to free memory after training is complete or when switching
    /// between training and inference modes.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastPreActivation = null;
        _lastOutput = null;
        _kernelsGradient = null;
        _biasesGradient = null;
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Serializes the layer to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(KernelSize);
        writer.Write(Stride);
        writer.Write(Padding);
        writer.Write(_inputDepth);
        writer.Write(_inputHeight);
        writer.Write(_inputWidth);

        var kernelArray = _kernels.ToArray();
        for (int i = 0; i < kernelArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(kernelArray[i]));
        }

        var biasArray = _biases.ToArray();
        for (int i = 0; i < biasArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(biasArray[i]));
        }
    }

    /// <summary>
    /// Deserializes the layer from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        InputChannels = reader.ReadInt32();
        OutputChannels = reader.ReadInt32();
        KernelSize = reader.ReadInt32();
        Stride = reader.ReadInt32();
        Padding = reader.ReadInt32();
        _inputDepth = reader.ReadInt32();
        _inputHeight = reader.ReadInt32();
        _inputWidth = reader.ReadInt32();

        _kernels = new Tensor<T>([OutputChannels, InputChannels, KernelSize, KernelSize, KernelSize]);
        var kernelArray = new T[_kernels.Length];
        for (int i = 0; i < kernelArray.Length; i++)
        {
            kernelArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _kernels = new Tensor<T>(kernelArray, _kernels.Shape);

        _biases = new Tensor<T>([OutputChannels]);
        var biasArray = new T[_biases.Length];
        for (int i = 0; i < biasArray.Length; i++)
        {
            biasArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _biases = new Tensor<T>(biasArray, _biases.Shape);
    }

    #endregion

    #region JIT Compilation

    /// <summary>
    /// Exports the layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input nodes.</param>
    /// <returns>The output computation node.</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer is not properly initialized.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_kernels == null || _biases == null)
            throw new InvalidOperationException("Layer weights not initialized.");

        // Create input node with batch dimension
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "conv3d_input");
        inputNodes.Add(inputNode);

        // Create constant nodes for kernels and biases
        var kernelNode = TensorOperations<T>.Constant(_kernels, "conv3d_kernel");
        var biasNode = TensorOperations<T>.Constant(_biases, "conv3d_bias");

        // Build the actual convolution graph: Conv3D(input, kernel) + bias -> activation
        var convNode = TensorOperations<T>.Conv3D(
            inputNode,
            kernelNode,
            biasNode,
            new int[] { Stride, Stride, Stride },
            new int[] { Padding, Padding, Padding });

        // Apply activation function to the convolution result
        var activatedOutput = ApplyActivationToGraph(convNode);
        return activatedOutput;
    }

    #endregion

    #region GPU Training Methods

    /// <summary>
    /// Performs the backward pass on GPU tensors.
    /// </summary>
    /// <param name="outputGradient">GPU tensor containing the gradient of the loss with respect to the output.</param>
    /// <returns>GPU tensor containing the gradient of the loss with respect to the input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // CPU fallback: download gradient, run Backward(), upload result
        var outputGradCpu = outputGradient.ToTensor();
        var inputGradCpu = Backward(outputGradCpu);

        // Upload gradient buffers to GPU for UpdateParametersGpu
        if (_kernelsGradient is not null)
        {
            _gpuKernelsGradient?.Dispose();
            _gpuKernelsGradient = new GpuTensor<T>(backend, _kernelsGradient, GpuTensorRole.Gradient);
        }

        if (_biasesGradient is not null)
        {
            _gpuBiasesGradient?.Dispose();
            _gpuBiasesGradient = new GpuTensor<T>(backend, _biasesGradient, GpuTensorRole.Gradient);
        }

        return new GpuTensor<T>(backend, inputGradCpu, GpuTensorRole.Gradient);
    }

    /// <summary>
    /// Updates parameters on GPU using the configured optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Ensure GPU weight buffers exist
        _gpuKernels ??= new GpuTensor<T>(backend, _kernels, GpuTensorRole.Weight);
        _gpuBiases ??= new GpuTensor<T>(backend, _biases, GpuTensorRole.Weight);

        // Ensure optimizer state exists
        EnsureConv3DOptimizerState(config, backend);

        // Apply updates for kernels
        if (_gpuKernelsGradient is not null)
        {
            var kernelsState = BuildConv3DOptimizerState("kernels");
            config.ApplyUpdate(backend, _gpuKernels.Buffer, _gpuKernelsGradient.Buffer, kernelsState, _kernels.Length);
        }

        // Apply updates for biases
        if (_gpuBiasesGradient is not null)
        {
            var biasesState = BuildConv3DOptimizerState("biases");
            config.ApplyUpdate(backend, _gpuBiases.Buffer, _gpuBiasesGradient.Buffer, biasesState, _biases.Length);
        }

        // Download updated weights back to CPU tensors
        _kernels = _gpuKernels.ToTensor();
        _biases = _gpuBiases.ToTensor();

        // Notify engine that tensor data has changed
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    private void EnsureConv3DOptimizerState(IGpuOptimizerConfig config, IDirectGpuBackend backend)
    {
        var optimizerType = config.OptimizerType;

        // Ensure velocity buffers for SGD momentum, NAG, LARS
        if (optimizerType == GpuOptimizerType.Sgd || optimizerType == GpuOptimizerType.Nag || optimizerType == GpuOptimizerType.Lars)
        {
            _gpuKernelsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_kernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }

        // Ensure Adam moment buffers
        if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW)
        {
            _gpuKernelsM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_kernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuKernelsV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_kernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
    }

    private GpuOptimizerState BuildConv3DOptimizerState(string paramName)
    {
        return paramName switch
        {
            "kernels" => new GpuOptimizerState
            {
                Velocity = _gpuKernelsVelocity?.Buffer,
                M = _gpuKernelsM?.Buffer,
                V = _gpuKernelsV?.Buffer
            },
            "biases" => new GpuOptimizerState
            {
                Velocity = _gpuBiasesVelocity?.Buffer,
                M = _gpuBiasesM?.Buffer,
                V = _gpuBiasesV?.Buffer
            },
            _ => new GpuOptimizerState()
        };
    }

    #endregion
}
