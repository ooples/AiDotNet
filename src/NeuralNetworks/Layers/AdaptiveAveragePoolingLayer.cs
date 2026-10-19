using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements adaptive average pooling that outputs a fixed spatial size regardless of input dimensions.
/// </summary>
/// <remarks>
/// <para>
/// Adaptive average pooling automatically calculates the required kernel size and stride to produce
/// an output of the specified dimensions. This is particularly useful when you want to handle
/// variable input sizes but need a fixed output size (e.g., before a fully connected layer).
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular pooling uses a fixed window size (like 2x2) and reduces the image.
/// Adaptive pooling works in reverse: you specify the output size you want (like 1x1), and it
/// automatically figures out how to pool the entire input to get that size.
///
/// For example:
/// - Input: 14x14, Output: 1x1 → Pools each entire channel to a single value
/// - Input: 7x7, Output: 1x1 → Same result: each channel becomes one value
/// - Input: 56x56, Output: 7x7 → Divides into 7x7 regions and averages each
///
/// This is commonly used in ResNet and other architectures for "global average pooling" where
/// the final feature maps are reduced to a single value per channel before classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Pooling)]
[LayerTask(LayerTask.DownSampling)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 4, 4", TestConstructorArgs = "2, 2")]
// Roles from this layer's own guard - "requires rank>=3 [...,C,H,W]" in OnFirstForward - which reads
// Shape[rank-3], Shape[rank-2], Shape[rank-1] as channels, height and width. Batch is marked optional
// rather than declared as a second layout because the leading axis is genuinely absent at the rank the
// layer is tested at ([LayerProperty(TestInputShape = "1, 4, 4")]) and genuinely present one rank up;
// both forms run the same code, which carries every leading axis through untouched.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class AdaptiveAveragePoolingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _outputHeight;
    private readonly int _outputWidth;
    private int _channels;

    [Scratch]
    private Tensor<T>? _lastInput;
    private int[]? _lastInputShape;

    // GPU cached tensors for backward pass
    [ExternalState]
    private Tensor<T>? _gpuInput;
    private int _gpuBatch;
    private int _gpuChannels;
    private int _gpuInputHeight;
    private int _gpuInputWidth;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// Pooling layers don't have trainable parameters, but they support backpropagation.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdaptiveAveragePoolingLayer{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="inputHeight">The expected input height (can vary at runtime).</param>
    /// <param name="inputWidth">The expected input width (can vary at runtime).</param>
    /// <param name="outputHeight">The desired output height (default: 1 for global pooling).</param>
    /// <param name="outputWidth">The desired output width (default: 1 for global pooling).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The default output size of 1x1 creates "global average pooling",
    /// which averages all spatial positions in each channel into a single value.
    /// This is commonly used before the final classification layer in modern CNNs.
    /// </para>
    /// </remarks>
    public AdaptiveAveragePoolingLayer(
        int outputHeight = 1,
        int outputWidth = 1)
        : base(
            inputShape: [-1, -1, -1],
            outputShape: [-1, outputHeight, outputWidth])
    {
        if (outputHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputHeight), "Output height must be greater than 0.");
        }
        if (outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be greater than 0.");
        }

        _channels = -1;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written rather than generated, because the whole point of ADAPTIVE pooling is that the two
    /// spatial extents are set by configuration and not by the input: <c>OnFirstForward</c> resolves
    /// <c>ResolveShapes(new[] { c, h, w }, new[] { c, _outputHeight, _outputWidth })</c>. That is
    /// <c>Fixed</c> on both spatial axes, read off the constructor arguments, and <c>Same</c> on channels.
    /// </para>
    /// <para>
    /// This is exactly the case where a window relation would be WRONG. A fixed pooling window shrinks
    /// its axis by a ratio the caller chose; this layer instead picks whatever window makes the output
    /// come out at <c>_outputHeight</c> x <c>_outputWidth</c>, so the output extent is independent of the
    /// input extent - a 14x14 and a 56x56 feature map both leave as <c>outH</c> x <c>outW</c>.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // Ranks 3 and 4 only: rank 3 is the [C,H,W] form the layer is tested at, rank 4 the batched one.
        // Higher ranks run too - the guard is rank>=3 - but each extra leading axis would need a DISTINCT
        // role to be referred to by a relation, and there is no second batch-like role to give it.
        if (inputRank is not (3 or 4)) return null;

        var channels = new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels));
        var height = new OutputAxisContract(TensorAxis.Height, AxisRelation.Fixed(_outputHeight));
        var width = new OutputAxisContract(TensorAxis.Width, AxisRelation.Fixed(_outputWidth));

        return inputRank == 3
            ? new[] { channels, height, width }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels, height, width,
            };
    }

    /// <summary>
    /// Creates a global average pooling layer that pools to 1x1.
    /// </summary>
    /// <returns>An adaptive pooling layer that performs global average pooling.</returns>
    public static AdaptiveAveragePoolingLayer<T> GlobalPool()
    {
        return new AdaptiveAveragePoolingLayer<T>(1, 1);
    }

    /// <summary>
    /// Resolves channels and input spatial dims on first forward.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 3)
            throw new ArgumentException(
                $"AdaptiveAveragePoolingLayer requires rank>=3 [...,C,H,W] input; got rank {rank}.",
                nameof(input));

        int c = input.Shape[rank - 3];
        int h = input.Shape[rank - 2];
        int w = input.Shape[rank - 1];

        _channels = c;
        ResolveShapes(new[] { c, h, w }, new[] { c, _outputHeight, _outputWidth });
    }

    /// <summary>
    /// Performs the forward pass of adaptive average pooling.
    /// </summary>
    /// <param name="input">The input tensor of any rank >= 3. Last 3 dims are [C, H, W].</param>
    /// <returns>The pooled output tensor with same leading dims, [C, outH, outW].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Shape.Length < 3)
            throw new ArgumentException("Input must have at least 3 dimensions (channels, height, width).");

        EnsureInitializedFromInput(input);
        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)
        _lastInputShape = input._shape;

        // Handle any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        int rank = input.Shape.Length;
        int inputHeight = input.Shape[rank - 2];
        int inputWidth = input.Shape[rank - 1];

        // Global-pool fast path (output 1×1): delegate to Engine.ReduceMean so the
        // op is tape-tracked. ResNet/EfficientNet/MobileNet style classifiers use
        // GlobalPool() exclusively before the FC head, and a scalar-loop forward
        // here was returning a raw new Tensor<T> with GradFn=null — that broke the
        // backward chain at the pool boundary, leaving every conv/BN below it
        // with zero gradient and the optimizer with nothing to update.
        if (_outputHeight == 1 && _outputWidth == 1)
        {
            // Reduce H and W (last two axes), keepDims=true so the output keeps
            // [..., C, 1, 1] structure for downstream Flatten/Dense.
            int[] axes = new[] { rank - 2, rank - 1 };
            return Engine.ReduceMean(input, axes, keepDims: true);
        }

        // Non-trivial adaptive pooling (output > 1×1).
        //
        // Uniform-region fast path: when inputHeight % _outputHeight == 0 AND
        // inputWidth % _outputWidth == 0, the adaptive pool degenerates into a
        // regular average pool with stride == kernel. We can express this as
        // Reshape → ReduceMean → Reshape, all Engine ops, so the tape sees the
        // full chain and gradients flow through this layer correctly.
        //
        // Irregular case (non-divisible): handled below by the engine's
        // tape-recorded AdaptiveAvgPool2D.
        int channels = input.Shape[rank - 3];

        bool uniformH = inputHeight % _outputHeight == 0;
        bool uniformW = inputWidth % _outputWidth == 0;
        if (uniformH && uniformW)
        {
            int factorH = inputHeight / _outputHeight;
            int factorW = inputWidth / _outputWidth;

            // Reshape last three axes [C, H, W] into [C, outH, factorH, outW, factorW];
            // leading batch-like axes pass through unchanged.
            int[] expanded = new int[rank + 2];
            for (int d = 0; d < rank - 3; d++) expanded[d] = input.Shape[d];
            expanded[rank - 3] = channels;
            expanded[rank - 2] = _outputHeight;
            expanded[rank - 1] = factorH;
            expanded[rank]     = _outputWidth;
            expanded[rank + 1] = factorW;

            var reshaped = Engine.Reshape(input, expanded);
            // Reduce the two factor axes (positions rank-1 and rank+1 in the
            // expanded shape).
            var reduced = Engine.ReduceMean(reshaped, new[] { rank - 1, rank + 1 }, keepDims: false);
            return reduced;
        }

        // Irregular case (input H/W not divisible by output H/W): PyTorch's adaptive_avg_pool2d windows
        //   [floor(o * in / out), ceil((o + 1) * in / out))
        // as ONE tape-recorded engine op (a dedicated kernel and backward on GPU), with the leading batch-like
        // axes folded into the batch axis. The previous per-window TensorSlice + ReduceMean + Concatenate chain
        // issued O(outH * outW) ops per forward and as many scatters in backward.
        int planesBatch = 1;
        for (int d = 0; d < rank - 3; d++) planesBatch *= input.Shape[d];
        var input4D = rank == 4 ? input : Engine.Reshape(input, [planesBatch, channels, inputHeight, inputWidth]);
        var pooled = Engine.AdaptiveAvgPool2D(input4D, _outputHeight, _outputWidth);
        if (rank == 4) return pooled;
        int[] outShape = new int[rank];
        for (int d = 0; d < rank - 2; d++) outShape[d] = input.Shape[d];
        outShape[rank - 2] = _outputHeight;
        outShape[rank - 1] = _outputWidth;
        return Engine.Reshape(pooled, outShape);
    }

    /// <summary>
    /// Performs the forward pass of adaptive average pooling on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after pooling.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the native GPU AdaptiveAvgPool2D operation for efficient
    /// pooling to any target output size.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input._shape;
        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Handle different tensor ranks - need [batch, channels, height, width]
        int batch, channels, inputHeight, inputWidth;

        if (shape.Length == 3)
        {
            // [C, H, W] - add implicit batch of 1
            batch = 1;
            channels = shape[0];
            inputHeight = shape[1];
            inputWidth = shape[2];
        }
        else if (shape.Length == 4)
        {
            // [B, C, H, W]
            batch = shape[0];
            channels = shape[1];
            inputHeight = shape[2];
            inputWidth = shape[3];
        }
        else if (shape.Length >= 5)
        {
            // Flatten leading batch dimensions
            batch = 1;
            for (int d = 0; d < shape.Length - 3; d++)
                batch *= shape[d];
            channels = shape[shape.Length - 3];
            inputHeight = shape[shape.Length - 2];
            inputWidth = shape[shape.Length - 1];
        }
        else
        {
            throw new ArgumentException($"AdaptiveAveragePooling requires at least 3D input, got {shape.Length}D.");
        }

        // Cache for backward pass
        _lastInputShape = shape;
        if (IsTrainingMode)
        {
            _gpuInput = input;
            _gpuBatch = batch;
            _gpuChannels = channels;
            _gpuInputHeight = inputHeight;
            _gpuInputWidth = inputWidth;
        }

        // Allocate output buffer
        int outputSize = batch * channels * _outputHeight * _outputWidth;
        var outputBuffer = backend.AllocateBuffer(outputSize);

        // Use native GPU AdaptiveAvgPool2D operation
        backend.AdaptiveAvgPool2D(input.Buffer, outputBuffer, batch, channels, inputHeight, inputWidth, _outputHeight, _outputWidth);

        // Build output shape preserving leading dimensions
        int[] outputShape;
        if (shape.Length == 3)
        {
            outputShape = [channels, _outputHeight, _outputWidth];
        }
        else if (shape.Length == 4)
        {
            outputShape = [batch, channels, _outputHeight, _outputWidth];
        }
        else
        {
            // Restore leading dimensions
            outputShape = new int[shape.Length];
            for (int d = 0; d < shape.Length - 3; d++)
                outputShape[d] = shape[d];
            outputShape[shape.Length - 3] = channels;
            outputShape[shape.Length - 2] = _outputHeight;
            outputShape[shape.Length - 1] = _outputWidth;
        }

        return GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Resets the internal state.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastInputShape = null;

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuBatch = 0;
        _gpuChannels = 0;
        _gpuInputHeight = 0;
        _gpuInputWidth = 0;
    }
}
