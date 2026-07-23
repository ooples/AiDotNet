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
public class AdaptiveAveragePoolingLayer<T> : LayerBase<T>
{
    private readonly int _outputHeight;
    private readonly int _outputWidth;
    private int _channels;

    private Tensor<T>? _lastInput;
    private int[]? _lastInputShape;

    // GPU cached tensors for backward pass
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
    public override Tensor<T> Forward(Tensor<T> input)
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
        // Irregular case (non-divisible): the sliding-window region-mean has no
        // single Engine op and the scalar fallback below builds a raw new Tensor
        // with no tape connection — backward through that boundary returns null.
        // The fallback is documented as not tape-tracked; callers needing
        // backprop must use H/W that divide evenly into outH/outW.
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

        // Irregular fallback (input H/W not divisible by output H/W). The
        // PyTorch adaptive_avg_pool2d contract uses per-cell windows
        //   h_start = floor(oh * H / outH),  h_end = ceil((oh+1) * H / outH)
        //   w_start = floor(ow * W / outW),  w_end = ceil((ow+1) * W / outW)
        // We build the output by tape-tracked TensorSlice + ReduceMean over
        // each (oh, ow) window so gradients propagate correctly through the
        // pooling boundary even for non-divisible shapes. The slice/mean pair
        // is O(outH × outW) per (batch, channel) tape ops; that's a one-time
        // graph-build cost and the tape can JIT-fuse them, so it's
        // significantly cheaper than the previous raw-tensor copy that
        // silently dropped gradients.
        int hAxis = rank - 2;
        int wAxis = rank - 1;

        // Per-output-row mean: for each oh, slice the input rows
        // [hStart:hEnd] along hAxis, mean over hAxis (keepDims=true so the
        // result has 1 row), then collect rows. Same for the W axis.
        var rowOutputs = new Tensor<T>[_outputHeight];
        for (int oh = 0; oh < _outputHeight; oh++)
        {
            int hStart = (int)Math.Floor((double)oh * inputHeight / _outputHeight);
            int hEnd = (int)Math.Ceiling((double)(oh + 1) * inputHeight / _outputHeight);
            int hLen = hEnd - hStart;

            int[] rowStart = new int[rank];
            int[] rowLen = new int[rank];
            for (int d = 0; d < rank; d++)
            {
                rowStart[d] = 0;
                rowLen[d] = input.Shape[d];
            }
            rowStart[hAxis] = hStart;
            rowLen[hAxis] = hLen;
            var rowSlab = Engine.TensorSlice(input, rowStart, rowLen);
            var rowMean = Engine.ReduceMean(rowSlab, new[] { hAxis }, keepDims: true);

            // Now reduce W axis per-output-column.
            var colOutputs = new Tensor<T>[_outputWidth];
            for (int ow = 0; ow < _outputWidth; ow++)
            {
                int wStart = (int)Math.Floor((double)ow * inputWidth / _outputWidth);
                int wEnd = (int)Math.Ceiling((double)(ow + 1) * inputWidth / _outputWidth);
                int wLen = wEnd - wStart;

                int[] colStart = new int[rank];
                int[] colLen = new int[rank];
                for (int d = 0; d < rank; d++)
                {
                    colStart[d] = 0;
                    colLen[d] = rowMean.Shape[d];
                }
                colStart[wAxis] = wStart;
                colLen[wAxis] = wLen;
                var colSlab = Engine.TensorSlice(rowMean, colStart, colLen);
                colOutputs[ow] = Engine.ReduceMean(colSlab, new[] { wAxis }, keepDims: true);
            }

            // Concatenate the per-column means along W to form the row.
            rowOutputs[oh] = Engine.TensorConcatenate(colOutputs, axis: wAxis);
        }

        // Concatenate the per-row outputs along H to form the final output.
        return Engine.TensorConcatenate(rowOutputs, axis: hAxis);
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
    /// Updates the parameters. Pooling layers have no trainable parameters.
    /// </summary>
    /// <param name="learningRate">The learning rate (unused).</param>
    public override void UpdateParameters(T learningRate)
    {
        // No trainable parameters
    }

    /// <summary>
    /// Gets all trainable parameters. Returns empty for pooling layers.
    /// </summary>
    /// <returns>An empty vector.</returns>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
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
