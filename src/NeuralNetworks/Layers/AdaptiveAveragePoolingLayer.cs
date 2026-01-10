using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
public class AdaptiveAveragePoolingLayer<T> : LayerBase<T>
{
    private readonly int _outputHeight;
    private readonly int _outputWidth;
    private readonly int _channels;

    private Tensor<T>? _lastInput;
    private int[]? _lastInputShape;

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuInput;
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
        int inputChannels,
        int inputHeight,
        int inputWidth,
        int outputHeight = 1,
        int outputWidth = 1)
        : base(
            inputShape: [inputChannels, inputHeight, inputWidth],
            outputShape: [inputChannels, outputHeight, outputWidth])
    {
        if (outputHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputHeight), "Output height must be greater than 0.");
        }
        if (outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputWidth), "Output width must be greater than 0.");
        }

        _channels = inputChannels;
        _outputHeight = outputHeight;
        _outputWidth = outputWidth;
    }

    /// <summary>
    /// Creates a global average pooling layer that pools to 1x1.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="inputHeight">The expected input height.</param>
    /// <param name="inputWidth">The expected input width.</param>
    /// <returns>An adaptive pooling layer that performs global average pooling.</returns>
    public static AdaptiveAveragePoolingLayer<T> GlobalPool(int inputChannels, int inputHeight, int inputWidth)
    {
        return new AdaptiveAveragePoolingLayer<T>(inputChannels, inputHeight, inputWidth, 1, 1);
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

        _lastInput = input;
        _lastInputShape = input.Shape;

        // Handle any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        int rank = input.Shape.Length;
        int channels = input.Shape[rank - 3];
        int inputHeight = input.Shape[rank - 2];
        int inputWidth = input.Shape[rank - 1];

        // Calculate total batch size (product of all dims except last 3)
        int batchSize = 1;
        for (int d = 0; d < rank - 3; d++)
            batchSize *= input.Shape[d];

        // Create output tensor with same leading dimensions
        int[] outputShape = new int[rank];
        for (int d = 0; d < rank - 3; d++)
            outputShape[d] = input.Shape[d];
        outputShape[rank - 3] = channels;
        outputShape[rank - 2] = _outputHeight;
        outputShape[rank - 1] = _outputWidth;

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];

        // Calculate adaptive pooling parameters for each output cell
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < _outputHeight; oh++)
                {
                    for (int ow = 0; ow < _outputWidth; ow++)
                    {
                        // Calculate input region for this output cell
                        int hStart = (int)Math.Floor((double)oh * inputHeight / _outputHeight);
                        int hEnd = (int)Math.Ceiling((double)(oh + 1) * inputHeight / _outputHeight);
                        int wStart = (int)Math.Floor((double)ow * inputWidth / _outputWidth);
                        int wEnd = (int)Math.Ceiling((double)(ow + 1) * inputWidth / _outputWidth);

                        // Average the values in the region
                        T sum = NumOps.Zero;
                        int count = 0;
                        for (int h = hStart; h < hEnd; h++)
                        {
                            for (int w = wStart; w < wEnd; w++)
                            {
                                int inputIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + h * inputWidth + w;
                                sum = NumOps.Add(sum, input.Data[inputIndex]);
                                count++;
                            }
                        }

                        T avg = NumOps.Divide(sum, NumOps.FromDouble(count));
                        int outputIndex = b * channels * _outputHeight * _outputWidth + c * _outputHeight * _outputWidth + oh * _outputWidth + ow;
                        outputData[outputIndex] = avg;
                    }
                }
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
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

        return new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the GPU-resident backward pass of adaptive average pooling.
    /// </summary>
    /// <param name="outputGradient">The GPU tensor containing the gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (_lastInputShape == null)
            throw new InvalidOperationException("ForwardGpu must be called in training mode before BackwardGpu.");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Calculate the effective pool size and stride for adaptive pooling
        // For adaptive pooling: poolSize = ceil(inputSize / outputSize), stride = floor(inputSize / outputSize)
        int poolSizeH = (int)Math.Ceiling((double)_gpuInputHeight / _outputHeight);
        int poolSizeW = (int)Math.Ceiling((double)_gpuInputWidth / _outputWidth);
        int strideH = (int)Math.Floor((double)_gpuInputHeight / _outputHeight);
        int strideW = (int)Math.Floor((double)_gpuInputWidth / _outputWidth);

        // Ensure minimum stride of 1
        strideH = Math.Max(1, strideH);
        strideW = Math.Max(1, strideW);

        // Allocate gradient buffer for input
        int inputSize = _gpuBatch * _gpuChannels * _gpuInputHeight * _gpuInputWidth;
        var gradInputBuffer = backend.AllocateBuffer(inputSize);

        // Use AvgPool2DBackward with calculated pool parameters
        backend.AvgPool2DBackward(
            outputGradient.Buffer,
            gradInputBuffer,
            _gpuBatch,
            _gpuChannels,
            _gpuInputHeight,
            _gpuInputWidth,
            _outputHeight,
            _outputWidth,
            poolSizeH,
            poolSizeW,
            strideH,
            strideW,
            0, 0,  // no padding
            true); // count includes padding

        // Build input gradient shape matching original input shape
        return new GpuTensor<T>(backend, gradInputBuffer, _lastInputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// Performs the backward pass of adaptive average pooling.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _lastInputShape is null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Handle any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        int rank = _lastInputShape.Length;
        int channels = _lastInputShape[rank - 3];
        int inputHeight = _lastInputShape[rank - 2];
        int inputWidth = _lastInputShape[rank - 1];

        // Calculate total batch size (product of all dims except last 3)
        int batchSize = 1;
        for (int d = 0; d < rank - 3; d++)
            batchSize *= _lastInputShape[d];

        // Create input gradient tensor (same shape as input)
        int inputSize = _lastInputShape.Aggregate(1, (a, b) => a * b);
        var inputGradData = new T[inputSize];

        // Distribute gradient back to input regions
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < _outputHeight; oh++)
                {
                    for (int ow = 0; ow < _outputWidth; ow++)
                    {
                        // Calculate input region for this output cell
                        int hStart = (int)Math.Floor((double)oh * inputHeight / _outputHeight);
                        int hEnd = (int)Math.Ceiling((double)(oh + 1) * inputHeight / _outputHeight);
                        int wStart = (int)Math.Floor((double)ow * inputWidth / _outputWidth);
                        int wEnd = (int)Math.Ceiling((double)(ow + 1) * inputWidth / _outputWidth);

                        int count = (hEnd - hStart) * (wEnd - wStart);

                        int outputIndex = b * channels * _outputHeight * _outputWidth + c * _outputHeight * _outputWidth + oh * _outputWidth + ow;

                        // The gradient from the average is distributed equally to all inputs
                        T gradPerInput = NumOps.Divide(outputGradient.Data[outputIndex], NumOps.FromDouble(count));

                        for (int h = hStart; h < hEnd; h++)
                        {
                            for (int w = wStart; w < wEnd; w++)
                            {
                                int inputIndex = b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + h * inputWidth + w;
                                inputGradData[inputIndex] = NumOps.Add(inputGradData[inputIndex], gradPerInput);
                            }
                        }
                    }
                }
            }
        }

        return new Tensor<T>(_lastInputShape, new Vector<T>(inputGradData));
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

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <remarks>
    /// AdaptiveAveragePoolingLayer supports JIT compilation by computing the appropriate
    /// pool size and stride to achieve the desired output dimensions, then using AvgPool2D.
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the AdaptiveAveragePoolingLayer.</returns>
    /// <remarks>
    /// <para>
    /// Adaptive average pooling is implemented by calculating the appropriate pool size
    /// and stride to achieve the target output dimensions. For global average pooling (1x1),
    /// the pool size equals the entire input spatial dimensions.
    /// </para>
    /// <para>
    /// For a given input size H_in and target output H_out:
    /// - Pool size H = ceiling(H_in / H_out)
    /// - Stride H = floor(H_in / H_out)
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        if (InputShape is null || InputShape.Length < 3)
        {
            throw new InvalidOperationException("Layer input shape not configured or invalid.");
        }

        // Get input dimensions [channels, height, width]
        int inputHeight = InputShape[1];
        int inputWidth = InputShape[2];

        // Create symbolic input node with batch dimension [1, C, H, W]
        var symbolicInput = new Tensor<T>(new int[] { 1, InputShape[0], inputHeight, inputWidth });
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Calculate adaptive pooling parameters
        // For adaptive avg pool: we need to compute pool_size and stride such that
        // the output has the desired dimensions
        //
        // Standard formula from PyTorch/TensorFlow:
        // For each dimension:
        //   stride = floor(input_size / output_size)
        //   kernel_size = input_size - (output_size - 1) * stride
        //
        // This ensures proper coverage of all input elements

        int strideH = inputHeight / _outputHeight;
        int strideW = inputWidth / _outputWidth;

        int poolH = inputHeight - (_outputHeight - 1) * strideH;
        int poolW = inputWidth - (_outputWidth - 1) * strideW;

        // For global pooling (1x1 output), pool size equals input size
        if (_outputHeight == 1 && _outputWidth == 1)
        {
            poolH = inputHeight;
            poolW = inputWidth;
            strideH = inputHeight;
            strideW = inputWidth;
        }

        // Use AvgPool2D operation
        var outputNode = Autodiff.TensorOperations<T>.AvgPool2D(
            inputNode,
            poolSize: new int[] { poolH, poolW },
            strides: new int[] { strideH, strideW });

        return outputNode;
    }
}
