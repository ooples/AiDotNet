using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;

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
public class AdaptiveAvgPoolingLayer<T> : LayerBase<T>
{
    private readonly int _outputHeight;
    private readonly int _outputWidth;
    private readonly int _channels;

    private Tensor<T>? _lastInput;
    private int[]? _lastInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// Pooling layers don't have trainable parameters, but they support backpropagation.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdaptiveAvgPoolingLayer{T}"/> class.
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
    public AdaptiveAvgPoolingLayer(
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
    public static AdaptiveAvgPoolingLayer<T> GlobalPool(int inputChannels, int inputHeight, int inputWidth)
    {
        return new AdaptiveAvgPoolingLayer<T>(inputChannels, inputHeight, inputWidth, 1, 1);
    }

    /// <summary>
    /// Performs the forward pass of adaptive average pooling.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The pooled output tensor [C, outH, outW] or [B, C, outH, outW].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        _lastInputShape = input.Shape;

        // Handle both 3D [C, H, W] and 4D [B, C, H, W] inputs
        bool is4D = input.Shape.Length == 4;
        int batchSize = is4D ? input.Shape[0] : 1;
        int channels = is4D ? input.Shape[1] : input.Shape[0];
        int inputHeight = is4D ? input.Shape[2] : input.Shape[1];
        int inputWidth = is4D ? input.Shape[3] : input.Shape[2];

        // Create output tensor
        int[] outputShape = is4D
            ? [batchSize, channels, _outputHeight, _outputWidth]
            : [channels, _outputHeight, _outputWidth];

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
                                int inputIndex = is4D
                                    ? b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + h * inputWidth + w
                                    : c * inputHeight * inputWidth + h * inputWidth + w;
                                sum = NumOps.Add(sum, input.Data[inputIndex]);
                                count++;
                            }
                        }

                        T avg = NumOps.Divide(sum, NumOps.FromDouble(count));
                        int outputIndex = is4D
                            ? b * channels * _outputHeight * _outputWidth + c * _outputHeight * _outputWidth + oh * _outputWidth + ow
                            : c * _outputHeight * _outputWidth + oh * _outputWidth + ow;
                        outputData[outputIndex] = avg;
                    }
                }
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
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

        bool is4D = _lastInputShape.Length == 4;
        int batchSize = is4D ? _lastInputShape[0] : 1;
        int channels = is4D ? _lastInputShape[1] : _lastInputShape[0];
        int inputHeight = is4D ? _lastInputShape[2] : _lastInputShape[1];
        int inputWidth = is4D ? _lastInputShape[3] : _lastInputShape[2];

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

                        int outputIndex = is4D
                            ? b * channels * _outputHeight * _outputWidth + c * _outputHeight * _outputWidth + oh * _outputWidth + ow
                            : c * _outputHeight * _outputWidth + oh * _outputWidth + ow;

                        // The gradient from the average is distributed equally to all inputs
                        T gradPerInput = NumOps.Divide(outputGradient.Data[outputIndex], NumOps.FromDouble(count));

                        for (int h = hStart; h < hEnd; h++)
                        {
                            for (int w = wStart; w < wEnd; w++)
                            {
                                int inputIndex = is4D
                                    ? b * channels * inputHeight * inputWidth + c * inputHeight * inputWidth + h * inputWidth + w
                                    : c * inputHeight * inputWidth + h * inputWidth + w;
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
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <remarks>
    /// AdaptiveAvgPoolingLayer supports JIT compilation by computing the appropriate
    /// pool size and stride to achieve the desired output dimensions, then using AvgPool2D.
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the AdaptiveAvgPoolingLayer.</returns>
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
