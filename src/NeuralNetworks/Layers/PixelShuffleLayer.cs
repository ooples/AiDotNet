using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Pixel shuffle (sub-pixel convolution) layer for efficient spatial upsampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Pixel shuffle rearranges elements from the channel dimension into spatial dimensions,
/// effectively upscaling the spatial resolution. This is more efficient than transposed
/// convolution (deconvolution) for upsampling operations.
/// </para>
/// <para>
/// For a 2x upscaling, the layer takes 4 channel values and arranges them as a 2x2 spatial block.
/// The operation follows the formula: [batch, channels * r^2, height, width] -> [batch, channels, height * r, width * r]
/// </para>
/// <para><b>For Beginners:</b> Imagine you have a small image and want to make it bigger.
///
/// Pixel shuffle works by:
/// 1. Starting with extra channel information (4x more channels for 2x upscaling)
/// 2. Rearranging those channel values into spatial positions
/// 3. Creating a larger image with the same amount of total information
///
/// For example, with 2x upscaling:
/// - Input: 64 channels × 32×32 pixels
/// - Output: 16 channels × 64×64 pixels (same total data, different arrangement)
///
/// This is commonly used in super-resolution models like Real-ESRGAN and ESPCN.
/// </para>
/// </remarks>
public class PixelShuffleLayer<T> : LayerBase<T>
{
    #region Fields

    /// <summary>
    /// The upscaling factor for spatial dimensions.
    /// </summary>
    private readonly int _upscaleFactor;

    /// <summary>
    /// Cached input from the last forward pass for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached original input shape for backward pass with higher-rank tensors.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Cached GPU input shape for backward pass.
    /// </summary>
    private int[]? _gpuCachedInputShape;

    /// <summary>
    /// Whether a batch dimension was added for 3D input in GPU forward.
    /// </summary>
    private bool _gpuAdded3DBatch;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the upscaling factor used by this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The upscaling factor determines how much the spatial dimensions are increased.
    /// A factor of 2 doubles both width and height, a factor of 4 quadruples them.
    /// </para>
    /// </remarks>
    public int UpscaleFactor => _upscaleFactor;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Indicates whether this layer supports GPU execution.
    /// PixelShuffle uses GPU Reshape and Permute operations for efficient rearrangement.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="PixelShuffleLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor. Supports any rank >= 3.</param>
    /// <param name="upscaleFactor">The spatial upscaling factor (e.g., 2 for 2x upscaling).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a pixel shuffle layer to upscale your feature maps.
    ///
    /// The input channels must be divisible by upscaleFactor². For example:
    /// - 2x upscaling requires input channels divisible by 4
    /// - 4x upscaling requires input channels divisible by 16
    ///
    /// Example usage:
    /// <code>
    /// // Create a 2x upscaling layer for 64-channel 32×32 feature maps
    /// var pixelShuffle = new PixelShuffleLayer&lt;float&gt;(
    ///     inputShape: new[] { 64, 32, 32 },  // [channels, height, width]
    ///     upscaleFactor: 2
    /// );
    /// // Output will be [16, 64, 64] - 4x fewer channels, 4x more pixels
    /// </code>
    /// </para>
    /// </remarks>
    public PixelShuffleLayer(int[] inputShape, int upscaleFactor)
        : base(inputShape, CalculateOutputShape(inputShape, upscaleFactor))
    {
        ValidateInputShape(inputShape, upscaleFactor);
        _upscaleFactor = upscaleFactor;
    }

    #endregion

    #region Shape Calculation

    /// <summary>
    /// Calculates the output shape based on input shape and upscale factor.
    /// </summary>
    /// <param name="inputShape">The input tensor shape.</param>
    /// <param name="upscaleFactor">The upscaling factor.</param>
    /// <returns>The calculated output shape.</returns>
    private static int[] CalculateOutputShape(int[] inputShape, int upscaleFactor)
    {
        int r2 = upscaleFactor * upscaleFactor;

        return inputShape.Length switch
        {
            // 3D: [channels, height, width] -> [channels/r², height*r, width*r]
            3 => [inputShape[0] / r2, inputShape[1] * upscaleFactor, inputShape[2] * upscaleFactor],

            // 4D: [batch, channels, height, width] -> [batch, channels/r², height*r, width*r]
            4 => [inputShape[0], inputShape[1] / r2, inputShape[2] * upscaleFactor, inputShape[3] * upscaleFactor],

            // 5D: [batch, frames, channels, height, width] -> [batch, frames, channels/r², height*r, width*r]
            5 => [inputShape[0], inputShape[1], inputShape[2] / r2, inputShape[3] * upscaleFactor, inputShape[4] * upscaleFactor],

            // General case for higher dimensions: reduce channel dim, expand last two spatial dims
            _ when inputShape.Length > 5 => CalculateHighDimensionalOutputShape(inputShape, upscaleFactor),

            // Less than 3D is not supported
            _ => throw new ArgumentException($"Pixel shuffle requires at least 3 dimensions, got {inputShape.Length}.")
        };
    }

    /// <summary>
    /// Calculates output shape for tensors with more than 5 dimensions.
    /// </summary>
    private static int[] CalculateHighDimensionalOutputShape(int[] inputShape, int upscaleFactor)
    {
        int r2 = upscaleFactor * upscaleFactor;
        var result = new int[inputShape.Length];

        // Copy batch and other leading dimensions
        for (int i = 0; i < inputShape.Length - 3; i++)
        {
            result[i] = inputShape[i];
        }

        // Channel dimension (third from last) gets reduced
        int channelIdx = inputShape.Length - 3;
        result[channelIdx] = inputShape[channelIdx] / r2;

        // Height and width (last two) get expanded
        result[^2] = inputShape[^2] * upscaleFactor;
        result[^1] = inputShape[^1] * upscaleFactor;

        return result;
    }

    /// <summary>
    /// Validates that the input shape is compatible with the upscale factor.
    /// </summary>
    private static void ValidateInputShape(int[] inputShape, int upscaleFactor)
    {
        // Validate upscaleFactor to prevent division by zero and invalid shapes
        if (upscaleFactor < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(upscaleFactor),
                $"upscaleFactor must be at least 1. Got: {upscaleFactor}");
        }

        if (inputShape.Length < 3)
        {
            throw new ArgumentException(
                $"Pixel shuffle requires at least 3 dimensions (channels, height, width), got {inputShape.Length}.");
        }

        int r2 = upscaleFactor * upscaleFactor;

        // Determine channel dimension based on rank
        int channelIdx = inputShape.Length switch
        {
            3 => 0,  // [channels, height, width]
            4 => 1,  // [batch, channels, height, width]
            _ => inputShape.Length - 3  // General: third from last
        };

        int channels = inputShape[channelIdx];
        if (channels % r2 != 0)
        {
            throw new ArgumentException(
                $"Number of input channels ({channels}) must be divisible by upscaleFactor² ({r2}) for upscale factor {upscaleFactor}.");
        }
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// This method uses the GPU-accelerated IEngine.PixelShuffle operation for optimal performance.
    /// For tensors with more than 4 dimensions, it reshapes to 4D, applies the operation, and
    /// restores the original shape structure.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        var shape = input.Shape;

        // For 4D tensors (batch, channels, height, width), use Engine directly
        if (shape.Length == 4)
        {
            _lastInput = input;
            return Engine.PixelShuffle(input, _upscaleFactor);
        }

        // For 3D tensors (channels, height, width), add batch dimension
        if (shape.Length == 3)
        {
            var input4D = input.Reshape([1, shape[0], shape[1], shape[2]]);
            _lastInput = input4D;
            var output4D = Engine.PixelShuffle(input4D, _upscaleFactor);
            // Remove batch dimension from output
            return output4D.Reshape([output4D.Shape[1], output4D.Shape[2], output4D.Shape[3]]);
        }

        // For higher-rank tensors, collapse batch dimensions and use Engine
        if (shape.Length > 4)
        {
            // Collapse all leading dimensions into single batch
            int batchSize = 1;
            for (int i = 0; i < shape.Length - 3; i++)
            {
                batchSize *= shape[i];
            }

            int channels = shape[^3];
            int height = shape[^2];
            int width = shape[^1];

            var input4D = input.Reshape([batchSize, channels, height, width]);
            _lastInput = input4D;
            var output4D = Engine.PixelShuffle(input4D, _upscaleFactor);

            // Restore original batch dimensions with new spatial dimensions
            var outputShape = new int[shape.Length];
            for (int i = 0; i < shape.Length - 3; i++)
            {
                outputShape[i] = shape[i];
            }
            outputShape[^3] = output4D.Shape[1]; // new channels
            outputShape[^2] = output4D.Shape[2]; // new height
            outputShape[^1] = output4D.Shape[3]; // new width

            return output4D.Reshape(outputShape);
        }

        throw new ArgumentException($"Pixel shuffle requires at least 3 dimensions, got {shape.Length}.");
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="inputs">The GPU-resident input tensors.</param>
    /// <returns>A GPU-resident output tensor after pixel shuffle.</returns>
    /// <remarks>
    /// <para>
    /// Pixel shuffle is implemented as: Reshape -> Permute -> Reshape
    /// [N, C*r², H, W] -> [N, C, r, r, H, W] -> [N, C, H, r, W, r] -> [N, C, H*r, W*r]
    /// All operations stay GPU-resident.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input.Shape;
        int r = _upscaleFactor;
        int r2 = r * r;

        // Handle different input ranks
        int batch, channels, height, width;
        bool added3DBatch = false;

        if (shape.Length == 4)
        {
            batch = shape[0];
            channels = shape[1];
            height = shape[2];
            width = shape[3];
        }
        else if (shape.Length == 3)
        {
            batch = 1;
            channels = shape[0];
            height = shape[1];
            width = shape[2];
            added3DBatch = true;
            input = gpuEngine.ReshapeGpu(input, new[] { 1, channels, height, width });
        }
        else
        {
            throw new ArgumentException($"PixelShuffle requires 3D or 4D input, got {shape.Length}D.");
        }

        // Cache for backward pass
        if (IsTrainingMode)
        {
            _gpuCachedInputShape = (int[])shape.Clone();
            _gpuAdded3DBatch = added3DBatch;
        }

        int outChannels = channels / r2;

        // Step 1: Reshape [N, C*r², H, W] -> [N, C, r, r, H, W]
        var reshaped1 = gpuEngine.ReshapeGpu(input, new[] { batch, outChannels, r, r, height, width });

        // Step 2: Permute [N, C, r, r, H, W] -> [N, C, H, r, W, r]
        var permuted = gpuEngine.PermuteGpu(reshaped1, new[] { 0, 1, 4, 2, 5, 3 });

        // Step 3: Reshape [N, C, H, r, W, r] -> [N, C, H*r, W*r]
        int outHeight = height * r;
        int outWidth = width * r;
        var result = gpuEngine.ReshapeGpu(permuted, new[] { batch, outChannels, outHeight, outWidth });

        // Remove batch dimension if we added it
        if (added3DBatch)
        {
            result = gpuEngine.ReshapeGpu(result, new[] { outChannels, outHeight, outWidth });
        }

        return result;
    }

    #endregion

    #region Backward Pass

    /// <summary>
    /// Performs the backward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="outputGradient">The gradient from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass of pixel shuffle (pixel unshuffle) reverses the forward:
    /// Reshape [N, C, H*r, W*r] -> [N, C, H, r, W, r] -> Permute (inverse) -> [N, C, r, r, H, W] -> Reshape [N, C*r², H, W]
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_gpuCachedInputShape == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var gradShape = outputGradient.Shape;
        int r = _upscaleFactor;
        int r2 = r * r;

        // Get dimensions from cached input shape
        int batch, channels, height, width;
        var grad = outputGradient;

        if (_gpuAdded3DBatch)
        {
            // Original was 3D, add batch dimension to gradient
            batch = 1;
            channels = gradShape[0] * r2; // output had reduced channels
            int outHeight = gradShape[1];
            int outWidth = gradShape[2];
            height = outHeight / r;
            width = outWidth / r;
            grad = gpuEngine.ReshapeGpu(outputGradient, new[] { 1, gradShape[0], outHeight, outWidth });
        }
        else
        {
            batch = gradShape[0];
            channels = gradShape[1] * r2; // output had reduced channels
            int outHeight = gradShape[2];
            int outWidth = gradShape[3];
            height = outHeight / r;
            width = outWidth / r;
        }

        int outChannels = channels / r2; // = gradShape[1] or gradShape[0] for 3D
        int outHeight2 = height * r;
        int outWidth2 = width * r;

        // Step 1: Reshape [N, C, H*r, W*r] -> [N, C, H, r, W, r]
        var reshaped1 = gpuEngine.ReshapeGpu(grad, new[] { batch, outChannels, height, r, width, r });

        // Step 2: Permute [N, C, H, r, W, r] -> [N, C, r, r, H, W]
        // Original forward permutation was [0, 1, 4, 2, 5, 3]
        // Inverse permutation: [0, 1, 3, 5, 2, 4]
        var permuted = gpuEngine.PermuteGpu(reshaped1, new[] { 0, 1, 3, 5, 2, 4 });

        // Step 3: Reshape [N, C, r, r, H, W] -> [N, C*r², H, W]
        var result = gpuEngine.ReshapeGpu(permuted, new[] { batch, channels, height, width });

        // Remove batch dimension if we added it
        if (_gpuAdded3DBatch)
        {
            result = gpuEngine.ReshapeGpu(result, new[] { channels, height, width });
        }

        return result;
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Uses the GPU-accelerated IEngine.PixelShuffleBackward operation for optimal performance.
    /// This reverses the pixel shuffle operation for backpropagation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _originalInputShape == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        var inShape = _lastInput.Shape;

        // For 4D tensors, use Engine directly
        if (_originalInputShape.Length == 4)
        {
            return Engine.PixelShuffleBackward(outputGradient, inShape, _upscaleFactor);
        }

        // For 3D tensors, add batch dimension
        if (_originalInputShape.Length == 3)
        {
            var grad4D = outputGradient.Reshape([1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2]]);
            var inputGrad4D = Engine.PixelShuffleBackward(grad4D, inShape, _upscaleFactor);
            return inputGrad4D.Reshape([inputGrad4D.Shape[1], inputGrad4D.Shape[2], inputGrad4D.Shape[3]]);
        }

        // For higher-rank tensors, collapse batch dimensions
        if (_originalInputShape.Length > 4)
        {
            // Collapse all leading dimensions into single batch
            int batchSize = 1;
            for (int i = 0; i < _originalInputShape.Length - 3; i++)
            {
                batchSize *= _originalInputShape[i];
            }

            var grad4D = outputGradient.Reshape([batchSize, outputGradient.Shape[^3], outputGradient.Shape[^2], outputGradient.Shape[^1]]);
            var inputGrad4D = Engine.PixelShuffleBackward(grad4D, inShape, _upscaleFactor);

            // Restore original batch dimensions
            return inputGrad4D.Reshape(_originalInputShape);
        }

        throw new ArgumentException($"Pixel shuffle requires at least 3 dimensions, got {_originalInputShape.Length}.");
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update - this is a purely structural layer
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _originalInputShape = null;
        _gpuCachedInputShape = null;
        _gpuAdded3DBatch = false;
    }

    #endregion

    #region JIT Compilation

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Exports this layer's computation as a differentiable computation graph for JIT compilation.
    /// The pixel shuffle operation is supported in the computation graph via TensorOperations.PixelShuffle.
    /// </para>
    /// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation creates an optimized version of this
    /// layer that can run faster during inference. Once compiled, the computation graph can be executed
    /// as native code rather than interpreted operations.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension
        // Input shape: [batch, channels, height, width] (NCHW format)
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "pixelshuffle_input");
        inputNodes.Add(inputNode);

        // Apply PixelShuffle operation from TensorOperations
        var shuffled = TensorOperations<T>.PixelShuffle(inputNode, _upscaleFactor);

        return shuffled;
    }

    #endregion
}
