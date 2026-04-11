using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an upsampling layer that increases the spatial dimensions of input tensors using nearest-neighbor interpolation.
/// </summary>
/// <remarks>
/// <para>
/// An upsampling layer increases the spatial dimensions (height and width) of input tensors by repeating values from
/// the input to create a larger output. This implementation uses nearest-neighbor interpolation, which repeats each
/// value in the input tensor multiple times based on the scale factor to create the upsampled output.
/// </para>
/// <para><b>For Beginners:</b> This layer makes images or feature maps larger by simply repeating pixels.
/// 
/// Think of it like zooming in on a digital image:
/// - When you zoom in on a pixelated image, each original pixel becomes a larger square
/// - This layer does the same thing to feature maps inside the neural network
/// - It's like stretching an image without adding any new information
/// 
/// For example, with a scale factor of 2:
/// - A 4Ã—4 image becomes an 8Ã—8 image
/// - Each pixel in the original image is copied to a 2Ã—2 block in the output
/// - This creates a larger image that preserves the original content but with more pixels
/// 
/// This is useful for tasks like image generation or upscaling, where you need to increase
/// the resolution of features that the network has processed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Upsampling)]
[LayerTask(LayerTask.UpSampling)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 4, 4", TestConstructorArgs = "new[] { 1, 4, 4 }, 2")]
public class UpsamplingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The factor by which to increase spatial dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the scale factor used to increase the spatial dimensions (height and width) of the input.
    /// A value of 2 means the output height and width will be twice the input dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how much larger the output will be compared to the input.
    /// 
    /// For example:
    /// - With a scale factor of 2: A 10Ã—10 image becomes 20Ã—20
    /// - With a scale factor of 3: A 10Ã—10 image becomes 30Ã—30
    /// - With a scale factor of 4: A 10Ã—10 image becomes 40Ã—40
    /// 
    /// The scale factor applies equally to both height and width, so the total number of pixels
    /// increases by the square of the scale factor (e.g., a scale factor of 2 means 4 times more pixels).
    /// </para>
    /// </remarks>
    private readonly int _scaleFactor;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what it last processed.
    ///
    /// Storing the input is necessary because:
    /// - During training, the layer needs to remember what input it processed
    /// - This helps calculate the correct gradients during the backward pass
    /// - It's part of the layer's "working memory" for the learning process
    ///
    /// This cached input helps the layer understand how to adjust the network's behavior
    /// to improve its performance on future inputs.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached input shape from GPU forward pass for backward pass.
    /// </summary>
    private int[]? _gpuCachedInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, even though it has no trainable parameters, to allow gradient propagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the upsampling layer can be included in the training process. Although this layer
    /// does not have trainable parameters, it returns true to allow gradient propagation through the layer during backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer participates in the learning process.
    /// 
    /// A value of true means:
    /// - The layer is part of the training process
    /// - It can pass gradients backward to previous layers
    /// - It helps the network learn, even though it doesn't have its own parameters to adjust
    /// 
    /// This is like being a messenger that relays feedback to earlier parts of the network,
    /// even though the messenger doesn't change its own behavior.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="UpsamplingLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="scaleFactor">The factor by which to increase spatial dimensions.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an upsampling layer with the specified input shape and scale factor. The output shape
    /// is calculated based on the input shape and scale factor.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new upsampling layer.
    /// 
    /// The parameters you provide determine:
    /// - inputShape: The dimensions of the data coming into this layer
    /// - scaleFactor: How much larger the output should be compared to the input
    /// 
    /// For example, if inputShape is [3, 32, 32] (representing 3 channels of a 32Ã—32 image)
    /// and scaleFactor is 2, the output shape will be [3, 64, 64] - the same number of
    /// channels but twice the height and width.
    /// </para>
    /// </remarks>
    public UpsamplingLayer(int[] inputShape, int scaleFactor)
        : base(inputShape, CalculateOutputShape(inputShape, scaleFactor))
    {
        _scaleFactor = scaleFactor;
    }

    /// <summary>
    /// Calculates the output shape based on input shape and scale factor.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="scaleFactor">The factor by which to increase spatial dimensions.</param>
    /// <returns>The calculated output shape.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the upsampling layer by multiplying the height and width dimensions
    /// of the input shape by the scale factor, while keeping the number of channels the same.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of data that will come out of this layer.
    /// 
    /// It works by:
    /// - Taking the input shape
    /// - Keeping the channel dimension (first element) the same
    /// - Multiplying the height and width (second and third elements) by the scale factor
    /// 
    /// For example, if the input shape is [16, 20, 30] (16 channels, 20 height, 30 width)
    /// and the scale factor is 2, the output shape will be [16, 40, 60].
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int scaleFactor)
    {
        // Industry-standard: support tensors of any rank
        // The last two dimensions are always height and width for upsampling
        // Supports: 2D [H, W], 3D [C, H, W], 4D [B, C, H, W], etc.
        if (inputShape.Length < 2)
            throw new ArgumentException("Input shape must have at least 2 dimensions for upsampling.");

        var outputShape = new int[inputShape.Length];

        // Copy all dimensions except the last two
        for (int i = 0; i < inputShape.Length - 2; i++)
        {
            outputShape[i] = inputShape[i];
        }

        // Scale the last two dimensions (height and width)
        int heightIdx = inputShape.Length - 2;
        int widthIdx = inputShape.Length - 1;
        outputShape[heightIdx] = inputShape[heightIdx] * scaleFactor;
        outputShape[widthIdx] = inputShape[widthIdx] * scaleFactor;

        return outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the upsampling layer.
    /// </summary>
    /// <param name="input">The input tensor to upsample.</param>
    /// <returns>The upsampled output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the upsampling layer using nearest-neighbor interpolation. It repeats
    /// each value in the input tensor according to the scale factor to create a larger output tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a larger version of the input by repeating values.
    /// 
    /// During the forward pass:
    /// 1. The layer receives an input tensor (like a stack of feature maps)
    /// 2. For each value in the input:
    ///    - The value is copied multiple times based on the scale factor
    ///    - These copies form a block in the output tensor
    /// 3. This creates an output that is larger but contains the same information
    /// 
    /// For example, with a scale factor of 2, each pixel becomes a 2Ã—2 block of identical pixels.
    /// This is the simplest form of upsampling, which preserves the original content
    /// but increases the spatial dimensions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        return Engine.Upsample(input, _scaleFactor, _scaleFactor);
    }

    /// <summary>
    /// Performs the forward pass on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after upsampling.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensor is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is unavailable.</exception>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];

        // Cache input shape for backward pass during training
        if (IsTrainingMode)
        {
            _gpuCachedInputShape = (int[])input._shape.Clone();
        }

        return gpuEngine.UpsampleGpu(input, _scaleFactor);
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is empty as the upsampling layer does not have any trainable parameters to update.
    /// It is included to conform to the base class interface.
    /// </para>
    /// <para><b>For Beginners:</b> This method does nothing because this layer has no learnable values.
    /// 
    /// The upsampling layer:
    /// - Performs a fixed, predefined operation (repeating values)
    /// - Has no weights or biases to adjust during training
    /// - Only passes gradients backward without changing itself
    /// 
    /// This method exists only to satisfy the requirements of the base layer class,
    /// similar to how a purely functional node in a network would need to implement
    /// this method even though it has nothing to update.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>An empty vector, as the layer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector as the upsampling layer does not have any trainable parameters.
    /// It is included to conform to the base class interface.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because this layer has no learnable values.
    /// 
    /// Since the upsampling layer:
    /// - Has no weights or biases
    /// - Performs a fixed operation that doesn't need to be learned
    /// - Only transforms the input according to predefined rules
    /// 
    /// It returns an empty vector, indicating to the optimization process that
    /// there are no parameters to update for this layer.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // This layer doesn't have any trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the upsampling layer by clearing the cached input tensor.
    /// This is useful when starting to process a new, unrelated input.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory of what it last processed.
    ///
    /// When resetting the state:
    /// - The layer forgets what input it recently processed
    /// - This helps prepare it for processing new, unrelated inputs
    /// - It's like clearing a workspace before starting a new project
    ///
    /// This is mostly important during training, where the layer needs to
    /// maintain consistency between forward and backward passes.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear the cached input
        _lastInput = null;
        _gpuCachedInputShape = null;
    }
}
