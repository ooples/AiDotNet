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
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 4, 4", TestConstructorArgs = "2")]
// Roles from this layer's own guard - "requires rank>=3 [...,C,H,W]" in OnFirstForward - which reads
// Shape[rank-3], Shape[rank-2], Shape[rank-1] as channels, height and width. Batch is marked optional
// rather than declared as a second layout because the layer is tested at rank 3
// ([LayerProperty(TestInputShape = "1, 4, 4")]) and runs identically one rank up: Engine.Upsample
// scales the trailing two axes and carries every leading axis through untouched.
// OutputAxesFor below is HAND-WRITTEN, not generated: the scale comes from _scaleFactor.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class UpsamplingLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Derived from <c>OnFirstForward</c>, which resolves
    /// <c>ResolveShapes(new[] { c, h, w }, new[] { c, h * _scaleFactor, w * _scaleFactor })</c>.
    /// Channels pass through; both spatial axes are multiplied by the constructor's scale factor.
    /// </para>
    /// <para>
    /// <c>Scaled(axis, _scaleFactor)</c> and not <c>Window</c>: a window relation SHRINKS an axis, and
    /// this layer is nearest-neighbour upsampling - each input position becomes a
    /// <c>_scaleFactor</c> x <c>_scaleFactor</c> block, so the extent grows exactly by the factor with
    /// no rounding involved.
    /// </para>
    /// <para>
    /// Ranks 3 and 4 only: rank 3 is the <c>[C,H,W]</c> form the layer is tested at, rank 4 the batched
    /// one. Higher ranks run too - the guard is rank&gt;=3 - but each extra leading axis would need a
    /// DISTINCT role to be referred to by a relation, and there is no second batch-like role to give it.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (3 or 4) || _scaleFactor <= 0) return null;

        var channels = new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels));
        var height = new OutputAxisContract(
            TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, _scaleFactor));
        var width = new OutputAxisContract(
            TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, _scaleFactor));

        return inputRank == 3
            ? new[] { channels, height, width }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels, height, width,
            };
    }

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
    public UpsamplingLayer(int scaleFactor)
        : base(new[] { -1, -1, -1 }, new[] { -1, -1, -1 })
    {
        if (scaleFactor <= 0) throw new ArgumentOutOfRangeException(nameof(scaleFactor));
        _scaleFactor = scaleFactor;
    }

    /// <summary>
    /// Resolves channel/spatial dims and computes output shape on first forward.
    /// Output: [C, H*scaleFactor, W*scaleFactor].
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 3)
            throw new ArgumentException(
                $"UpsamplingLayer requires rank>=3 [...,C,H,W] input; got rank {rank}.",
                nameof(input));

        int c = input.Shape[rank - 3];
        int h = input.Shape[rank - 2];
        int w = input.Shape[rank - 1];

        ResolveShapes(new[] { c, h, w }, new[] { c, h * _scaleFactor, w * _scaleFactor });
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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)

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

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ScaleFactor"] = _scaleFactor.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
