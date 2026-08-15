using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a 3D upsampling layer that increases the spatial dimensions of volumetric data using nearest-neighbor interpolation.
/// </summary>
/// <remarks>
/// <para>
/// A 3D upsampling layer increases the spatial dimensions (depth, height, width) of volumetric tensors
/// by repeating values from the input to create a larger output. This implementation uses nearest-neighbor
/// interpolation, which copies each voxel value to fill a block in the output based on the scale factors.
/// </para>
/// <para><b>For Beginners:</b> This layer makes 3D volumes larger by simply repeating voxel values.
///
/// Think of it like zooming in on a 3D image:
/// - When you zoom in on a voxelized object, each original voxel becomes a larger block
/// - This layer does the same thing to 3D feature volumes inside the neural network
/// - It's like stretching a 3D volume without adding any new information
///
/// For example, with a scale factor of 2:
/// - A 4×4×4 volume becomes an 8×8×8 volume
/// - Each voxel in the original volume is copied to a 2×2×2 block in the output
/// - This creates a larger volume that preserves the original content but with more voxels
///
/// This is essential for 3D U-Net decoder paths, where we need to progressively increase
/// the spatial resolution to match the original input size.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Upsampling)]
[LayerTask(LayerTask.UpSampling)]
[LayerTask(LayerTask.VolumetricProcessing)]
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 4, 4, 4", TestConstructorArgs = "2")]
// Roles taken verbatim from this layer's own guard in OnFirstForward - "requires rank-4 [C,D,H,W] or
// rank-5 [B,C,D,H,W] input" - which is also the only pair of ranks it resolves shapes for. Batch is
// marked optional rather than declared as a second layout because the layer is tested at rank 4
// ([LayerProperty(TestInputShape = "1, 4, 4, 4")]) and runs the same code one rank up; ForwardTraced
// reshapes the unbatched form to [1,C,D,H,W] and reshapes the result straight back.
// OutputAxesFor below is HAND-WRITTEN, not generated: the three scale factors are constructor arguments.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class Upsample3DLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Derived from <c>OnFirstForward</c>, which resolves
    /// <c>ResolveShapes(new[] { c, d, h, w }, new[] { c, d * ScaleDepth, h * ScaleHeight, w * ScaleWidth })</c>.
    /// Channels pass through; each of the three volumetric axes is multiplied by its own scale factor,
    /// so the relations are per-axis and NOT interchangeable - the three-argument constructor exists
    /// precisely so they can differ.
    /// </para>
    /// <para>
    /// <c>Scaled</c> and not <c>Window</c>: a window relation shrinks an axis, and this is
    /// nearest-neighbour upsampling - each voxel becomes a
    /// <c>ScaleDepth</c> x <c>ScaleHeight</c> x <c>ScaleWidth</c> block, so each extent grows by exactly
    /// its factor with no rounding.
    /// </para>
    /// <para>
    /// Ranks 4 and 5 only, because those are the only ranks <c>OnFirstForward</c> accepts - it throws
    /// for anything else. <c>ForwardTraced</c> does contain a rank&gt;=6 path that folds the leading axes
    /// into the batch, but that path is unreachable through the normal entry point, and each extra
    /// leading axis would need a distinct role to be named by a relation anyway.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (4 or 5)) return null;
        if (ScaleDepth <= 0 || ScaleHeight <= 0 || ScaleWidth <= 0) return null;

        var channels = new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels));
        var depth = new OutputAxisContract(
            TensorAxis.Depth, AxisRelation.Scaled(TensorAxis.Depth, ScaleDepth));
        var height = new OutputAxisContract(
            TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, ScaleHeight));
        var width = new OutputAxisContract(
            TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, ScaleWidth));

        return inputRank == 4
            ? new[] { channels, depth, height, width }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels, depth, height, width,
            };
    }

    #region Properties

    /// <summary>
    /// Gets the scale factor for the depth dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the factor by which to increase the depth dimension.
    /// A value of 2 means the output depth will be twice the input depth.
    /// </para>
    /// </remarks>
    public int ScaleDepth { get; }

    /// <summary>
    /// Gets the scale factor for the height dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the factor by which to increase the height dimension.
    /// A value of 2 means the output height will be twice the input height.
    /// </para>
    /// </remarks>
    public int ScaleHeight { get; }

    /// <summary>
    /// Gets the scale factor for the width dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the factor by which to increase the width dimension.
    /// A value of 2 means the output width will be twice the input width.
    /// </para>
    /// </remarks>
    public int ScaleWidth { get; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, even though it has no trainable parameters, to allow gradient propagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// Although this layer does not have trainable parameters, it returns true to allow
    /// gradient propagation through the layer during backpropagation.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    /// <remarks>
    /// Upsample3D supports GPU execution via CUDA, OpenCL, and HIP backends using nearest neighbor interpolation.
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    #endregion

    #region Private Fields

    /// <summary>
    /// The input tensor from the last forward pass, cached for backward computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached GPU input shape for backward pass.
    /// </summary>
    private int[]? _gpuInputShape;

    /// <summary>
    /// Whether batch dimension was added in ForwardGpu.
    /// </summary>
    private bool _addedBatchDimension;
    private int[]? _originalInputShape;

    #endregion

    #region Constructors

    /// <summary>Construction state: the 'scaleFactor' the layer was built with.</summary>
    private readonly int _scaleFactor;

    /// <summary>
    /// Initializes a new instance of the <see cref="Upsample3DLayer{T}"/> class with uniform scaling.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [channels, depth, height, width].</param>
    /// <param name="scaleFactor">The factor by which to increase all spatial dimensions.</param>
    /// <exception cref="ArgumentException">Thrown when inputShape is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when scaleFactor is not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a 3D upsampling layer with the same scale for all dimensions.</para>
    /// <para>
    /// For example, with scaleFactor=2 and input shape [32, 8, 8, 8]:
    /// - Output shape becomes [32, 16, 16, 16]
    /// - Each voxel becomes a 2×2×2 block
    /// </para>
    /// </remarks>
    public Upsample3DLayer(int scaleFactor)
        : this(scaleFactor, scaleFactor, scaleFactor)
    {
        _scaleFactor = scaleFactor;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Upsample3DLayer{T}"/> class with separate scale factors.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [channels, depth, height, width].</param>
    /// <param name="scaleDepth">The factor by which to increase the depth dimension.</param>
    /// <param name="scaleHeight">The factor by which to increase the height dimension.</param>
    /// <param name="scaleWidth">The factor by which to increase the width dimension.</param>
    /// <exception cref="ArgumentException">Thrown when inputShape is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any scale factor is not positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a 3D upsampling layer with different scales per dimension.</para>
    /// <para>
    /// This is useful when you want non-uniform upsampling, for example:
    /// - Medical imaging where slices may have different spacing
    /// - Video data where temporal and spatial scales differ
    /// </para>
    /// </remarks>
    public Upsample3DLayer(int scaleDepth, int scaleHeight, int scaleWidth)
        : base(new[] { -1, -1, -1, -1 }, new[] { -1, -1, -1, -1 })
    {
        if (scaleDepth <= 0) throw new ArgumentOutOfRangeException(nameof(scaleDepth));
        if (scaleHeight <= 0) throw new ArgumentOutOfRangeException(nameof(scaleHeight));
        if (scaleWidth <= 0) throw new ArgumentOutOfRangeException(nameof(scaleWidth));

        ScaleDepth = scaleDepth;
        ScaleHeight = scaleHeight;
        ScaleWidth = scaleWidth;
    }

    /// <summary>
    /// Resolves channel/spatial dims and registers the resolved output shape on first forward.
    /// Output dims: [C, D*scaleDepth, H*scaleHeight, W*scaleWidth].
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        int c, d, h, w;
        if (rank == 5) { c = input.Shape[1]; d = input.Shape[2]; h = input.Shape[3]; w = input.Shape[4]; }
        else if (rank == 4) { c = input.Shape[0]; d = input.Shape[1]; h = input.Shape[2]; w = input.Shape[3]; }
        else throw new ArgumentException(
            $"Upsample3DLayer requires rank-4 [C,D,H,W] or rank-5 [B,C,D,H,W] input; got rank {rank}.",
            nameof(input));

        ResolveShapes(
            new[] { c, d, h, w },
            new[] { c, d * ScaleDepth, h * ScaleHeight, w * ScaleWidth });
    }

    #endregion

    #region Static Helper Methods

    /// <summary>
    /// Calculates the output shape based on input shape and scale factors.
    /// </summary>
    /// <param name="inputShape">The input shape [channels, depth, height, width].</param>
    /// <param name="scaleDepth">The depth scaling factor.</param>
    /// <param name="scaleHeight">The height scaling factor.</param>
    /// <param name="scaleWidth">The width scaling factor.</param>
    /// <returns>The output shape [channels, outDepth, outHeight, outWidth].</returns>
    private static int[] CalculateOutputShape(int[] inputShape, int scaleDepth, int scaleHeight, int scaleWidth)
    {
        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("Input shape must be [channels, depth, height, width].", nameof(inputShape));

        return [
            inputShape[0],
            inputShape[1] * scaleDepth,
            inputShape[2] * scaleHeight,
            inputShape[3] * scaleWidth
        ];
    }

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    /// <param name="inputShape">The input shape.</param>
    /// <param name="scaleDepth">The depth scale factor.</param>
    /// <param name="scaleHeight">The height scale factor.</param>
    /// <param name="scaleWidth">The width scale factor.</param>
    private static void ValidateParameters(int[] inputShape, int scaleDepth, int scaleHeight, int scaleWidth)
    {
        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("Input shape must be [channels, depth, height, width].", nameof(inputShape));
        if (scaleDepth <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleDepth), "Scale factor must be positive.");
        if (scaleHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleHeight), "Scale factor must be positive.");
        if (scaleWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleWidth), "Scale factor must be positive.");
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs the forward pass of the 3D upsampling layer.
    /// </summary>
    /// <param name="input">
    /// The input tensor with shape [batch, channels, depth, height, width] or [channels, depth, height, width].
    /// </param>
    /// <returns>
    /// The upsampled output tensor with increased spatial dimensions.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown when input tensor has invalid rank.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the vectorized Engine.Upsample3D operation for CPU/GPU acceleration.
    /// Each voxel in the input is replicated to fill a block of size [scaleD × scaleH × scaleW] in the output.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)
        _originalInputShape = input._shape;
        int rank = input.Rank;

        Tensor<T> batchedInput;

        if (rank == 5)
        {
            batchedInput = input;
        }
        else if (rank == 4)
        {
            batchedInput = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3] });
        }
        else if (rank >= 6)
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 4; d++)
                flatBatch *= input.Shape[d];
            batchedInput = Engine.Reshape(input, new[] { flatBatch, input.Shape[rank - 4], input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1] });
        }
        else
        {
            throw new ArgumentException(
                $"Upsample3D layer requires at least 4D tensor [C,D,H,W]. Got rank {rank}.", nameof(input));
        }

        var output = Engine.Upsample3D(batchedInput, ScaleDepth, ScaleHeight, ScaleWidth);

        // Restore original tensor rank
        if (_originalInputShape.Length > 5)
        {
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 4; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 4] = output.Shape[1];
            outputShape[_originalInputShape.Length - 3] = output.Shape[2];
            outputShape[_originalInputShape.Length - 2] = output.Shape[3];
            outputShape[_originalInputShape.Length - 1] = output.Shape[4];
            return Engine.Reshape(output, outputShape);
        }
        if (_originalInputShape.Length == 4)
        {
            return Engine.Reshape(output, new[] { output.Shape[1], output.Shape[2], output.Shape[3], output.Shape[4] });
        }

        return output;
    }

    /// <summary>
    /// Performs GPU-resident forward pass of 3D upsampling, keeping all data on GPU.
    /// </summary>
    /// <param name="inputs">The input tensors on GPU (uses first input).</param>
    /// <returns>The upsampled output as a GPU-resident tensor.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var input = inputs[0];

        // Support any rank >= 4
        if (input.Shape.Length < 4)
            throw new ArgumentException($"Upsample3D layer requires at least 4D tensor [C,D,H,W]. Got rank {input.Shape.Length}.");

        Tensor<T> input5D;
        bool addedBatch = false;
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        if (rank == 4)
        {
            addedBatch = true;
            input5D = input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3] });
        }
        else if (rank == 5)
        {
            input5D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 4; d++)
                flatBatch *= input.Shape[d];
            input5D = input.Reshape(new[] { flatBatch, input.Shape[rank - 4], input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1] });
        }

        _gpuInputShape = input5D._shape;
        _addedBatchDimension = addedBatch;

        // Store _lastInput for backward pass
        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)

        var output = gpuEngine.NearestNeighborUpsample3DGpu<T>(input5D, ScaleDepth, ScaleHeight, ScaleWidth);

        // Restore original tensor rank
        if (_originalInputShape.Length > 5)
        {
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 4; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 4] = output.Shape[1];
            outputShape[_originalInputShape.Length - 3] = output.Shape[2];
            outputShape[_originalInputShape.Length - 2] = output.Shape[3];
            outputShape[_originalInputShape.Length - 1] = output.Shape[4];
            return output.Reshape(outputShape);
        }
        if (addedBatch)
        {
            return output.Reshape(new[] { output.Shape[1], output.Shape[2], output.Shape[3], output.Shape[4] });
        }
        return output;
    }

    #endregion

    #region Backward Pass

    #endregion

    #region Parameter Management

    #endregion

    #region State Management

    /// <summary>
    /// Resets the cached state from forward/backward passes.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _gpuInputShape = null;
        _addedBatchDimension = false;
    }

    #endregion

    #region Cloning

    #endregion

    #region Serialization

    /// <summary>
    /// Creates a new Upsample3DLayer instance from serialized data.
    /// </summary>
    /// <param name="reader">The binary reader containing serialized data.</param>
    /// <returns>A new Upsample3DLayer instance with the deserialized configuration.</returns>
    /// <remarks>
    /// <para>
    /// This factory method properly deserializes Upsample3DLayer by creating a new instance
    /// with the correct scale factors and input shape from the serialized data.
    /// </para>
    /// </remarks>
    public static Upsample3DLayer<T> DeserializeFrom(BinaryReader reader)
    {
        // Read base layer data (ParameterCount + parameters)
        int paramCount = reader.ReadInt32();
        for (int i = 0; i < paramCount; i++)
        {
            reader.ReadDouble(); // Skip parameters (not used for this layer type)
        }

        // Read input shape
        int inputShapeLength = reader.ReadInt32();
        var inputShape = new int[inputShapeLength];
        for (int i = 0; i < inputShapeLength; i++)
        {
            inputShape[i] = reader.ReadInt32();
        }

        // Read scale factors
        var scaleD = reader.ReadInt32();
        var scaleH = reader.ReadInt32();
        var scaleW = reader.ReadInt32();

        return new Upsample3DLayer<T>(scaleD, scaleH, scaleW);
    }

    #endregion

    #region Computation Graph

    #endregion
}
