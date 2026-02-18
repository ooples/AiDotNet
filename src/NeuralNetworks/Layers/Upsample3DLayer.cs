using AiDotNet.Autodiff;
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
public class Upsample3DLayer<T> : LayerBase<T>
{
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

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value><c>true</c> if the input shape is configured.</value>
    public override bool SupportsJitCompilation => InputShape != null && InputShape.Length > 0;

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <value>Always 0 as this layer has no trainable parameters.</value>
    public override int ParameterCount => 0;

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
    public Upsample3DLayer(int[] inputShape, int scaleFactor)
        : this(inputShape, scaleFactor, scaleFactor, scaleFactor)
    {
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
    public Upsample3DLayer(int[] inputShape, int scaleDepth, int scaleHeight, int scaleWidth)
        : base(inputShape, CalculateOutputShape(inputShape, scaleDepth, scaleHeight, scaleWidth))
    {
        ValidateParameters(inputShape, scaleDepth, scaleHeight, scaleWidth);

        ScaleDepth = scaleDepth;
        ScaleHeight = scaleHeight;
        ScaleWidth = scaleWidth;
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
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        _originalInputShape = input.Shape;
        int rank = input.Rank;

        Tensor<T> batchedInput;

        if (rank == 5)
        {
            batchedInput = input;
        }
        else if (rank == 4)
        {
            batchedInput = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]);
        }
        else if (rank >= 6)
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 4; d++)
                flatBatch *= input.Shape[d];
            batchedInput = input.Reshape(flatBatch, input.Shape[rank - 4], input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]);
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
            return output.Reshape(outputShape);
        }
        if (_originalInputShape.Length == 4)
        {
            return output.Reshape(output.Shape[1], output.Shape[2], output.Shape[3], output.Shape[4]);
        }

        return output;
    }

    /// <summary>
    /// Performs GPU-resident forward pass of 3D upsampling, keeping all data on GPU.
    /// </summary>
    /// <param name="inputs">The input tensors on GPU (uses first input).</param>
    /// <returns>The upsampled output as a GPU-resident tensor.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var input = inputs[0];

        // Support any rank >= 4
        if (input.Shape.Length < 4)
            throw new ArgumentException($"Upsample3D layer requires at least 4D tensor [C,D,H,W]. Got rank {input.Shape.Length}.");

        IGpuTensor<T> input5D;
        bool addedBatch = false;
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        if (rank == 4)
        {
            addedBatch = true;
            input5D = input.CreateView(0, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3] });
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
            input5D = input.CreateView(0, new[] { flatBatch, input.Shape[rank - 4], input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1] });
        }

        _gpuInputShape = input5D.Shape;
        _addedBatchDimension = addedBatch;

        // Store _lastInput for backward pass
        _lastInput = input.ToTensor();

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
            return output.CreateView(0, outputShape);
        }
        if (addedBatch)
        {
            return output.CreateView(0, new[] { output.Shape[1], output.Shape[2], output.Shape[3], output.Shape[4] });
        }
        return output;
    }

    /// <summary>
    /// Performs GPU-resident backward pass of 3D upsampling.
    /// </summary>
    /// <param name="outputGradient">The gradient of the output on GPU.</param>
    /// <returns>The gradient with respect to input as a GPU-resident tensor.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine");

        if (_gpuInputShape == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu");

        // Flatten gradient to 5D the same way forward flattened input
        int rank = outputGradient.Shape.Length;
        IGpuTensor<T> gradient5D;

        if (rank == 4)
        {
            gradient5D = outputGradient.CreateView(0, new[] { 1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2], outputGradient.Shape[3] });
        }
        else if (rank == 5)
        {
            gradient5D = outputGradient;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 4; d++)
                flatBatch *= outputGradient.Shape[d];
            gradient5D = outputGradient.CreateView(0, new[] { flatBatch, outputGradient.Shape[rank - 4], outputGradient.Shape[rank - 3], outputGradient.Shape[rank - 2], outputGradient.Shape[rank - 1] });
        }

        var inputGrad = gpuEngine.NearestNeighborUpsample3DBackwardGpu<T>(gradient5D, _gpuInputShape, ScaleDepth, ScaleHeight, ScaleWidth);

        // Restore to original input shape
        if (_originalInputShape != null && _originalInputShape.Length > 5)
        {
            var restoreShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 4; d++)
                restoreShape[d] = _originalInputShape[d];
            restoreShape[_originalInputShape.Length - 4] = inputGrad.Shape[1];
            restoreShape[_originalInputShape.Length - 3] = inputGrad.Shape[2];
            restoreShape[_originalInputShape.Length - 2] = inputGrad.Shape[3];
            restoreShape[_originalInputShape.Length - 1] = inputGrad.Shape[4];
            return inputGrad.CreateView(0, restoreShape);
        }
        if (_addedBatchDimension)
        {
            return inputGrad.CreateView(0, new[] { inputGrad.Shape[1], inputGrad.Shape[2], inputGrad.Shape[3], inputGrad.Shape[4] });
        }
        return inputGrad;
    }

    #endregion

    #region Backward Pass

    /// <summary>
    /// Performs the backward pass to propagate gradients through the upsampling.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called.</exception>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients from each output voxel in a [scaleD × scaleH × scaleW] block
    /// are summed and assigned to the corresponding input voxel.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int rank = outputGradient.Rank;
        Tensor<T> batchedGradient;
        int[] inputShape;

        if (rank == 5)
        {
            batchedGradient = outputGradient;
            inputShape = _lastInput.Shape.Length == 5
                ? _lastInput.Shape
                : new[] { 1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2], _lastInput.Shape[3] };
        }
        else if (rank == 4)
        {
            batchedGradient = outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2], outputGradient.Shape[3]);
            inputShape = new[] { 1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2], _lastInput.Shape[3] };
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 4; d++)
                flatBatch *= outputGradient.Shape[d];
            batchedGradient = outputGradient.Reshape(flatBatch, outputGradient.Shape[rank - 4], outputGradient.Shape[rank - 3], outputGradient.Shape[rank - 2], outputGradient.Shape[rank - 1]);

            int inputFlatBatch = 1;
            for (int d = 0; d < _lastInput.Shape.Length - 4; d++)
                inputFlatBatch *= _lastInput.Shape[d];
            inputShape = new[] { inputFlatBatch, _lastInput.Shape[_lastInput.Shape.Length - 4], _lastInput.Shape[_lastInput.Shape.Length - 3], _lastInput.Shape[_lastInput.Shape.Length - 2], _lastInput.Shape[_lastInput.Shape.Length - 1] };
        }

        var inputGrad = Engine.Upsample3DBackward(batchedGradient, inputShape, ScaleDepth, ScaleHeight, ScaleWidth);

        // Restore to original input shape
        return inputGrad.Reshape(_lastInput.Shape);
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates parameters. This layer has no trainable parameters.
    /// </summary>
    /// <param name="learningRate">The learning rate (unused).</param>
    public override void UpdateParameters(T learningRate)
    {
        // No trainable parameters to update
    }

    /// <summary>
    /// Gets all trainable parameters. This layer has none.
    /// </summary>
    /// <returns>An empty vector.</returns>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Sets parameters from a vector. This layer has no trainable parameters.
    /// </summary>
    /// <param name="parameters">Parameter vector (should be empty).</param>
    public override void SetParameters(Vector<T> parameters)
    {
        // No parameters to set
    }

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

    /// <summary>
    /// Creates a deep copy of the layer with the same configuration.
    /// </summary>
    /// <returns>A new instance of the <see cref="Upsample3DLayer{T}"/> with identical configuration.</returns>
    public override LayerBase<T> Clone()
    {
        return new Upsample3DLayer<T>(InputShape, ScaleDepth, ScaleHeight, ScaleWidth);
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

        // Write input shape for proper deserialization
        writer.Write(InputShape.Length);
        foreach (var dim in InputShape)
        {
            writer.Write(dim);
        }

        writer.Write(ScaleDepth);
        writer.Write(ScaleHeight);
        writer.Write(ScaleWidth);
    }

    /// <summary>
    /// Deserializes the layer from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="InvalidOperationException">Thrown because Upsample3DLayer uses readonly properties and cannot be deserialized in-place.</exception>
    /// <remarks>
    /// <para>
    /// This method validates that the serialized scale factors match the current instance's values.
    /// For full deserialization support, use the static factory method <see cref="DeserializeFrom"/> instead.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);

        // Read input shape
        int inputShapeLength = reader.ReadInt32();
        var inputShape = new int[inputShapeLength];
        for (int i = 0; i < inputShapeLength; i++)
        {
            inputShape[i] = reader.ReadInt32();
        }

        var scaleD = reader.ReadInt32();
        var scaleH = reader.ReadInt32();
        var scaleW = reader.ReadInt32();

        // Validate serialized values match current instance (readonly properties cannot be changed)
        if (scaleD != ScaleDepth || scaleH != ScaleHeight || scaleW != ScaleWidth)
        {
            throw new InvalidOperationException(
                $"Deserialized scale factors [{scaleD}, {scaleH}, {scaleW}] do not match current instance " +
                $"[{ScaleDepth}, {ScaleHeight}, {ScaleWidth}]. Use DeserializeFrom factory method instead.");
        }
    }

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

        return new Upsample3DLayer<T>(inputShape, scaleD, scaleH, scaleW);
    }

    #endregion

    #region Computation Graph

    /// <summary>
    /// Exports the layer's computation as a graph node for JIT compilation or autodiff.
    /// </summary>
    /// <param name="inputNodes">List to append input nodes to.</param>
    /// <returns>A computation node representing the upsampling operation.</returns>
    /// <exception cref="NotSupportedException">Thrown because Upsample3D autodiff is not yet implemented.</exception>
    /// <remarks>
    /// <para>
    /// This method is not yet implemented. The TensorOperations.Upsample3D method needs to be added
    /// to support JIT compilation and automatic differentiation for 3D upsampling operations.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // TODO: Implement TensorOperations.Upsample3D for autodiff support
        throw new NotSupportedException(
            "Upsample3DLayer.ExportComputationGraph is not yet implemented. " +
            "TensorOperations.Upsample3D needs to be added for JIT compilation and autodiff support.");
    }

    #endregion
}
