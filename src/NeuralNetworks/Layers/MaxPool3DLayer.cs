using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a 3D max pooling layer for downsampling volumetric data.
/// </summary>
/// <remarks>
/// <para>
/// A 3D max pooling layer reduces the spatial dimensions (depth, height, width) of volumetric
/// data while preserving the most prominent features. This helps reduce computational cost
/// and provides translation invariance.
/// </para>
/// <para><b>For Beginners:</b> Max pooling works like summarizing a 3D region by keeping only
/// the largest value.
///
/// Think of it like this:
/// - You have a 3D grid of numbers
/// - You divide it into small cubes (e.g., 2x2x2)
/// - For each cube, you keep only the largest number
/// - This makes your data smaller while keeping the important features
///
/// This is useful because:
/// - It reduces the amount of computation needed
/// - It helps the network focus on the most important features
/// - It makes the network more robust to small position changes
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MaxPool3DLayer<T> : LayerBase<T>
{
    #region Properties

    /// <summary>
    /// Gets the size of the pooling window (same for depth, height, width).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The pool size determines the size of the region from which to extract the maximum value.
    /// Common values are 2 (halves each spatial dimension) or 3.
    /// </para>
    /// </remarks>
    public int PoolSize { get; private set; }

    /// <summary>
    /// Gets the stride (step size) for moving the pooling window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stride controls how much the pooling window moves between positions.
    /// A stride equal to pool size produces non-overlapping regions.
    /// A stride of 1 with pool size > 1 produces overlapping regions.
    /// </para>
    /// </remarks>
    public int Stride { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training (backpropagation).
    /// </summary>
    /// <value>Always <c>true</c> as MaxPool3D supports gradient flow through max indices.</value>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value><c>true</c> if the input shape is configured.</value>
    public override bool SupportsJitCompilation => InputShape != null && InputShape.Length > 0;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    /// <remarks>
    /// MaxPool3D supports GPU execution via CUDA, OpenCL, and HIP backends.
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    #endregion

    #region Private Fields

    /// <summary>
    /// Stores the indices of maximum values from the forward pass for gradient routing.
    /// Shape: [batch, channels, outD, outH, outW, 3] where the last dimension stores [d, h, w] indices.
    /// </summary>
    private int[,,,,,]? _maxIndices;

    /// <summary>
    /// Cached input from the last forward pass.
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

    /// <summary>
    /// GPU buffer containing pooling indices for backward pass.
    /// </summary>
    private IGpuBuffer? _gpuIndicesBuffer;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="MaxPool3DLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [channels, depth, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (applied to all three dimensions).</param>
    /// <param name="stride">The stride for moving the pooling window. Defaults to poolSize if 0.</param>
    /// <exception cref="ArgumentException">Thrown when inputShape has invalid length.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when poolSize or stride is non-positive.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a max pooling layer that shrinks 3D data.</para>
    /// <para>
    /// If stride equals pool size, the output will have dimensions reduced by the pool size factor.
    /// For example, with pool size 2 and stride 2, a 32x32x32 input becomes 16x16x16.
    /// </para>
    /// </remarks>
    public MaxPool3DLayer(int[] inputShape, int poolSize, int stride = 0)
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, stride == 0 ? poolSize : stride))
    {
        ValidateParameters(inputShape, poolSize, stride);

        PoolSize = poolSize;
        Stride = stride == 0 ? poolSize : stride;
    }

    #endregion

    #region Static Helper Methods

    /// <summary>
    /// Calculates the output shape based on input shape and pooling parameters.
    /// </summary>
    /// <param name="inputShape">The input shape [channels, depth, height, width].</param>
    /// <param name="poolSize">The pooling window size.</param>
    /// <param name="stride">The stride value.</param>
    /// <returns>The output shape [channels, outputDepth, outputHeight, outputWidth].</returns>
    /// <exception cref="ArgumentException">Thrown when output dimensions would be invalid.</exception>
    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int stride)
    {
        if (inputShape.Length != 4)
            throw new ArgumentException("Input shape must be [channels, depth, height, width].", nameof(inputShape));

        int channels = inputShape[0];
        int depth = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outputDepth = (depth - poolSize) / stride + 1;
        int outputHeight = (height - poolSize) / stride + 1;
        int outputWidth = (width - poolSize) / stride + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException(
                $"Pool size {poolSize} with stride {stride} produces invalid output dimensions " +
                $"[{outputDepth}, {outputHeight}, {outputWidth}] for input [{depth}, {height}, {width}].",
                nameof(inputShape));

        return [channels, outputDepth, outputHeight, outputWidth];
    }

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    /// <param name="inputShape">The input shape.</param>
    /// <param name="poolSize">The pool size.</param>
    /// <param name="stride">The stride.</param>
    /// <exception cref="ArgumentException">Thrown when inputShape is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when poolSize or stride is invalid.</exception>
    private static void ValidateParameters(int[] inputShape, int poolSize, int stride)
    {
        if (inputShape == null || inputShape.Length != 4)
            throw new ArgumentException("Input shape must be [channels, depth, height, width].", nameof(inputShape));
        if (poolSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(poolSize), "Pool size must be positive.");
        if (stride < 0)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride cannot be negative.");
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs the forward pass of 3D max pooling.
    /// </summary>
    /// <param name="input">
    /// The input tensor with shape [batch, channels, depth, height, width] or [channels, depth, height, width].
    /// </param>
    /// <returns>
    /// The pooled output tensor with reduced spatial dimensions.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown when input tensor has invalid rank.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the vectorized Engine.MaxPool3DWithIndices operation for CPU/GPU acceleration.
    /// The max indices are cached for use in the backward pass.
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
                $"MaxPool3D layer requires at least 4D tensor [C,D,H,W]. Got rank {rank}.", nameof(input));
        }

        var output = Engine.MaxPool3DWithIndices(
            batchedInput,
            [PoolSize, PoolSize, PoolSize],
            [Stride, Stride, Stride],
            out _maxIndices);

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
    /// Performs GPU-resident forward pass of 3D max pooling, keeping all data on GPU.
    /// </summary>
    /// <param name="inputs">The input tensors on GPU (uses first input).</param>
    /// <returns>The pooled output as a GPU-resident tensor.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var input = inputs[0];

        // Support any rank >= 4
        if (input.Shape.Length < 4)
            throw new ArgumentException($"MaxPool3D layer requires at least 4D tensor [C,D,H,W]. Got rank {input.Shape.Length}.");

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

        var poolSizeArr = new[] { PoolSize, PoolSize, PoolSize };
        var strideArr = new[] { Stride, Stride, Stride };

        var output = gpuEngine.MaxPool3DGpu<T>(input5D, poolSizeArr, strideArr, out _gpuIndicesBuffer);

        // Store _lastInput for backward pass
        _lastInput = input.ToTensor();

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
    /// Performs GPU-resident backward pass of 3D max pooling.
    /// </summary>
    /// <param name="outputGradient">The gradient of the output on GPU.</param>
    /// <returns>The gradient with respect to input as a GPU-resident tensor.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine");

        if (_gpuInputShape == null || _gpuIndicesBuffer == null)
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

        var poolSizeArr = new[] { PoolSize, PoolSize, PoolSize };
        var strideArr = new[] { Stride, Stride, Stride };

        var inputGrad = gpuEngine.MaxPool3DBackwardGpu<T>(gradient5D, _gpuIndicesBuffer, _gpuInputShape, poolSizeArr, strideArr);

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
    /// Performs the backward pass to route gradients through max pooling.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when Forward has not been called.</exception>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients are routed only to the positions that had the maximum
    /// values in the forward pass. All other positions receive zero gradient.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _maxIndices == null)
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

        var inputGrad = Engine.MaxPool3DBackward(
            batchedGradient,
            _maxIndices,
            inputShape,
            [PoolSize, PoolSize, PoolSize],
            [Stride, Stride, Stride]);

        // Restore to original input shape
        return inputGrad.Reshape(_lastInput.Shape);
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates parameters. Max pooling has no trainable parameters.
    /// </summary>
    /// <param name="learningRate">The learning rate (unused).</param>
    public override void UpdateParameters(T learningRate)
    {
        // Max pooling has no trainable parameters - nothing to update
    }

    /// <summary>
    /// Gets all trainable parameters. Max pooling has none.
    /// </summary>
    /// <returns>An empty vector.</returns>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
    }

    #endregion

    #region State Management

    /// <summary>
    /// Resets the cached state from forward/backward passes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this method to free memory after inference is complete or when
    /// switching between different inputs.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _maxIndices = null;
        _gpuInputShape = null;
        _addedBatchDimension = false;
        _gpuIndicesBuffer?.Dispose();
        _gpuIndicesBuffer = null;
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
        writer.Write(PoolSize);
        writer.Write(Stride);
    }

    /// <summary>
    /// Deserializes the layer from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        PoolSize = reader.ReadInt32();
        Stride = reader.ReadInt32();
    }

    #endregion

    #region Activation Info

    /// <summary>
    /// Gets the activation function types used by this layer.
    /// </summary>
    /// <returns>An empty enumerable as max pooling has no activation function.</returns>
    public override IEnumerable<ActivationFunction> GetActivationTypes()
    {
        return [];
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

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "maxpool3d_input");
        inputNodes.Add(inputNode);

        var poolNode = TensorOperations<T>.MaxPool3D(
            inputNode,
            new int[] { PoolSize, PoolSize, PoolSize },
            new int[] { Stride, Stride, Stride });

        return poolNode;
    }

    #endregion
}
