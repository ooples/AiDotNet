using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a max pooling layer for neural networks, which reduces the spatial dimensions
/// of the input by taking the maximum value in each pooling window.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A max pooling layer helps reduce the size of data flowing through a neural network
/// while keeping the most important information. It works by dividing the input into small windows
/// (determined by the pool size) and keeping only the largest value from each window.
/// 
/// Think of it like summarizing a detailed picture: instead of describing every pixel,
/// you just point out the most noticeable feature in each area of the image.
/// 
/// This helps the network:
/// 1. Focus on the most important features
/// 2. Reduce computation needs
/// 3. Make the model more robust to small changes in input position
/// </remarks>
[LayerCategory(LayerCategory.Pooling)]
[LayerTask(LayerTask.DownSampling)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 4, 4", TestConstructorArgs = "new[] { 1, 4, 4 }, 2, 2")]
public class MaxPoolingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the size of the pooling window.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how large of an area we look at when selecting the maximum value.
    /// For example, a pool size of 2 means we look at 2×2 squares of the input.
    /// </remarks>
    public int PoolSize { get; private set; }

    /// <summary>
    /// Gets the step size when moving the pooling window across the input.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This controls how much we move our window each time.
    /// For example, a stride of 2 means we move the window 2 pixels at a time,
    /// which reduces the output size to half of the input size (assuming pool size is also 2).
    /// </remarks>
    public int Stride { get; private set; }

    /// <summary>
    /// Indicates whether this layer supports training operations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This property tells the neural network system whether this layer
    /// can be trained (adjusted) during the learning process. Max pooling layers don't have
    /// parameters to train, but they do support the training process by allowing gradients
    /// to flow backward through them.
    /// </remarks>
    /// <summary>
    /// Gets the pool size for the pooling operation.
    /// </summary>
    /// <returns>An array containing the pool size for height and width dimensions.</returns>
    public int[] GetPoolSize()
    {
        return new int[] { PoolSize, PoolSize };
    }

    /// <summary>
    /// Gets the stride for the pooling operation.
    /// </summary>
    /// <returns>An array containing the stride for height and width dimensions.</returns>
    public int[] GetStride()
    {
        return new int[] { Stride, Stride };
    }

    public override bool SupportsTraining => true;

    /// <summary>
    /// Stores the indices of the maximum values found during the forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This keeps track of which input value was the maximum in each pooling window.
    /// We need this information during the backward pass to know where to send the gradients.
    /// </remarks>
    private int[,,,,]? _maxIndices;

    /// <summary>
    /// Stores GPU-resident pooling indices for backward pass.
    /// </summary>
    private IGpuBuffer? _gpuIndices;

    /// <summary>
    /// Stores the input shape from the GPU forward pass for backward pass.
    /// </summary>
    private int[]? _gpuInputShape;

    /// <summary>
    /// Stores the last input tensor from the forward pass for use in autodiff backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Tracks whether a batch dimension was added during the forward pass.
    /// </summary>
    private bool _addedBatchDimension;
    private int[]? _originalInputShape;

    /// <summary>
    /// Stores the actual output shape for 3D inputs (may differ from pre-computed OutputShape).
    /// </summary>
    private int[]? _lastOutputShape;

    /// <summary>
    /// Creates a new max pooling layer with the specified parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input data (channels, height, width).</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="stride">The step size when moving the pooling window.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor sets up the max pooling layer with your chosen settings.
    /// It calculates what the output shape will be based on your input shape, pool size, and strides.
    /// </remarks>
    public MaxPoolingLayer(int[] inputShape, int poolSize, int stride)
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, stride))
    {
        PoolSize = poolSize;
        Stride = stride;
    }

    /// <summary>
    /// Indicates that this layer supports GPU-accelerated execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs GPU-resident forward pass of max pooling, keeping all data on GPU.
    /// </summary>
    /// <param name="input">The input tensor on GPU.</param>
    /// <returns>The pooled output as a GPU-resident tensor.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var input = inputs[0];

        // Support any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        if (input.Shape.Length < 3)
            throw new ArgumentException($"MaxPooling layer requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");

        Tensor<T> input4D;
        bool addedBatch = false;
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        if (rank == 3)
        {
            addedBatch = true;
            input4D = input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        }
        else if (rank == 4)
        {
            input4D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            input4D = input.Reshape(new[] { flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1] });
        }

        _gpuInputShape = input4D._shape;
        _addedBatchDimension = addedBatch;

        var poolSizeArr = new[] { PoolSize, PoolSize };
        var strideArr = new[] { Stride, Stride };

        // Dispose previous GPU indices if any
        _gpuIndices?.Dispose();

        var output = gpuEngine.MaxPool2DGpu<T>(input4D, poolSizeArr, strideArr, out var indices);
        _gpuIndices = indices;

        // Return with matching dimensions to preserve original tensor rank
        if (_originalInputShape.Length > 4)
        {
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 3] = output.Shape[1];
            outputShape[_originalInputShape.Length - 2] = output.Shape[2];
            outputShape[_originalInputShape.Length - 1] = output.Shape[3];
            return output.Reshape(outputShape);
        }
        if (addedBatch)
        {
            return output.Reshape(new[] { output.Shape[1], output.Shape[2], output.Shape[3] });
        }
        return output;
    }

    /// <summary>
    /// Calculates the output shape based on the input shape and pooling parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input data.</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="stride">The step size when moving the pooling window.</param>
    /// <returns>The calculated output shape.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method figures out how big the output will be after max pooling.
    /// The formula used is a standard way to calculate how many complete windows fit into the input,
    /// taking into account the stride (step size).
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int stride)
    {
        // Industry-standard: support tensors of any rank
        // The last two dimensions are always height and width for pooling
        // Supports: 2D [H, W], 3D [C, H, W], 4D [B, C, H, W], etc.
        if (inputShape.Length < 2)
        {
            throw new ArgumentException("Input shape must have at least 2 dimensions for pooling.");
        }

        int heightIdx = inputShape.Length - 2;
        int widthIdx = inputShape.Length - 1;

        int outputHeight = (inputShape[heightIdx] - poolSize) / stride + 1;
        int outputWidth = (inputShape[widthIdx] - poolSize) / stride + 1;

        // Create output shape preserving all leading dimensions
        var outputShape = new int[inputShape.Length];
        for (int i = 0; i < inputShape.Length - 2; i++)
        {
            outputShape[i] = inputShape[i];
        }
        outputShape[heightIdx] = outputHeight;
        outputShape[widthIdx] = outputWidth;

        return outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the max pooling operation.
    /// </summary>
    /// <param name="input">The input tensor to apply max pooling to.</param>
    /// <returns>The output tensor after max pooling.</returns>
    /// <exception cref="ArgumentException">Thrown when the input tensor doesn't have 3 dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This is where the actual max pooling happens. For each small window in the input:
    /// 1. We look at all values in that window
    /// 2. We find the largest value
    /// 3. We put that value in the output
    /// 4. We remember where that maximum value was located (for the backward pass)
    /// 
    /// The method processes the input channel by channel, sliding the pooling window across
    /// the height and width dimensions.
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Support any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        if (input.Shape.Length < 3)
            throw new ArgumentException($"MaxPooling layer requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");

        _lastInput = input;
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        Tensor<T> input4D;
        if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            _addedBatchDimension = true;
            input4D = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        }
        else if (rank == 4)
        {
            // 4D [B, C, H, W] - no reshaping needed
            _addedBatchDimension = false;
            input4D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            _addedBatchDimension = false;
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            input4D = Engine.Reshape(input, new[] { flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1] });
        }

        var poolSizeArr = new[] { PoolSize, PoolSize };
        var strideArr = new[] { Stride, Stride };

        // Use Engine operation (expects 4D); final output shape will match the original input rank
        var output4D = Engine.MaxPool2DWithIndices(input4D, poolSizeArr, strideArr, out _maxIndices);

        // Return with matching dimensions to preserve original tensor rank
        if (_originalInputShape.Length > 4)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 3] = output4D.Shape[1];
            outputShape[_originalInputShape.Length - 2] = output4D.Shape[2];
            outputShape[_originalInputShape.Length - 1] = output4D.Shape[3];
            _lastOutputShape = outputShape;
            return Engine.Reshape(output4D, outputShape);
        }
        if (_addedBatchDimension)
        {
            var actualOutputShape = new int[] { output4D.Shape[1], output4D.Shape[2], output4D.Shape[3] };
            _lastOutputShape = actualOutputShape;
            return Engine.Reshape(output4D, actualOutputShape);
        }
        return output4D;
    }

    /// <summary>
    /// Saves the layer's configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the data to.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method saves the layer's settings (pool size and stride)
    /// so that you can reload the exact same layer later. It's like saving your game
    /// progress so you can continue from where you left off.
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(PoolSize);
        writer.Write(Stride);
    }

    /// <summary>
    /// Loads the layer's configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the data from.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads previously saved settings for the layer.
    /// It's the counterpart to Serialize - if Serialize is like saving your game,
    /// Deserialize is like loading that saved game.
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        PoolSize = reader.ReadInt32();
        Stride = reader.ReadInt32();
    }

    /// <summary>
    /// Returns the activation functions used by this layer.
    /// </summary>
    /// <returns>An empty collection since max pooling layers don't use activation functions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Activation functions are mathematical operations that determine
    /// the output of a neural network node. They introduce non-linearity, which helps
    /// neural networks learn complex patterns.
    /// 
    /// However, max pooling layers don't use activation functions - they simply
    /// select the maximum value from each window. That's why this method returns an empty collection.
    /// </remarks>
    public override IEnumerable<ActivationFunction> GetActivationTypes()
    {
        // Max pooling doesn't have an activation function
        return Array.Empty<ActivationFunction>();
    }

    /// <summary>
    /// Updates the layer's parameters during training.
    /// </summary>
    /// <param name="learningRate">The learning rate that controls how much parameters change.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method is part of the neural network training process.
    /// 
    /// During training, most layers need to update their internal values (parameters) to learn
    /// from data. However, max pooling layers don't have any trainable parameters - they just
    /// pass through the maximum values from each window.
    /// 
    /// Think of it like a simple rule that doesn't need to be adjusted: "Always pick the largest number."
    /// Since this rule never changes, there's nothing to update in this method.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Max pooling layer doesn't have trainable parameters
    }

    /// <summary>
    /// Gets all trainable parameters of the layer.
    /// </summary>
    /// <returns>An empty vector since max pooling layers have no trainable parameters.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method returns all the values that can be adjusted during training.
    /// 
    /// Many neural network layers have weights and biases that get updated as the network learns.
    /// However, max pooling layers simply select the maximum value from each window - there are
    /// no weights or biases to adjust.
    /// 
    /// This is why the method returns an empty vector (essentially a list with no elements).
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // MaxPoolingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method clears any information the layer has stored from previous
    /// calculations.
    /// 
    /// During the forward pass, the max pooling layer remembers which positions had the maximum
    /// values (stored in _maxIndices). This is needed for the backward pass.
    /// 
    /// Resetting the state clears this memory, which is useful when:
    /// 1. Starting a new training session
    /// 2. Processing a new batch of data
    /// 3. Switching from training to evaluation mode
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _maxIndices = null;
        _addedBatchDimension = false;

        // Dispose GPU resources
        _gpuIndices?.Dispose();
        _gpuIndices = null;
        _gpuInputShape = null;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["PoolSize"] = PoolSize.ToString();
        metadata["Stride"] = Stride.ToString();
        return metadata;
    }
}
