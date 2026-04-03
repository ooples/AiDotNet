using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements an average pooling layer for neural networks, which reduces the spatial dimensions
/// of the input by taking the average value in each pooling window.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> An average pooling layer helps reduce the size of data flowing through a neural network
/// while preserving overall characteristics. It works by dividing the input into small windows
/// (determined by the pool size) and computing the average of all values in each window.
///
/// Think of it like creating a lower-resolution summary: instead of keeping every detail,
/// you average all the values in each area to get a representative value.
///
/// This helps the network:
/// 1. Preserve background information and overall context
/// 2. Reduce computation needs
/// 3. Smooth out noisy features
///
/// Average pooling is often used in the final layers of a network or when you want to
/// preserve more spatial information compared to max pooling.
/// </remarks>
[LayerCategory(LayerCategory.Pooling)]
[LayerTask(LayerTask.DownSampling)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 4, 4", TestConstructorArgs = "new[] { 1, 4, 4 }, 2, 2")]
public class AveragePoolingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the size of the pooling window.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how large of an area we look at when computing the average value.
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
    public int Strides { get; private set; }

    /// <summary>
    /// Indicates whether this layer supports training operations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This property tells the neural network system whether this layer
    /// can be trained (adjusted) during the learning process. Average pooling layers don't have
    /// parameters to train, but they do support the training process by allowing gradients
    /// to flow backward through them.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Stores the last input tensor from the forward pass for use in autodiff backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the input shape from GPU forward pass for backward pass.
    /// </summary>
    private int[]? _gpuInputShape;

    /// <summary>
    /// Stores the output shape for backward pass gradient distribution.
    /// </summary>
    private int[]? _lastOutputShape;

    /// <summary>
    /// Tracks whether a batch dimension was added during the forward pass.
    /// </summary>
    private bool _addedBatchDimension;
    private int[]? _originalInputShape;

    /// <summary>
    /// Creates a new average pooling layer with the specified parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input data (channels, height, width).</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="strides">The step size when moving the pooling window.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor sets up the average pooling layer with your chosen settings.
    /// It calculates what the output shape will be based on your input shape, pool size, and strides.
    /// </remarks>
    public AveragePoolingLayer(int[] inputShape, int poolSize, int strides)
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, strides))
    {
        PoolSize = poolSize;
        Strides = strides;
    }

    /// <summary>
    /// Indicates that this layer supports GPU-accelerated execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs GPU-resident forward pass of average pooling, keeping all data on GPU.
    /// </summary>
    /// <param name="inputs">The input tensors on GPU (uses first input).</param>
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
            throw new ArgumentException($"AveragePooling layer requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");

        Tensor<T> input4D;
        bool addedBatch = false;
        _originalInputShape = input.Shape.ToArray();
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

        _gpuInputShape = input4D.Shape.ToArray();
        _addedBatchDimension = addedBatch;

        var poolSizeArr = new[] { PoolSize, PoolSize };
        var strideArr = new[] { Strides, Strides };

        var output = gpuEngine.AvgPool2DGpu<T>(input4D, poolSizeArr, strideArr);

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
    /// <param name="strides">The step size when moving the pooling window.</param>
    /// <returns>The calculated output shape.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method figures out how big the output will be after average pooling.
    /// The formula used is a standard way to calculate how many complete windows fit into the input,
    /// taking into account the stride (step size).
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int strides)
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

        int outputHeight = (inputShape[heightIdx] - poolSize) / strides + 1;
        int outputWidth = (inputShape[widthIdx] - poolSize) / strides + 1;

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
    /// Gets the pool size as a 2D array (height, width).
    /// </summary>
    /// <returns>An array containing [poolSize, poolSize].</returns>
    /// <remarks>
    /// This method is used by the JIT compiler to extract pooling parameters.
    /// </remarks>
    public int[] GetPoolSize()
    {
        return new int[] { PoolSize, PoolSize };
    }

    /// <summary>
    /// Gets the stride as a 2D array (height stride, width stride).
    /// </summary>
    /// <returns>An array containing [strides, strides].</returns>
    /// <remarks>
    /// This method is used by the JIT compiler to extract pooling parameters.
    /// </remarks>
    public int[] GetStride()
    {
        return new int[] { Strides, Strides };
    }

    /// <summary>
    /// Performs the forward pass of the average pooling operation.
    /// </summary>
    /// <param name="input">The input tensor to apply average pooling to.</param>
    /// <returns>The output tensor after average pooling.</returns>
    /// <exception cref="ArgumentException">Thrown when the input tensor doesn't have 3 dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This is where the actual average pooling happens. For each small window in the input:
    /// 1. We look at all values in that window
    /// 2. We calculate the average of those values
    /// 3. We put that average value in the output
    ///
    /// The method processes the input channel by channel, sliding the pooling window across
    /// the height and width dimensions.
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Support any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        if (input.Shape.Length < 3)
            throw new ArgumentException($"AveragePooling layer requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");

        // Store input for autodiff backward pass
        _lastInput = input;
        _originalInputShape = input.Shape.ToArray();
        int rank = input.Shape.Length;

        Tensor<T> input4D;
        if (rank == 3)
        {
            _addedBatchDimension = true;
            input4D = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2]);
        }
        else if (rank == 4)
        {
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
            input4D = input.Reshape(flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]);
        }

        // Use Engine's GPU-accelerated AvgPool2D (operates on 4D); return shape matches input rank
        var output4D = Engine.AvgPool2D(input4D, PoolSize, Strides, padding: 0);

        // Return with matching dimensions to preserve original tensor rank
        if (_originalInputShape.Length > 4)
        {
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 3] = output4D.Shape[1];
            outputShape[_originalInputShape.Length - 2] = output4D.Shape[2];
            outputShape[_originalInputShape.Length - 1] = output4D.Shape[3];
            _lastOutputShape = outputShape;
            return output4D.Reshape(outputShape);
        }
        if (_addedBatchDimension)
        {
            var actualOutputShape = new int[] { output4D.Shape[1], output4D.Shape[2], output4D.Shape[3] };
            var output = output4D.Reshape(actualOutputShape);
            _lastOutputShape = actualOutputShape;
            return output;
        }
        else
        {
            _lastOutputShape = output4D.Shape.ToArray();
            return output4D;
        }
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
        writer.Write(Strides);
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
        Strides = reader.ReadInt32();
    }

    /// <summary>
    /// Returns the activation functions used by this layer.
    /// </summary>
    /// <returns>An empty collection since average pooling layers don't use activation functions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Activation functions are mathematical operations that determine
    /// the output of a neural network node. They introduce non-linearity, which helps
    /// neural networks learn complex patterns.
    ///
    /// However, average pooling layers don't use activation functions - they simply
    /// compute the average of values in each window. That's why this method returns an empty collection.
    /// </remarks>
    public override IEnumerable<ActivationFunction> GetActivationTypes()
    {
        // Average pooling doesn't have an activation function
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
    /// from data. However, average pooling layers don't have any trainable parameters - they just
    /// compute the average of values in each window.
    ///
    /// Think of it like a simple rule that doesn't need to be adjusted: "Always compute the average."
    /// Since this rule never changes, there's nothing to update in this method.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Average pooling layer doesn't have trainable parameters
    }

    /// <summary>
    /// Gets all trainable parameters of the layer.
    /// </summary>
    /// <returns>An empty vector since average pooling layers have no trainable parameters.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method returns all the values that can be adjusted during training.
    ///
    /// Many neural network layers have weights and biases that get updated as the network learns.
    /// However, average pooling layers simply compute the average of values in each window - there are
    /// no weights or biases to adjust.
    ///
    /// This is why the method returns an empty vector (essentially a list with no elements).
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // AveragePoolingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method clears any information the layer has stored from previous
    /// calculations.
    ///
    /// During the forward pass, the average pooling layer stores the input for use in the backward pass.
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
        _lastOutputShape = null;
        _addedBatchDimension = false;
        _gpuInputShape = null;
    }
}
