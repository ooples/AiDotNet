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
    public override IGpuTensor<T> ForwardGpu(IGpuTensor<T> input)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        // Ensure input is 4D [batch, channels, height, width]
        IGpuTensor<T> input4D;
        bool addedBatch = false;

        if (input.Shape.Length == 3)
        {
            // Add batch dimension: [C, H, W] -> [1, C, H, W]
            addedBatch = true;
            input4D = input.CreateView(0, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        }
        else if (input.Shape.Length == 4)
        {
            input4D = input;
        }
        else
        {
            throw new ArgumentException("Input must be 3D [C, H, W] or 4D [batch, C, H, W]");
        }

        _gpuInputShape = input4D.Shape;
        _addedBatchDimension = addedBatch;

        var poolSizeArr = new[] { PoolSize, PoolSize };
        var strideArr = new[] { Stride, Stride };

        // Dispose previous GPU indices if any
        _gpuIndices?.Dispose();

        var output = gpuEngine.MaxPool2DGpu<T>(input4D, poolSizeArr, strideArr, out var indices);
        _gpuIndices = indices;

        // Return with matching dimensions
        if (addedBatch)
        {
            return output.CreateView(0, new[] { output.Shape[1], output.Shape[2], output.Shape[3] });
        }
        return output;
    }

    /// <summary>
    /// Performs GPU-resident backward pass of max pooling.
    /// </summary>
    /// <param name="outputGradient">The gradient of the output on GPU.</param>
    /// <returns>The gradient with respect to input as a GPU-resident tensor.</returns>
    public IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine");

        if (_gpuIndices == null || _gpuInputShape == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu");

        // Ensure gradient is 4D
        IGpuTensor<T> gradient4D;
        if (outputGradient.Shape.Length == 3)
        {
            gradient4D = outputGradient.CreateView(0, new[] { 1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2] });
        }
        else
        {
            gradient4D = outputGradient;
        }

        var poolSizeArr = new[] { PoolSize, PoolSize };
        var strideArr = new[] { Stride, Stride };

        var inputGrad = gpuEngine.MaxPool2DBackwardGpu<T>(gradient4D, _gpuIndices, _gpuInputShape, poolSizeArr, strideArr);

        // Return with matching dimensions
        if (_addedBatchDimension)
        {
            return inputGrad.CreateView(0, new[] { inputGrad.Shape[1], inputGrad.Shape[2], inputGrad.Shape[3] });
        }
        return inputGrad;
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
        // Support both 3D [C, H, W] and 4D [B, C, H, W] input
        if (input.Shape.Length != 3 && input.Shape.Length != 4)
            throw new ArgumentException("Input tensor must have 3 dimensions (channels, height, width) or 4 dimensions (batch, channels, height, width)");

        _lastInput = input;

        Tensor<T> input4D;
        if (input.Shape.Length == 3)
        {
            // Add batch dimension: [C, H, W] -> [1, C, H, W]
            _addedBatchDimension = true;
            input4D = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2]);
        }
        else
        {
            // Already 4D: [B, C, H, W]
            _addedBatchDimension = false;
            input4D = input;
        }

        var poolSizeArr = new[] { PoolSize, PoolSize };
        var strideArr = new[] { Stride, Stride };

        // Use Engine operation (expects 4D); final output shape will match the original input (3D or 4D)
        var output4D = Engine.MaxPool2DWithIndices(input4D, poolSizeArr, strideArr, out _maxIndices);

        // Return with matching dimensions (3D if batch was added, 4D otherwise)
        if (_addedBatchDimension)
        {
            // Use actual output shape from pooling, not pre-computed OutputShape
            var actualOutputShape = new int[] { output4D.Shape[1], output4D.Shape[2], output4D.Shape[3] };
            _lastOutputShape = actualOutputShape;
            return output4D.Reshape(actualOutputShape);
        }
        return output4D;
    }

    /// <summary>
    /// Performs the backward pass of the max pooling operation.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the output gradient tensor doesn't have 3 dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> During training, neural networks need to adjust their parameters based on
    /// how much error they made. This adjustment flows backward through the network.
    ///
    /// In max pooling, only the maximum value from each window contributed to the output.
    /// So during the backward pass, the gradient only flows back to that maximum value's position.
    /// All other positions receive zero gradient because they didn't contribute to the output.
    ///
    /// Think of it like giving credit only to the team member who contributed the most to a project.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the output gradient tensor doesn't have 3 dimensions.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Support both 3D and 4D gradients (matching forward pass)
        if (outputGradient.Shape.Length != 3 && outputGradient.Shape.Length != 4)
            throw new ArgumentException("Output gradient tensor must have 3 or 4 dimensions");

        if (_maxIndices == null)
            throw new InvalidOperationException("Max indices not available for backward pass.");

        Tensor<T> gradient4D;
        int[] inputShape4D;

        if (outputGradient.Shape.Length == 3)
        {
            // Reshape to 4D: [C, H, W] -> [1, C, H, W]
            gradient4D = outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2]);
            inputShape4D = new int[] { 1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2] };
        }
        else
        {
            // Already 4D
            gradient4D = outputGradient;
            inputShape4D = _lastInput.Shape;
        }

        var poolSizeArr = new int[] { PoolSize, PoolSize };
        var strideArr = new int[] { Stride, Stride };

        // Use Engine operation in 4D; reshape so the returned gradient matches the original input dimensions
        var inputGradient4D = Engine.MaxPool2DBackward(gradient4D, _maxIndices, inputShape4D, poolSizeArr, strideArr);

        // Return with matching dimensions
        return _addedBatchDimension
            ? inputGradient4D.Reshape(_lastInput.Shape)
            : inputGradient4D;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs a MaxPool2D operation using automatic differentiation
    /// and propagates gradients through the computation graph. It serves as an alternative
    /// reference implementation to the manual engine-backed backward pass.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Handle both 3D and 4D input
        Tensor<T> input4D;
        Tensor<T> gradient4D;

        if (_lastInput.Shape.Length == 3)
        {
            // Reshape to 4D: [C, H, W] -> [1, C, H, W]
            input4D = _lastInput.Reshape(1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2]);
            gradient4D = outputGradient.Shape.Length == 3
                ? outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2])
                : outputGradient;
        }
        else
        {
            // Already 4D
            input4D = _lastInput;
            gradient4D = outputGradient;
        }

        // Convert input to computation node
        var inputNode = Autodiff.TensorOperations<T>.Variable(input4D, "input", requiresGradient: true);

        // Forward pass using autodiff MaxPool2D operation
        var poolSize = new int[] { PoolSize, PoolSize };
        var strideArr = new int[] { Stride, Stride };
        var outputNode = Autodiff.TensorOperations<T>.MaxPool2D(inputNode, poolSize, strideArr);

        // Set output gradient
        outputNode.Gradient = gradient4D;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract input gradient and reshape back to original dimensions
        var inputGrad4D = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
        return _addedBatchDimension ? inputGrad4D.Reshape(_lastInput.Shape) : inputGrad4D;
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

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        var poolSize = GetPoolSize();
        var strides = GetStride();

        var maxPoolNode = TensorOperations<T>.MaxPool2D(inputNode, poolSize: poolSize, strides: strides);
        return maxPoolNode;
    }

    public override bool SupportsJitCompilation
    {
        get
        {
            return InputShape != null && InputShape.Length > 0;
        }
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["PoolSize"] = PoolSize.ToString();
        metadata["Stride"] = Stride.ToString();
        return metadata;
    }
}
