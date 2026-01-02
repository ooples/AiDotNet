

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
    /// Stores the output shape for backward pass gradient distribution.
    /// </summary>
    private int[]? _lastOutputShape;

    /// <summary>
    /// Tracks whether a batch dimension was added during the forward pass.
    /// </summary>
    private bool _addedBatchDimension;

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
        // Support both 3D [C, H, W] and 4D [B, C, H, W] input
        if (input.Shape.Length != 3 && input.Shape.Length != 4)
            throw new ArgumentException("Input tensor must have 3 dimensions (channels, height, width) or 4 dimensions (batch, channels, height, width)");

        // Store input for autodiff backward pass
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

        // Use Engine's GPU-accelerated AvgPool2D (operates on 4D); return shape matches input rank (3D or 4D)
        var output4D = Engine.AvgPool2D(input4D, PoolSize, Strides, padding: 0);

        // Return with matching dimensions
        if (_addedBatchDimension)
        {
            // Remove batch dimension: [1, C, H, W] -> [C, H, W]
            // Use actual output shape from pooling, not pre-computed OutputShape
            var actualOutputShape = new int[] { output4D.Shape[1], output4D.Shape[2], output4D.Shape[3] };
            var output = output4D.Reshape(actualOutputShape);
            _lastOutputShape = actualOutputShape;
            return output;
        }
        else
        {
            // Keep 4D: [B, C, outH, outW]
            _lastOutputShape = output4D.Shape;
            return output4D;
        }
    }

    /// <summary>
    /// Performs the backward pass of the average pooling operation.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the output gradient tensor doesn't have 3 dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> During training, neural networks need to adjust their parameters based on
    /// how much error they made. This adjustment flows backward through the network.
    ///
    /// In average pooling, all values in each window contributed equally to the output average.
    /// So during the backward pass, the gradient is distributed equally to all positions in the window.
    /// Each position receives (output gradient) / (pool size × pool size).
    ///
    /// This is different from max pooling, where only the maximum value gets the gradient.
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
        var strideArr = new int[] { Strides, Strides };

        // Use GPU-accelerated AvgPool2DBackward via Engine (operates on 4D tensors internally; result is reshaped to match the input dimensions)
        var inputGradient4D = Engine.AvgPool2DBackward(gradient4D, inputShape4D, poolSizeArr, strideArr);

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
    /// This method uses automatic differentiation to compute gradients using the AvgPool2D
    /// operation from TensorOperations. This provides:
    /// - Automatic gradient computation through the computation graph
    /// - Verification of manual gradient implementations
    /// - Support for rapid prototyping with custom modifications
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
            input4D = _lastInput.Reshape(new int[] { 1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2] });
            gradient4D = outputGradient.Shape.Length == 3
                ? outputGradient.Reshape(new int[] { 1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2] })
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

        // Forward pass using autodiff AvgPool2D operation
        var poolSize = new int[] { PoolSize, PoolSize };
        var strides = new int[] { Strides, Strides };
        var outputNode = Autodiff.TensorOperations<T>.AvgPool2D(inputNode, poolSize, strides);

        // Perform backward pass with 4D gradient
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
    }

    /// <summary>
    /// Exports the average pooling layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the average pooling operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node with shape [batch=1, channels, height, width]
    /// 2. Applies the AvgPool2D operation with specified pool size and strides
    /// 3. No learnable parameters needed (average pooling is parameter-free)
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of average pooling for JIT.
    ///
    /// JIT compilation converts the average pooling operation into optimized native code.
    /// Average pooling:
    /// - Reduces spatial dimensions by averaging values in each pooling window
    /// - Slides a window across the input with specified stride
    /// - Provides smoother downsampling compared to max pooling
    /// - Has no trainable parameters (purely computational)
    ///
    /// The symbolic graph allows the JIT compiler to:
    /// - Optimize the sliding window computation
    /// - Generate SIMD-optimized code for parallel averaging
    /// - Fuse operations with adjacent layers
    ///
    /// Average pooling is commonly used in CNNs for downsampling and global pooling.
    /// JIT compilation provides 5-10x speedup by optimizing the window operations.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer shape is not configured.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured. Initialize the layer first.");

        // Create symbolic input node (shape definition only, batch size adapts at runtime)
        // AveragePoolingLayer expects input shape: [channels, height, width]
        // AvgPool2D expects: [batch, channels, height, width]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Get pooling parameters
        var poolSize = GetPoolSize();    // [poolSize, poolSize]
        var strides = GetStride();       // [strides, strides]

        // Apply AvgPool2D operation
        var avgPoolNode = TensorOperations<T>.AvgPool2D(
            inputNode,
            poolSize: poolSize,
            strides: strides);

        return avgPoolNode;
    }

    /// <summary>
    /// Gets whether this average pooling layer supports JIT compilation.
    /// </summary>
    /// <value>True if the layer is properly configured.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. The layer supports JIT if:
    /// - Input shape is configured
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this layer can use JIT compilation for faster inference.
    ///
    /// The layer can be JIT compiled if:
    /// - The layer has been initialized with valid input shape
    ///
    /// Average pooling has no trainable parameters, so it can be JIT compiled immediately
    /// after initialization. It's a purely computational operation that:
    /// - Averages values in sliding windows
    /// - Reduces spatial dimensions
    /// - Provides translation invariance
    ///
    /// JIT compilation optimizes:
    /// - Window sliding and boundary handling
    /// - Parallel averaging across channels
    /// - Memory access patterns for cache efficiency
    ///
    /// Once initialized, JIT compilation can provide significant speedup (5-10x)
    /// especially for large feature maps in CNNs.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // AveragePooling supports JIT if input shape is configured
            // No trainable parameters needed
            return InputShape != null && InputShape.Length > 0;
        }
    }
}
