using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a reshape layer that transforms the dimensions of input data without changing its content.
/// </summary>
/// <remarks>
/// <para>
/// The ReshapeLayer rearranges the elements of the input tensor into a new shape without changing the data itself.
/// This operation is useful for connecting layers with different shape requirements or for preparing data for
/// specific layer types. The total number of elements must remain the same between the input and output shapes.
/// </para>
/// <para><b>For Beginners:</b> This layer changes how your data is organized without changing the data itself.
/// 
/// Think of the ReshapeLayer like reorganizing a deck of playing cards:
/// - If you have cards arranged in 4 rows of 13 cards (representing the 4 suits)
/// - You could reorganize them into 13 rows of 4 cards (representing the 13 ranks)
/// - The cards themselves haven't changed, just how they're arranged
/// 
/// For example, in image processing:
/// - You might have an image of shape [height, width, channels]
/// - But a particular layer might need the data as a flat vector
/// - A reshape layer can convert between these formats without losing information
/// 
/// Common use cases include:
/// - Flattening data (e.g., converting a 2D image to a 1D vector for a dense layer)
/// - Reshaping for convolutional operations (e.g., turning a vector into a 3D tensor)
/// - Batch dimension manipulation (e.g., splitting or combining batch items)
/// 
/// The key requirement is that the total number of elements stays the same - you're just
/// reorganizing them into a different dimensional structure.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ReshapeLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The shape of the input tensor, excluding the batch dimension.
    /// </summary>
    /// <remarks>
    /// This array stores the dimensions of the input tensor not including the batch dimension (which is 
    /// always the first dimension). It is used to validate input shapes and to perform the reshaping operation.
    /// </remarks>
    private int[] _inputShape;

    /// <summary>
    /// The shape of the output tensor, excluding the batch dimension.
    /// </summary>
    /// <remarks>
    /// This array stores the dimensions of the output tensor not including the batch dimension (which is 
    /// always the first dimension). It defines the target shape for the reshaping operation.
    /// </remarks>
    private int[] _outputShape;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute the appropriate gradients.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for ReshapeLayer, indicating that the layer can participate in backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the ReshapeLayer supports the training process through backpropagation.
    /// While the layer itself has no trainable parameters, it needs to properly propagate gradients during
    /// the backward pass, reshaping them to match the input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can participate in the learning process.
    /// 
    /// A value of true means:
    /// - The layer can pass learning signals (gradients) backward through it
    /// - It contributes to the training of the entire network
    /// 
    /// While this layer doesn't have any internal values that it learns directly,
    /// it's designed to let learning signals flow through it to previous layers.
    /// It just needs to reshape these signals to match the original input shape.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReshapeLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor, excluding the batch dimension.</param>
    /// <param name="outputShape">The shape of the output tensor, excluding the batch dimension.</param>
    /// <exception cref="ArgumentException">Thrown when the total number of elements in the input and output shapes are not equal.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReshapeLayer with the specified input and output shapes. It validates that
    /// the total number of elements remains the same between the input and output shapes, as the layer can only
    /// rearrange elements, not create or remove them.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new reshape layer for your neural network.
    /// 
    /// When you create this layer, you specify:
    /// - inputShape: The current organization of your data (not including the batch dimension)
    /// - outputShape: The desired organization of your data (not including the batch dimension)
    /// 
    /// For example:
    /// - If inputShape is [28, 28] (like a 28×28 image)
    /// - You could set outputShape to [784] to flatten it into a single vector
    /// 
    /// The constructor checks that the total number of elements stays the same:
    /// - For the example above, 28×28 = 784, so the shapes are compatible
    /// - If the total elements don't match, you'll get an error
    /// 
    /// The batch dimension (first dimension) is handled automatically and not included in these shapes.
    /// </para>
    /// </remarks>
    public ReshapeLayer(int[] inputShape, int[] outputShape)
        : base(inputShape, outputShape)
    {
        _inputShape = inputShape;
        _outputShape = outputShape;

        if (inputShape.Aggregate(1, (a, b) => a * b) != outputShape.Aggregate(1, (a, b) => a * b))
        {
            throw new ArgumentException("Input and output shapes must have the same total number of elements.");
        }
    }

    /// <summary>
    /// Gets the target shape for the reshape operation.
    /// </summary>
    /// <returns>The target shape array (excluding batch dimension).</returns>
    public int[] GetTargetShape()
    {
        return _outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the reshape layer.
    /// </summary>
    /// <param name="input">The input tensor to reshape.</param>
    /// <returns>The reshaped output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the reshape layer. It creates a new tensor with the specified
    /// output shape and copies the elements from the input tensor into the output tensor while preserving their
    /// order. The input tensor is cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method reorganizes your data into the new shape.
    /// 
    /// During the forward pass:
    /// 1. The layer saves the original input for later use in the backward pass
    /// 2. It creates a new, empty tensor with the target shape
    /// 3. It copies all values from the input to the output tensor
    /// 4. The data values themselves stay exactly the same, just arranged differently
    /// 
    /// The layer handles each item in your batch separately, maintaining the batch structure.
    /// 
    /// Think of it like pouring water from one differently-shaped container to another - 
    /// the amount of water stays the same, but it takes the shape of the new container.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int[] targetShape = new int[_outputShape.Length + 1];
        targetShape[0] = batchSize;
        Array.Copy(_outputShape, 0, targetShape, 1, _outputShape.Length);

        return Engine.Reshape(input, targetShape);
    }

    /// <summary>
    /// Performs the backward pass of the reshape layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the reshape layer, which is used during training to propagate
    /// error gradients back through the network. It reshapes the output gradient tensor back to the original
    /// input shape, effectively reversing the transformation done in the forward pass. This ensures that
    /// gradients flow correctly to previous layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms the learning signals back to the original shape.
    ///
    /// During the backward pass:
    /// 1. The layer receives gradients in the reshaped format (output shape)
    /// 2. It needs to convert these gradients back to the original format (input shape)
    /// 3. This allows previous layers to properly use these gradients for learning
    ///
    /// Essentially, this is the reverse of the forward pass - it takes the learning signals
    /// and reorganizes them to match the original data structure, without changing their values.
    ///
    /// This is necessary because each layer in the network expects gradients in the same shape
    /// as its original output.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Performs the backward pass using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Convert input to computation node
        var input = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Replay forward pass using autodiff operations
        int batchSize = _lastInput.Shape[0];
        int[] targetShape = [batchSize, .. _outputShape];
        var reshaped = Autodiff.TensorOperations<T>.Reshape(input, targetShape);

        // Set gradient at output and perform backward pass
        reshaped.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((reshaped, false));

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

        // Extract and return input gradient
        if (input.Gradient == null)
            throw new InvalidOperationException("Input gradient was not computed during backward pass.");

        return input.Gradient;
    }

    /// <summary>
    /// Performs the backward pass using manual gradient computation (optimized implementation).
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Reshape gradient back to input shape
        // Input shape is [batchSize, ..._inputShape]
        return Engine.Reshape(outputGradient, _lastInput.Shape);
    }

    /// <summary>
    /// Updates the parameters of the reshape layer.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is required by the LayerBase class but does nothing in the ReshapeLayer because this layer
    /// has no trainable parameters to update. The ReshapeLayer only transforms the data structure without
    /// applying any learned transformations.
    /// </para>
    /// <para><b>For Beginners:</b> This method is empty because the layer has no internal values to update.
    /// 
    /// Unlike most layers in a neural network, the reshape layer doesn't have any
    /// weights or biases that need to be adjusted during training. It's purely a 
    /// geometric transformation of the data structure.
    /// 
    /// The actual learning happens in other layers of the network that have
    /// trainable parameters like weights and biases.
    /// 
    /// This method exists only because all layers in the network must implement it.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // ReshapeLayer has no parameters to update
    }

    /// <summary>
    /// Gets all trainable parameters of the reshape layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since this layer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector because the ReshapeLayer has no trainable parameters. The method
    /// is required by the LayerBase class but is essentially a no-op for this layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because the layer has no learnable values.
    /// 
    /// As mentioned earlier, the reshape layer doesn't have any weights or biases
    /// that it learns during training. It just reorganizes the data structure.
    /// 
    /// This method returns an empty vector to indicate that there are no parameters to retrieve.
    /// It exists only because all layers in the network must implement it.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // ReshapeLayer has no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the reshape layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the reshape layer, clearing the cached input tensor from the
    /// forward pass. This is useful when starting to process a new batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input from the previous forward pass is cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Managing memory usage efficiently
    /// 
    /// Since this layer has no learned parameters, resetting just clears the temporarily
    /// stored input that was used for the backward pass.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because reshape is a simple reshape operation that can be JIT compiled.
    /// </value>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because reshape is a zero-copy operation that can be done via GPU tensor view.
    /// </value>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs the forward pass on GPU using a zero-copy view reshape.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU tensor view with the reshaped dimensions.</returns>
    /// <remarks>
    /// <para>
    /// This method implements GPU-resident reshape by creating a view into the input tensor
    /// with the target shape. No data is copied - only the shape interpretation changes.
    /// </para>
    /// <para><b>For Beginners:</b> The GPU version of reshape is very efficient because:
    /// - It doesn't move any data
    /// - It just tells the GPU "interpret this same data with a different shape"
    /// - This is called a "view" operation
    ///
    /// For example, if input has shape [32, 28, 28, 1] and target is [784],
    /// the view will have shape [32, 784] but still points to the same GPU memory.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(IGpuTensor<T> input)
    {
        // Calculate full target shape including batch dimension
        int batchSize = input.Shape[0];
        int[] targetShape = new int[_outputShape.Length + 1];
        targetShape[0] = batchSize;
        Array.Copy(_outputShape, 0, targetShape, 1, _outputShape.Length);

        return input.CreateView(0, targetShape);
    }

    /// <summary>
    /// Exports the reshape layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the reshaped result.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the reshape operation using a reshape node.
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (OutputShape == null || OutputShape.Length == 0)
            throw new InvalidOperationException("Layer output shape not configured.");

        // Create placeholder for input data with symbolic batch dimension
        var inputPlaceholder = new Tensor<T>(new int[] { 1 }.Concat(_inputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(inputPlaceholder, "input");

        inputNodes.Add(inputNode);

        // Reshape operation: reshape to target shape
        var targetShape = new int[] { -1 }.Concat(_outputShape).ToArray(); // -1 means variable batch size
        var outputNode = Autodiff.TensorOperations<T>.Reshape(inputNode, targetShape);

        return outputNode;
    }
}
