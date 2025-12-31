namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a flatten layer that reshapes multi-dimensional input data into a 1D vector.
/// </summary>
/// <remarks>
/// <para>
/// A flatten layer transforms multi-dimensional input data (such as images or feature maps) into a one-dimensional
/// vector. This is often necessary when transitioning from convolutional layers to fully connected layers
/// in a neural network. The flatten operation preserves all values and their order, just changing the way
/// they are arranged from a multi-dimensional tensor to a single vector.
/// </para>
/// <para><b>For Beginners:</b> A flatten layer converts multi-dimensional data into a simple list of numbers.
/// 
/// Imagine you have a 2D grid of numbers (like a small image):
/// ```
/// [
///   [1, 2, 3],
///   [4, 5, 6]
/// ]
/// ```
/// 
/// The flatten layer turns this into a single row:
/// ```
/// [1, 2, 3, 4, 5, 6]
/// ```
/// 
/// This transformation is needed because:
/// - Convolutional layers work with 2D or 3D data (like images)
/// - Fully connected layers expect a simple list of numbers
/// - Flatten layers bridge these two types of layers
/// 
/// Think of it like taking a book (a 3D object with pages) and reading all the text 
/// in order from beginning to end (a 1D sequence). All the information is preserved,
/// but it's rearranged into a different shape.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class FlattenLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The shape of the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This array stores the dimensions of the input tensor (excluding the batch dimension).
    /// It is used during the forward and backward passes to correctly flatten and unflatten the tensors.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers the original shape of the input data.
    /// 
    /// For example:
    /// - For a 28×28 grayscale image: [28, 28, 1]
    /// - For RGB color channels: [height, width, 3]
    /// - For a feature map with multiple channels: [height, width, channels]
    /// 
    /// The layer needs to store this original shape:
    /// - To correctly convert multi-dimensional data to a flat vector
    /// - To convert gradients back to the original shape during training
    /// 
    /// It's like keeping a map of how the data was originally organized so you
    /// can "unfold" it in exactly the same way later.
    /// </para>
    /// </remarks>
    private int[] _inputShape;

    /// <summary>
    /// The size of the output vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the total size of the flattened output vector, which is the product of all
    /// dimensions in the input shape. It represents the number of elements in the input tensor
    /// for a single example.
    /// </para>
    /// <para><b>For Beginners:</b> This is the total number of values after flattening.
    /// 
    /// The output size is calculated by multiplying all the dimensions of the input:
    /// - For a 28×28 image: 28 × 28 = 784 values
    /// - For a 16×16×32 feature map: 16 × 16 × 32 = 8,192 values
    /// 
    /// This number tells us:
    /// - How long the flattened vector will be
    /// - How many neurons the next layer (usually a fully connected layer) will receive
    /// 
    /// Pre-calculating this size makes processing more efficient.
    /// </para>
    /// </remarks>
    private int _outputSize;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input received during the last forward pass. It is necessary
    /// for computing gradients during the backward pass, as it provides information about
    /// the original shape of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what input data was processed most recently.
    /// 
    /// During training:
    /// - The layer needs to remember the shape and organization of its input
    /// - This helps when calculating how to send gradients back to previous layers
    /// - Without this information, the layer couldn't "unflatten" the gradients correctly
    /// 
    /// This is automatically cleared between training batches to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> because flatten layers have no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the flatten layer does not have any trainable parameters.
    /// The layer simply performs a reshape operation and does not learn during training.
    /// However, it still participates in backpropagation by passing gradients back to previous
    /// layers in the correct shape.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer doesn't learn or change during training.
    /// 
    /// A value of false means:
    /// - The layer has no weights or biases to adjust
    /// - It performs the same operation regardless of training
    /// - It's a fixed transformation layer, not a learning layer
    /// 
    /// Unlike convolutional or fully connected layers (which learn patterns from data),
    /// the flatten layer just reorganizes data without changing its content.
    /// 
    /// It's like rearranging furniture in a room - you're not adding or removing
    /// anything, just changing how it's organized.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="FlattenLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor (excluding the batch dimension).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new flatten layer that will reshape input data with the specified shape
    /// into a one-dimensional vector. The output size is calculated as the product of all dimensions in
    /// the input shape. The layer expects input tensors with shape [batchSize, ...inputShape].
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the flatten layer by specifying what shape of data it will receive.
    /// 
    /// When creating a flatten layer, you need to specify:
    /// - The dimensions of your input data (not counting the batch size)
    /// 
    /// For example:
    /// ```csharp
    /// // Create a flatten layer for 28×28 grayscale images
    /// var flattenLayer = new FlattenLayer<float>(new int[] { 28, 28, 1 });
    /// 
    /// // Create a flatten layer for output from a convolutional layer with 64 feature maps of size 7×7
    /// var flattenConvOutput = new FlattenLayer<float>(new int[] { 7, 7, 64 });
    /// ```
    /// 
    /// The constructor automatically calculates how large the output vector will be
    /// by multiplying all the dimensions together.
    /// </para>
    /// </remarks>
    public FlattenLayer(int[] inputShape)
        : base(inputShape, [inputShape.Aggregate(1, (a, b) => a * b)])
    {
        _inputShape = inputShape;
        _outputSize = inputShape.Aggregate(1, (a, b) => a * b);
    }

    /// <summary>
    /// Performs the forward pass of the flatten layer, reshaping multi-dimensional data into a vector.
    /// </summary>
    /// <param name="input">The input tensor to flatten. Shape: [batchSize, ...inputShape].</param>
    /// <returns>The flattened output tensor. Shape: [batchSize, outputSize].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the flatten layer. It takes a multi-dimensional tensor
    /// and reshapes it into a 2D tensor where each row corresponds to a flattened example from the batch.
    /// For unbatched inputs (rank <= 3), it returns a 1D vector of length input.Length. The values are
    /// preserved and their order is maintained according to a row-major traversal of the input tensor.
    /// The input tensor is cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts multi-dimensional data into simple vectors.
    /// 
    /// The forward pass works like this:
    /// 1. Take multi-dimensional input (like a 3D image)
    /// 2. For each example in the batch:
    ///    - Go through all positions in the multi-dimensional input
    ///    - Place each value into the corresponding position in a flat vector
    /// 3. Return a tensor with shape [batchSize, flattenedSize]
    /// 
    /// For example, with a batch of 3D data like [batchSize, height, width, channels]:
    /// - Input shape: [32, 7, 7, 64] (32 examples, each 7×7 with 64 channels)
    /// - Output shape: [32, 3136] (32 examples, each with 7×7—64=3136 values)
    /// - For an unbatched input like [7, 7, 64], the output is a 1D vector of length 3136
    /// 
    /// The method carefully preserves the order of values so they can be
    /// "unflattened" back to the original shape during backpropagation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Handle unbatched input (3D: [C, H, W] or 2D: [H, W])
        if (input.Rank <= 3)
        {
            // Unbatched input: flatten to 1D vector
            return Engine.Reshape(input, [input.Length]);
        }

        // Batched input: flatten spatial dimensions keeping batch dimension
        int batchSize = input.Shape[0];
        // Calculate actual flattened size from input dimensions, not pre-computed _outputSize
        int actualOutputSize = 1;
        for (int i = 1; i < input.Shape.Length; i++)
        {
            actualOutputSize *= input.Shape[i];
        }
        return Engine.Reshape(input, [batchSize, actualOutputSize]);
    }

    /// <summary>
    /// Performs the backward pass of the flatten layer, reshaping gradients back to the original input shape.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, outputSize].</param>
    /// <returns>The gradient tensor reshaped to the original input shape. Shape: [batchSize, ...inputShape].</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Performs the backward pass using manual gradient computation (optimized implementation).
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, outputSize].</param>
    /// <returns>The gradient tensor reshaped to the original input shape. Shape: [batchSize, ...inputShape].</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        return Engine.Reshape(outputGradient, _lastInput.Shape);
    }

    /// <summary>
    /// Performs the backward pass using automatic differentiation via TensorOperations.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, outputSize].</param>
    /// <returns>The gradient tensor reshaped to the original input shape. Shape: [batchSize, ...inputShape].</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Create computation node for input
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Replay forward pass: flatten is just a reshape operation
        int[] flattenedShape;
        if (_lastInput.Rank <= 3)
        {
            flattenedShape = new[] { _outputSize };
        }
        else
        {
            int batchSize = _lastInput.Shape[0];
            flattenedShape = new[] { batchSize, _outputSize };
        }
        var outputNode = Autodiff.TensorOperations<T>.Reshape(inputNode, flattenedShape);

        // Set the output gradient and perform backward pass manually
        outputNode.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                        stack.Push((parent, false));
                }
            }
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract and return the input gradient
        if (inputNode.Gradient == null)
            throw new InvalidOperationException("Gradient computation failed in automatic differentiation.");

        return inputNode.Gradient;
    }

    /// <summary>
    /// Updates the parameters of the layer based on the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the flatten layer has no
    /// trainable parameters to update, so it performs no operation.
    /// </para>
    /// <para><b>For Beginners:</b> This method does nothing for flatten layers because they have no adjustable weights.
    /// 
    /// Unlike most layers (like convolutional or fully connected layers):
    /// - Flatten layers don't have weights or biases to learn
    /// - They just rearrange the data without modifying it
    /// - There's nothing to update during training
    /// 
    /// This method exists only to fulfill the requirements of the base layer class.
    /// The flatten layer participates in training by reorganizing activations and gradients,
    /// not by updating internal parameters.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // FlattenLayer has no parameters to update
    }

    /// <summary>
    /// Gets the trainable parameters of the layer.
    /// </summary>
    /// <returns>
    /// An empty vector since flatten layers have no trainable parameters.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the flatten layer has no
    /// trainable parameters to retrieve, so it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because flatten layers have no learnable values.
    /// 
    /// Unlike layers with weights and biases:
    /// - Flatten layers don't have any parameters that change during training
    /// - They perform a fixed operation (reshaping) that doesn't involve learning
    /// - There are no values to save when storing a trained model
    /// 
    /// This method returns an empty vector, indicating there are no parameters to collect.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // FlattenLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input
    /// from the previous forward pass. This is useful when starting to process a new batch of
    /// data or when switching between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    ///
    /// When resetting the state:
    /// - The saved input is cleared
    /// - The layer forgets the previous data it processed
    /// - This frees up memory and prepares for new data
    ///
    /// This is typically called:
    /// - Between training batches
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    ///
    /// It's like wiping a whiteboard clean before starting a new calculation.
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
    /// Always <c>true</c> because flatten is a simple reshape operation that can be JIT compiled.
    /// </value>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the flatten layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the flattened result.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the flatten operation using a reshape node.
    /// The flatten operation is equivalent to reshaping the input to [batchSize, product of dimensions].
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create placeholder for input data with symbolic batch dimension
        var inputPlaceholder = new Tensor<T>(new int[] { 1 }.Concat(_inputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(inputPlaceholder, "input");

        inputNodes.Add(inputNode);

        // Flatten is just a reshape operation: reshape to [batchSize, outputSize]
        var flattenedShape = new int[] { -1, _outputSize }; // -1 means variable batch size
        var outputNode = Autodiff.TensorOperations<T>.Reshape(inputNode, flattenedShape);

        return outputNode;
    }
}
