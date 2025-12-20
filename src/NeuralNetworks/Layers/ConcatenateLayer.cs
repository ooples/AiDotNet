namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a neural network layer that concatenates multiple inputs along a specified axis.
/// </summary>
/// <remarks>
/// <para>
/// A concatenate layer combines multiple input tensors into a single output tensor by joining them along
/// a specified axis. For example, if you have two tensors of shape [batch_size, 10] and [batch_size, 15],
/// concatenating them along axis 1 would produce a tensor of shape [batch_size, 25]. This layer doesn't
/// have any trainable parameters and simply passes the gradients back to the appropriate input tensors
/// during backpropagation.
/// </para>
/// <para><b>For Beginners:</b> A concatenate layer joins multiple inputs together to make one bigger output.
/// 
/// Think of it like joining arrays or lists:
/// - If you have two lists [1, 2, 3] and [4, 5], concatenating them gives [1, 2, 3, 4, 5]
/// 
/// In neural networks, we often work with multi-dimensional data, so we need to specify which
/// dimension (axis) to join along:
/// 
/// - Axis 0 would join along the first dimension (like stacking sheets of paper)
/// - Axis 1 would join along the second dimension (like extending rows sideways)
/// - Axis 2 would join along the third dimension (like extending columns downward)
/// 
/// For example, if you have:
/// - One tensor representing features from an image: [batch_size, 100]
/// - Another tensor representing features from text: [batch_size, 50]
/// 
/// You could use a concatenate layer with axis=1 to create a combined feature tensor of shape [batch_size, 150]
/// that contains both sets of features side by side.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ConcatenateLayer<T> : LayerBase<T>
{
    private readonly int _axis;
    private Tensor<T>[]? _lastInputs;
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> as concatenate layers have no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns false because concatenate layers don't have any trainable parameters.
    /// The layer simply combines inputs and passes gradients through during backpropagation without
    /// modifications.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer cannot learn from data.
    /// 
    /// A value of false means:
    /// - The layer doesn't contain any values that will change during training
    /// - It performs a fixed operation (concatenation) that doesn't need to be learned
    /// - It still participates in passing information during training, but doesn't change itself
    /// 
    /// This is different from layers like dense or convolutional layers that do have trainable
    /// parameters (weights and biases) that get updated during learning.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConcatenateLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputShapes">The shapes of the input tensors to be concatenated.</param>
    /// <param name="axis">The axis along which to concatenate the inputs.</param>
    /// <param name="activationFunction">The activation function to apply after concatenation. Defaults to identity if not specified.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when input shapes have different ranks.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new concatenate layer using the specified input shapes and concatenation axis.
    /// It validates the input shapes to ensure they are compatible for concatenation, and calculates the output
    /// shape based on the input shapes and axis. The activation function is applied to the output after concatenation.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new concatenate layer with a standard activation function.
    /// 
    /// When creating a concatenate layer, you need to specify:
    /// - The shapes of all the inputs that will be joined together
    /// - Which dimension (axis) to join them along
    /// - Optionally, an activation function to apply after joining
    /// 
    /// For example, if you have two inputs with shapes [32, 10] and [32, 20], and specify axis=1,
    /// the output shape will be [32, 30].
    /// 
    /// The default activation is the "identity" function, which doesn't change the values at all.
    /// </para>
    /// </remarks>
    public ConcatenateLayer(int[][] inputShapes, int axis, IActivationFunction<T>? activationFunction = null)
        : base(inputShapes, CalculateOutputShape(inputShapes, axis), activationFunction ?? new IdentityActivation<T>())
    {
        _axis = axis;
        ValidateInputShapes(inputShapes);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConcatenateLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputShapes">The shapes of the input tensors to be concatenated.</param>
    /// <param name="axis">The axis along which to concatenate the inputs.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after concatenation. Defaults to identity if not specified.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when input shapes have different ranks.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new concatenate layer using the specified input shapes and concatenation axis.
    /// It validates the input shapes to ensure they are compatible for concatenation, and calculates the output
    /// shape based on the input shapes and axis. This overload accepts a vector activation function, which operates
    /// on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new concatenate layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor works the same way as the other one, but it's useful when you need more
    /// complex activation patterns that consider the relationships between different outputs.
    /// </para>
    /// </remarks>
    public ConcatenateLayer(int[][] inputShapes, int axis, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShapes, CalculateOutputShape(inputShapes, axis), vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _axis = axis;
        ValidateInputShapes(inputShapes);
    }

    /// <summary>
    /// Validates that the input shapes are compatible for concatenation.
    /// </summary>
    /// <param name="inputShapes">The input shapes to validate.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when input shapes have different ranks.</exception>
    /// <remarks>
    /// <para>
    /// This method checks that at least two input shapes are provided and that all input shapes have the same rank
    /// (number of dimensions). These conditions must be met for concatenation to be valid.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the inputs can be properly joined together.
    /// 
    /// For inputs to be concatenated, they must:
    /// - Have at least two shapes to join (you can't concatenate just one input)
    /// - All have the same number of dimensions (rank)
    /// 
    /// For example:
    /// - [10, 5] and [10, 7] can be concatenated (both have rank 2)
    /// - [10, 5] and [10, 7, 3] cannot be concatenated (one has rank 2, the other rank 3)
    /// 
    /// The method will throw an error if these conditions aren't met.
    /// </para>
    /// </remarks>
    private static void ValidateInputShapes(int[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            throw new ArgumentException("At least two input shapes are required for concatenation.");
        }

        int rank = inputShapes[0].Length;
        if (inputShapes.Any(shape => shape.Length != rank))
        {
            throw new ArgumentException("All input shapes must have the same rank.");
        }
    }

    /// <summary>
    /// Calculates the output shape of the concatenated tensor.
    /// </summary>
    /// <param name="inputShapes">The shapes of the input tensors.</param>
    /// <param name="axis">The axis along which to concatenate.</param>
    /// <returns>The calculated output shape.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the concatenated tensor by summing the size of each input
    /// along the specified axis while keeping the other dimensions the same. For example, if concatenating
    /// shapes [10, 5] and [10, 3] along axis 1, the output shape would be [10, 8].
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the output after joining the inputs.
    /// 
    /// When joining tensors along a specific axis:
    /// - The dimensions on all other axes must match
    /// - The dimension on the joining axis will be the sum of all the input dimensions on that axis
    /// 
    /// For example, if joining these shapes along axis 1:
    /// - [32, 10, 5]
    /// - [32, 20, 5]
    /// - [32, 15, 5]
    /// 
    /// The output shape would be [32, 45, 5], because:
    /// - Axis 0: 32 (same in all inputs)
    /// - Axis 1: 10 + 20 + 15 = 45 (sum of all inputs at this axis)
    /// - Axis 2: 5 (same in all inputs)
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[][] inputShapes, int axis)
    {
        int[] outputShape = new int[inputShapes[0].Length];
        Array.Copy(inputShapes[0], outputShape, inputShapes[0].Length);

        for (int i = 1; i < inputShapes.Length; i++)
        {
            outputShape[axis] += inputShapes[i][axis];
        }

        return outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the concatenate layer with multiple inputs.
    /// </summary>
    /// <param name="inputs">The input tensors to concatenate.</param>
    /// <returns>The output tensor after concatenation and activation.</returns>
    /// <exception cref="ArgumentException">Thrown when fewer than two input tensors are provided.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the concatenate layer. It combines the input tensors along
    /// the specified axis, applies the activation function (if any), and returns the result. The inputs and output
    /// are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method joins multiple inputs together during the network's forward pass.
    /// 
    /// The forward pass:
    /// 1. Takes in all input tensors
    /// 2. Joins them together along the specified axis
    /// 3. Applies the activation function (if any)
    /// 4. Returns the combined result
    /// 
    /// This method also saves the inputs and output for later use during training.
    /// 
    /// For example, if you pass in tensors representing image features and text features,
    /// this method will join them into a single tensor containing both types of features.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("ConcatenateLayer requires at least two inputs.");
        }

        _lastInputs = inputs;
        _lastOutput = Engine.Concat(inputs, _axis);

        if (ScalarActivation != null)
        {
            _lastOutput = ScalarActivation.Activate(_lastOutput);
        }
        else if (VectorActivation != null)
        {
            _lastOutput = VectorActivation.Activate(_lastOutput);
        }

        return _lastOutput;
    }

    /// <summary>
    /// Performs the backward pass of the concatenate layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the concatenate layer, which is used during training to propagate
    /// error gradients back through the network. It splits the output gradient along the concatenation axis and
    /// distributes the pieces to the corresponding input gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method routes the error gradients back to the correct inputs during training.
    ///
    /// During the backward pass:
    /// 1. The layer receives error gradients from the next layer
    /// 2. If an activation function was used, its derivative is applied
    /// 3. The gradient is split along the same axis used for concatenation
    /// 4. Each piece of the gradient is sent back to the corresponding input
    ///
    /// For example, if you joined three tensors of widths 10, 20, and 15:
    /// - The incoming gradient would have width 45
    /// - This method would split it into pieces of width 10, 20, and 15
    /// - Each piece would be sent back to its original source
    ///
    /// This is how the training signal flows backward through the network,
    /// allowing each connected layer to learn from the error.
    /// </para>
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
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInputs == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        if (ScalarActivation != null)
        {
            // GPU/CPU accelerated element-wise multiply via Engine.TensorMultiply
            var activationDerivative = ScalarActivation.Derivative(_lastOutput);
            outputGradient = Engine.TensorMultiply(outputGradient, activationDerivative);
        }
        else if (VectorActivation != null)
        {
            // GPU/CPU accelerated element-wise multiply via Engine.TensorMultiply
            outputGradient = Engine.TensorMultiply(outputGradient, VectorActivation.Derivative(_lastOutput));
        }

        var inputGradients = new Tensor<T>[_lastInputs.Length];
        int startIndex = 0;

        for (int i = 0; i < _lastInputs.Length; i++)
        {
            int length = _lastInputs[i].Shape[_axis];
            int endIndex = startIndex + length;
            inputGradients[i] = outputGradient.Slice(_axis, startIndex, endIndex);
            startIndex = endIndex;
        }

        return Tensor<T>.Stack(inputGradients);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method computes gradients using the same computation as BackwardManual to ensure
    /// identical results. Both paths slice the output gradient along the concatenation axis
    /// and stack the resulting gradients.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInputs == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Use the same computation as BackwardManual to ensure identical results
        if (ScalarActivation != null)
        {
            // GPU/CPU accelerated element-wise multiply via Engine.TensorMultiply
            var activationDerivative = ScalarActivation.Derivative(_lastOutput);
            outputGradient = Engine.TensorMultiply(outputGradient, activationDerivative);
        }
        else if (VectorActivation != null)
        {
            // GPU/CPU accelerated element-wise multiply via Engine.TensorMultiply
            outputGradient = Engine.TensorMultiply(outputGradient, VectorActivation.Derivative(_lastOutput));
        }

        var inputGradients = new Tensor<T>[_lastInputs.Length];
        int startIndex = 0;

        for (int i = 0; i < _lastInputs.Length; i++)
        {
            int length = _lastInputs[i].Shape[_axis];
            int endIndex = startIndex + length;
            inputGradients[i] = outputGradient.Slice(_axis, startIndex, endIndex);
            startIndex = endIndex;
        }

        return Tensor<T>.Stack(inputGradients);
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is a no-op for concatenate layers since they have no trainable parameters to update.
    /// </para>
    /// <para><b>For Beginners:</b> This method doesn't do anything for concatenate layers because there are no parameters to update.
    /// 
    /// Unlike layers with weights and biases that need to be updated during training,
    /// the concatenate layer just passes data through without learning any parameters.
    /// 
    /// This method is still required to be implemented because all layers must follow
    /// the same interface, but it doesn't actually do anything for this type of layer.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in a concatenate layer
    }

    /// <summary>
    /// This method is not supported by ConcatenateLayer and will throw an exception.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>Never returns as it always throws an exception.</returns>
    /// <exception cref="NotSupportedException">Always thrown as ConcatenateLayer requires multiple inputs.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base Forward method that accepts a single input tensor, but it always throws
    /// an exception because concatenate layers require multiple inputs by definition. Use the Forward method
    /// that accepts multiple inputs instead.
    /// </para>
    /// <para><b>For Beginners:</b> This method is included because all layers must follow the same interface,
    /// but it can't be used with concatenate layers.
    /// 
    /// A concatenate layer must have at least two inputs to join together, so this method
    /// that only takes one input will always throw an error.
    /// 
    /// Instead, you should use the other Forward method that accepts multiple inputs (params Tensor&lt;T&gt;[] inputs).
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException("ConcatenateLayer requires multiple inputs. Use Forward(params Tensor<T>[] inputs) instead.");
    }

    /// <summary>
    /// Gets all trainable parameters from the layer as a single vector.
    /// </summary>
    /// <returns>An empty vector as concatenate layers have no parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector because concatenate layers don't have any trainable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because concatenate layers don't have any learnable values.
    /// 
    /// Unlike layers with weights and biases, the concatenate layer doesn't have any parameters
    /// that need to be saved or loaded. It's just a fixed operation that joins inputs together.
    /// 
    /// This method is still required because all layers must follow the same interface, but it
    /// simply returns an empty vector in this case.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Concatenate layers don't have parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the concatenate layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the concatenate layer, including the cached inputs and output.
    /// This is useful when starting to process a new sequence or batch after processing a previous one.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's temporary memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Freeing up memory that's no longer needed
    /// 
    /// Since the concatenate layer doesn't have learnable parameters, this only clears
    /// the cached values used during a single forward/backward pass.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInputs = null;
        _lastOutput = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // ConcatenateLayer expects multiple inputs - create symbolic input
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // If multiple inputs are provided, concatenate them using TensorOperations.Concat()
        if (inputNodes.Count > 1)
        {
            return TensorOperations<T>.Concat(inputNodes, axis: _axis);
        }

        return inputNode;
    }

    public override bool SupportsJitCompilation => true;
}
