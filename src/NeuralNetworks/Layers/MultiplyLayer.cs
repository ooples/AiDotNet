using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that performs element-wise multiplication of multiple input tensors.
/// </summary>
/// <remarks>
/// <para>
/// The MultiplyLayer performs element-wise multiplication (Hadamard product) of two or more input tensors
/// of identical shape. This operation can be useful for implementing gating mechanisms, attention masks,
/// or feature-wise interactions in neural networks. The layer requires that all input tensors have the
/// same shape, and it produces an output tensor of that same shape.
/// </para>
/// <para><b>For Beginners:</b> This layer multiplies tensors together, element by element.
/// 
/// Think of it like multiplying numbers together in corresponding positions:
/// - If you have two vectors [1, 2, 3] and [4, 5, 6]
/// - The result would be [1×4, 2×5, 3×6] = [4, 10, 18]
/// 
/// This is useful for:
/// - Controlling information flow (like gates in LSTM or GRU cells)
/// - Applying masks (to selectively focus on certain values)
/// - Combining features in a multiplicative way
/// 
/// For example, in an attention mechanism, you might multiply feature values by attention weights
/// to focus on important features and diminish the influence of less relevant ones.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MultiplyLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The input tensors from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input tensors from the most recent forward pass, which are needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>[]? _lastInputs;

    /// <summary>
    /// The output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the output tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because the MultiplyLayer supports backpropagation, even though it has no parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer supports backpropagation during training. Although
    /// the MultiplyLayer has no trainable parameters, it still supports the backward pass to propagate
    /// gradients to previous layers.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can participate in the training process.
    /// 
    /// A value of true means:
    /// - The layer can pass gradient information backward during training
    /// - It's part of the learning process, even though it doesn't have learnable parameters
    /// 
    /// While this layer doesn't have weights or biases that get updated during training,
    /// it still needs to properly handle gradients to ensure that layers before it
    /// can learn correctly.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiplyLayer{T}"/> class with the specified input shapes
    /// and a scalar activation function.
    /// </summary>
    /// <param name="inputShapes">An array of input shapes, all of which must be identical.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when input shapes are not identical.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a MultiplyLayer that expects multiple input tensors with identical shapes.
    /// It validates that at least two input shapes are provided and that all shapes are identical, since
    /// element-wise multiplication requires matching dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer to handle multiple inputs of the same shape.
    /// 
    /// When creating a MultiplyLayer, you need to specify:
    /// - inputShapes: The shapes of all the inputs you'll provide (which must match)
    /// - activationFunction: The function that processes the final output (optional)
    /// 
    /// For example, if you want to multiply three tensors with shape [32, 10, 128]:
    /// - You would specify inputShapes as [[32, 10, 128], [32, 10, 128], [32, 10, 128]]
    /// - The layer would validate that all these shapes match
    /// - The output shape would also be [32, 10, 128]
    /// 
    /// The constructor throws an exception if you provide fewer than two input shapes
    /// or if the shapes don't all match exactly.
    /// </para>
    /// </remarks>
    public MultiplyLayer(int[][] inputShapes, IActivationFunction<T>? activationFunction = null)
        : base(inputShapes, inputShapes[0], activationFunction ?? new IdentityActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiplyLayer{T}"/> class with the specified input shapes
    /// and a vector activation function.
    /// </summary>
    /// <param name="inputShapes">An array of input shapes, all of which must be identical.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when input shapes are not identical.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a MultiplyLayer that expects multiple input tensors with identical shapes.
    /// It validates that at least two input shapes are provided and that all shapes are identical, since
    /// element-wise multiplication requires matching dimensions. This overload accepts a vector activation
    /// function, which operates on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different values after multiplication.
    /// </para>
    /// </remarks>
    public MultiplyLayer(int[][] inputShapes, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShapes, inputShapes[0], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    /// <summary>
    /// Validates that the input shapes are appropriate for a multiply layer.
    /// </summary>
    /// <param name="inputShapes">The array of input shapes to validate.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when input shapes are not identical.</exception>
    /// <remarks>
    /// <para>
    /// This method validates that at least two input shapes are provided and that all shapes are identical.
    /// Element-wise multiplication requires that all tensors have the same dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the input shapes are valid for multiplication.
    /// 
    /// For element-wise multiplication to work:
    /// - You need at least two tensors (you can't multiply just one tensor)
    /// - All tensors must have exactly the same shape (dimensions)
    /// 
    /// For example:
    /// - Valid: Shapes [3,4] and [3,4]
    /// - Invalid: Shapes [3,4] and [3,5] (different second dimension)
    /// - Invalid: Shapes [3,4] and [4,3] (dimensions swapped)
    /// 
    /// If these requirements aren't met, the method throws an exception with a helpful error message.
    /// </para>
    /// </remarks>
    private static void ValidateInputShapes(int[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            throw new ArgumentException("MultiplyLayer requires at least two inputs.");
        }
        for (int i = 1; i < inputShapes.Length; i++)
        {
            if (!inputShapes[i].SequenceEqual(inputShapes[0]))
            {
                throw new ArgumentException("All input shapes must be identical for MultiplyLayer.");
            }
        }
    }

    /// <summary>
    /// This method is not supported for MultiplyLayer as it requires multiple input tensors.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>Not applicable as this method throws an exception.</returns>
    /// <exception cref="NotSupportedException">Always thrown when this method is called.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base Forward method but is not supported for MultiplyLayer because
    /// element-wise multiplication requires multiple input tensors. Calling this method will always
    /// result in a NotSupportedException.
    /// </para>
    /// <para><b>For Beginners:</b> This method exists to satisfy the base class requirements but should not be used.
    /// 
    /// Since the MultiplyLayer needs multiple input tensors to work properly,
    /// this simplified version that only takes a single input tensor cannot function correctly.
    /// 
    /// If you call this method, you'll get an error message directing you to use the
    /// correct Forward method that accepts multiple input tensors.
    /// 
    /// Always use Forward(params Tensor<T>[] inputs) instead of Forward(input) with this layer.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException("MultiplyLayer requires multiple inputs. Use Forward(params Tensor<T>[] inputs) instead.");
    }

    /// <summary>
    /// Performs the forward pass of the multiply layer with multiple input tensors.
    /// </summary>
    /// <param name="inputs">The array of input tensors to multiply.</param>
    /// <returns>The output tensor after element-wise multiplication and activation.</returns>
    /// <exception cref="ArgumentException">Thrown when fewer than two input tensors are provided.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the multiply layer. It performs element-wise multiplication
    /// of all input tensors, then applies the activation function to the result. The input tensors and output
    /// tensor are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs the actual multiplication operation.
    /// 
    /// During the forward pass:
    /// - The method checks that you've provided at least two input tensors
    /// - It makes a copy of the first input tensor as the starting point
    /// - It then multiplies this copy element-by-element with each of the other input tensors
    /// - Finally, it applies the activation function to the result
    /// 
    /// For example, with inputs [1,2,3], [4,5,6], and [0.5,0.5,0.5]:
    /// 1. Start with [1,2,3]
    /// 2. Multiply by [4,5,6] to get [4,10,18]
    /// 3. Multiply by [0.5,0.5,0.5] to get [2,5,9]
    /// 4. Apply activation function (if any)
    /// 
    /// The method also saves all inputs and the output for later use in backpropagation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("MultiplyLayer requires at least two inputs.");
        }

        _lastInputs = inputs;

        // Use Engine.TensorMultiplyMany for GPU/CPU accelerated element-wise multiplication of all tensors
        // This is production-grade: no loops, single optimized call that batches all multiplications
        var result = Engine.TensorMultiplyMany(inputs);

        var activated = ApplyActivation(result);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = activated;
        }

        return activated;
    }

    /// <summary>
    /// Performs the forward pass on GPU using actual GPU element-wise multiplication.
    /// </summary>
    /// <param name="inputs">The GPU input tensors.</param>
    /// <returns>The GPU output tensor.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length < 2)
            throw new ArgumentException("MultiplyLayer requires at least two inputs.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable");

        int size = inputs[0].ElementCount;

        // Perform GPU element-wise multiplication of all inputs
        // Start with first two inputs
        var resultBuffer = backend.AllocateBuffer(size);
        backend.Multiply(inputs[0].Buffer, inputs[1].Buffer, resultBuffer, size);

        // Multiply with remaining inputs
        for (int i = 2; i < inputs.Length; i++)
        {
            using var tempBuffer = resultBuffer;
            resultBuffer = backend.AllocateBuffer(size);
            backend.Multiply(tempBuffer, inputs[i].Buffer, resultBuffer, size);
        }

        // Apply activation if needed
        var fusedActivation = GetFusedActivationType();
        if (fusedActivation != FusedActivationType.None)
        {
            var activatedBuffer = backend.AllocateBuffer(size);
            ApplyGpuActivation(backend, resultBuffer, activatedBuffer, size, fusedActivation);
            resultBuffer.Dispose();
            resultBuffer = activatedBuffer;
        }

        // Store for backward pass during training
        if (IsTrainingMode)
        {
            var cpuInputs = new Tensor<T>[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cpuInputs[i] = inputs[i].ToTensor();
            _lastInputs = cpuInputs;
        }

        return new GpuTensor<T>(backend, resultBuffer, inputs[0].Shape, GpuTensorRole.Activation, ownsBuffer: true);
    }

    /// <inheritdoc/>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");
        }

        var backend = gpuEngine.GetBackend();
        if (backend is null)
        {
            throw new InvalidOperationException("GPU backend unavailable.");
        }

        // CPU fallback: download gradient, compute via CPU Backward, upload result
        var outputGradCpu = outputGradient.ToTensor();
        var inputGradCpu = Backward(outputGradCpu);

        // Return the input gradient as GPU tensor
        return new GpuTensor<T>(backend, inputGradCpu, GpuTensorRole.Gradient);
    }

    /// <summary>
    /// Computes the gradients of the loss with respect to the inputs on the GPU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>Array of gradients for each input tensor.</returns>
    /// <remarks>
    /// For element-wise multiplication z = x * y, the gradient with respect to each input
    /// is the product of the output gradient and all other inputs.
    /// </remarks>
    public IGpuTensor<T>[] BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_lastInputs == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend unavailable.");

        int size = outputGradient.ElementCount;
        int numInputs = _lastInputs.Length;

        // Upload cached inputs to GPU
        var gpuInputBuffers = new IGpuBuffer[numInputs];
        for (int i = 0; i < numInputs; i++)
        {
            float[] floatData = DirectGpuEngine.ToFloatArray(_lastInputs[i].Data);
            gpuInputBuffers[i] = backend.AllocateBuffer(floatData);
        }

        // Compute gradient for each input
        // Gradient for input i = outputGradient * product(inputs[j] for j != i)
        var inputGradients = new IGpuTensor<T>[numInputs];
        for (int i = 0; i < numInputs; i++)
        {
            // Start with output gradient
            var gradBuffer = backend.AllocateBuffer(size);
            backend.Copy(outputGradient.Buffer, gradBuffer, size);

            // Multiply by all other inputs
            for (int j = 0; j < numInputs; j++)
            {
                if (i != j)
                {
                    var tempBuffer = backend.AllocateBuffer(size);
                    backend.Multiply(gradBuffer, gpuInputBuffers[j], tempBuffer, size);
                    gradBuffer.Dispose();
                    gradBuffer = tempBuffer;
                }
            }

            inputGradients[i] = new GpuTensor<T>(backend, gradBuffer, outputGradient.Shape, GpuTensorRole.Gradient, ownsBuffer: true);
        }

        // Dispose uploaded input buffers
        foreach (var buffer in gpuInputBuffers)
        {
            buffer.Dispose();
        }

        return inputGradients;
    }

    /// <summary>
    /// Performs the backward pass of the multiply layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's inputs.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the multiply layer, which is used during training to propagate
    /// error gradients back through the network. For element-wise multiplication, the gradient with respect to
    /// each input is the product of the output gradient and all other inputs. The method calculates and returns
    /// the gradients for all input tensors.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in each input affect the final output.
    ///
    /// During the backward pass:
    /// - The layer receives gradients indicating how the output should change
    /// - It calculates how each input tensor contributed to the output
    /// - For each input, its gradient is the product of:
    ///   - The output gradient (after applying the activation function derivative)
    ///   - All OTHER input tensors (not including itself)
    ///
    /// This follows the chain rule of calculus for multiplication:
    /// If z = x * y, then:
    /// - dz/dx = y * (gradient flowing back from later layers)
    /// - dz/dy = x * (gradient flowing back from later layers)
    ///
    /// The method returns a stacked tensor containing gradients for all inputs.
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

        Tensor<T> activationGradient;
        if (UsingVectorActivation && VectorActivation != null)
        {
            // Use element-wise multiplication for gradient computation
            activationGradient = Tensor<T>.ElementwiseMultiply(VectorActivation.Derivative(_lastOutput), outputGradient);
        }
        else if (ScalarActivation != null)
        {
            // Vectorized: compute activation derivatives and multiply element-wise using Engine
            var derivatives = ScalarActivation.Derivative(_lastOutput);
            activationGradient = Engine.TensorMultiply(derivatives, outputGradient);
        }
        else
        {
            activationGradient = outputGradient;
        }

        var inputGradients = new Tensor<T>[_lastInputs.Length];
        for (int i = 0; i < _lastInputs.Length; i++)
        {
            inputGradients[i] = activationGradient.Clone();
            for (int j = 0; j < _lastInputs.Length; j++)
            {
                if (i != j)
                {
                    // GPU/CPU accelerated element-wise multiply via Engine.TensorMultiply
                    inputGradients[i] = Engine.TensorMultiply(inputGradients[i], _lastInputs[j]);
                }
            }
        }
        return Tensor<T>.Stack(inputGradients);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation with production-grade optimizations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's inputs (stacked).</returns>
    /// <remarks>
    /// <para>
    /// This method uses a production-grade pattern for computing gradients:
    /// - Uses cached values from forward pass (locality caching)
    /// - Builds full computation graph including multiplication and activation
    /// - Executes inline topological sort for graph traversal
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInputs == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // If vector activation is configured, fall back to manual path
        if (VectorActivation != null)
        {
            return BackwardManual(outputGradient);
        }

        // 1. Create variable nodes for inputs that need gradients
        var inputNodes = new List<ComputationNode<T>>();
        for (int i = 0; i < _lastInputs.Length; i++)
        {
            inputNodes.Add(TensorOperations<T>.Variable(_lastInputs[i], $"input_{i}", requiresGradient: true));
        }

        // 2. Build computation graph (Element-wise Multiply)
        ComputationNode<T> productNode = inputNodes[0];
        for (int i = 1; i < inputNodes.Count; i++)
        {
            productNode = TensorOperations<T>.ElementwiseMultiply(productNode, inputNodes[i]);
        }

        // Apply activation function using LayerBase helper
        var outputNode = ApplyActivationToGraph(productNode);

        // 3. Set output gradient
        outputNode.Gradient = outputGradient;

        // 4. Inline topological sort
        var visited = new HashSet<ComputationNode<T>>();
        var topoOrder = new List<ComputationNode<T>>();
        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
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

        // 5. Execute backward pass
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // 6. Collect and stack gradients to match expected output format
        var gradients = new Tensor<T>[inputNodes.Count];
        for (int i = 0; i < inputNodes.Count; i++)
        {
            gradients[i] = inputNodes[i].Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
        }

        return Tensor<T>.Stack(gradients);
    }

    /// <summary>
    /// Updates the parameters of the multiply layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is part of the training process, but since MultiplyLayer has no trainable parameters,
    /// this method does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally update a layer's internal values during training.
    /// 
    /// However, since MultiplyLayer just performs a fixed mathematical operation (multiplication) and doesn't
    /// have any internal values that can be learned or adjusted, this method is empty.
    /// 
    /// This is unlike layers such as Dense or Convolutional layers, which have weights and biases
    /// that get updated during training.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all trainable parameters from the multiply layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since MultiplyLayer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. Since MultiplyLayer
    /// has no trainable parameters, it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the learnable values in the layer.
    /// 
    /// Since MultiplyLayer:
    /// - Only performs fixed mathematical operations (multiplication)
    /// - Has no weights, biases, or other learnable parameters
    /// - The method returns an empty list
    /// 
    /// This is different from layers like Dense layers, which would return their weights and biases.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // MultiplyLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the multiply layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the multiply layer, including the cached inputs and output.
    /// This is useful when starting to process a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous processing are cleared
    /// - The layer forgets any information from previous data batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Ensuring clean state before a new training epoch
    /// - Preventing information from one batch affecting another
    /// 
    /// While the MultiplyLayer doesn't maintain long-term state across samples,
    /// clearing these cached values helps with memory management and ensuring a clean processing pipeline.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInputs = null;
        _lastOutput = null;
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

        if (inputNodes.Count > 1)
        {
            var result = inputNodes[0];
            for (int i = 1; i < inputNodes.Count; i++)
            {
                result = TensorOperations<T>.ElementwiseMultiply(result, inputNodes[i]);
            }
            return result;
        }

        return inputNode;
    }

    public override bool SupportsJitCompilation => true;
}
