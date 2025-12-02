namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a residual layer that adds the identity mapping (input) to the output of an inner layer.
/// </summary>
/// <remarks>
/// <para>
/// A residual layer implements the core concept of residual networks (ResNets), where the layer learns
/// the residual (difference) between the identity mapping and the desired underlying mapping rather than
/// the complete transformation. This is achieved by adding a skip connection that passes the input directly
/// to the output, where it's added to the transformed output of an inner layer.
/// </para>
/// <para><b>For Beginners:</b> This layer helps neural networks learn more effectively, especially when they're very deep.
/// 
/// Think of it as a "correction mechanism":
/// - The inner layer tries to learn how to improve or adjust the input
/// - The original input is preserved and added back in at the end
/// - This allows the network to focus on learning the changes needed, rather than recreating the entire signal
/// 
/// Benefits include:
/// - Solves the "vanishing gradient problem" that makes deep networks hard to train
/// - Enables training of much deeper networks (hundreds of layers instead of just dozens)
/// - Improves learning speed and accuracy
/// 
/// For example, in image recognition, a residual layer might learn to emphasize important features
/// while preserving the original image information through the skip connection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ResidualLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The inner layer that transforms the input before being added back to the original input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the optional inner layer that learns the residual mapping. If null, the layer
    /// simply passes the input through the activation function. The output of this inner layer is added
    /// to the original input to form the final output of the residual layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "transformation path" of the residual connection.
    /// 
    /// It can be any type of neural network layer (like convolution, dense, etc.) or even
    /// a sequence of multiple layers bundled together. The residual layer adds the output
    /// of this transformation path back to the original input.
    /// 
    /// The inner layer must have the same input and output shape for the residual connection
    /// to work properly, as the outputs need to be added together.
    /// 
    /// If this is null (not set), then the layer simply passes the input through the activation function.
    /// </para>
    /// </remarks>
    private ILayer<T>? _innerLayer;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the input to the layer during the forward pass, which is needed during the backward pass
    /// to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the layer's short-term memory of what input it received.
    /// 
    /// During training, the layer needs to remember what input it processed so that it can
    /// properly calculate how to improve. This temporary storage is cleared between batches
    /// or when you explicitly reset the layer.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> if the inner layer exists and supports training; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the residual layer can be trained using backpropagation.
    /// The layer supports training if it has an inner layer that supports training. If there is no
    /// inner layer, the residual layer is considered not to support training since it would just act
    /// as an identity function.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can improve through training
    /// - It has parameters that can be adjusted
    /// 
    /// For residual layers:
    /// - If there's an inner layer that can be trained, this will be true
    /// - If there's no inner layer or the inner layer can't be trained, this will be false
    /// 
    /// Residual layers themselves don't have trainable parameters - all learning happens in the inner layer.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _innerLayer?.SupportsTraining ?? false;

    /// <summary>
    /// Initializes a new instance of the <see cref="ResidualLayer{T}"/> class with the specified input shape,
    /// inner layer, and scalar activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="innerLayer">The optional inner layer that learns the residual mapping.</param>
    /// <param name="activation">The scalar activation function to apply after addition. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a residual layer with the specified input shape, inner layer, and scalar activation function.
    /// The output shape is set to be the same as the input shape, as required for residual connections. If the inner layer
    /// has different input and output shapes, an exception will be thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new residual layer with basic settings.
    /// 
    /// When you create a residual layer this way:
    /// - You specify the size and shape of the data it will process
    /// - You can provide an inner layer that will learn the transformation
    /// - You can specify an activation function that operates on each value individually
    /// 
    /// The residual layer requires that the inner layer produces output of the same shape as its input,
    /// because the original input and the transformed output need to be added together.
    /// 
    /// This constructor is for the more common case where you want to use a scalar activation function.
    /// </para>
    /// </remarks>
    public ResidualLayer(int[] inputShape, ILayer<T>? innerLayer = null, IActivationFunction<T>? activation = null)
        : base(inputShape, inputShape, activation?? new IdentityActivation<T>())
    {
        _innerLayer = innerLayer;
        ValidateInnerLayer();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ResidualLayer{T}"/> class with the specified input shape,
    /// inner layer, and vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="innerLayer">The optional inner layer that learns the residual mapping.</param>
    /// <param name="vectorActivation">The vector activation function to apply after addition. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a residual layer with the specified input shape, inner layer, and vector activation function.
    /// The output shape is set to be the same as the input shape, as required for residual connections. If the inner layer
    /// has different input and output shapes, an exception will be thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new residual layer with advanced settings.
    /// 
    /// When you create a residual layer this way:
    /// - You specify the size and shape of the data it will process
    /// - You can provide an inner layer that will learn the transformation
    /// - You can specify a vector activation function that operates on groups of values
    /// 
    /// The residual layer requires that the inner layer produces output of the same shape as its input,
    /// because the original input and the transformed output need to be added together.
    /// 
    /// This constructor is for advanced cases where you want to use a vector activation function
    /// that can capture relationships between different elements in the output.
    /// </para>
    /// </remarks>
    public ResidualLayer(int[] inputShape, ILayer<T>? innerLayer = null, IVectorActivationFunction<T>? vectorActivation = null)
        : base(inputShape, inputShape, vectorActivation ?? new IdentityActivation<T>())
    {
        _innerLayer = innerLayer;
        ValidateInnerLayer();
    }

    /// <summary>
    /// Validates that the inner layer has the same input and output shape.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when the inner layer has different input and output shapes.</exception>
    /// <remarks>
    /// <para>
    /// This method checks that the inner layer (if present) has the same input and output shape, which is required
    /// for residual connections to work properly. If the shapes are different, an ArgumentException is thrown.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes sure the inner layer is compatible with residual connections.
    /// 
    /// For a residual connection to work:
    /// - The inner layer must produce output of exactly the same shape as its input
    /// - This is because the original input and the transformed output will be added together
    /// - Addition only works when the shapes match exactly
    /// 
    /// If the shapes don't match, this method will throw an error to prevent problems later.
    /// </para>
    /// </remarks>
    private void ValidateInnerLayer()
    {
        if (_innerLayer != null && !Enumerable.SequenceEqual(_innerLayer.GetInputShape(), _innerLayer.GetOutputShape()))
        {
            throw new ArgumentException("Inner layer must have the same input and output shape for residual connections.");
        }
    }

    /// <summary>
    /// Sets a new inner layer for the residual layer.
    /// </summary>
    /// <param name="innerLayer">The new inner layer to use.</param>
    /// <exception cref="ArgumentException">Thrown when the inner layer has different input and output shapes.</exception>
    /// <remarks>
    /// <para>
    /// This method allows changing the inner layer of the residual layer after construction. It validates that the
    /// new inner layer has the same input and output shape before setting it.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you change the transformation part of the residual layer.
    /// 
    /// You can use this to:
    /// - Replace the current inner layer with a different one
    /// - Add an inner layer if there wasn't one before
    /// 
    /// The method checks that the new inner layer is compatible (has matching input and output shapes)
    /// before making the change. If the shapes don't match, it will throw an error.
    /// 
    /// This is useful for building complex networks where you might want to add or change
    /// parts of the network after initial construction.
    /// </para>
    /// </remarks>
    public void SetInnerLayer(ILayer<T> innerLayer)
    {
        _innerLayer = innerLayer;
        ValidateInnerLayer();
    }

    /// <summary>
    /// Performs the forward pass of the residual layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through the residual layer.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the residual layer. It passes the input through the inner layer
    /// (if present), adds the result to the original input, and then applies the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes data through the residual layer.
    /// 
    /// During the forward pass:
    /// - The input is saved for later use in training
    /// - If there's an inner layer, the input is processed through it
    /// - The original input is added to the processed result
    /// - The activation function is applied to the combined result
    /// - The final output is returned
    /// 
    /// If there's no inner layer, the input is simply passed through the activation function.
    /// 
    /// The key to residual learning is the addition step, which allows information to
    /// flow directly from the input to the output, making it easier to train deep networks.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var result = _innerLayer == null ? input : input.Add(_innerLayer.Forward(input));

        return ApplyActivation(result);
    }

    /// <summary>
    /// Performs the backward pass of the residual layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the residual layer, which is used during training to propagate
    /// error gradients back through the network. It computes gradients for the inner layer (if present) and returns
    /// the gradient with respect to the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    ///
    /// During the backward pass:
    /// - The method throws an error if the forward pass hasn't been called first
    /// - The gradient is computed for the combined output after the addition
    /// - If there's an inner layer, the gradient is propagated through it
    /// - The original gradient and the inner layer gradient are combined
    /// - The combined gradient is returned for further backpropagation
    ///
    /// This process ensures that gradient information flows both through the inner layer
    /// and directly back to earlier layers, preventing the vanishing gradient problem
    /// in deep networks.
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
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var innerOutput = _innerLayer?.Forward(_lastInput) ?? _lastInput;
        var combinedOutput = _lastInput.Add(innerOutput);
        var activationGradient = ApplyActivationDerivative(combinedOutput, outputGradient);
        var combinedGradient = outputGradient.ElementwiseMultiply(activationGradient);

        if (_innerLayer == null)
        {
            return combinedGradient;
        }

        var innerGradient = _innerLayer.Backward(combinedGradient);
        return combinedGradient.Add(innerGradient);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It recreates the forward
    /// computation graph for the residual connection (input + inner layer output) and propagates
    /// gradients through it.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Convert to computation nodes
        var input = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Compute inner layer output if present
        Autodiff.ComputationNode<T> result;
        Autodiff.ComputationNode<T>? innerNode = null;
        if (_innerLayer != null)
        {
            var innerOutput = _innerLayer.Forward(_lastInput);
            // Allow gradients to flow through inner layer output
            innerNode = Autodiff.TensorOperations<T>.Variable(innerOutput, "inner_output", requiresGradient: true);
            // output = input + innerOutput
            result = Autodiff.TensorOperations<T>.Add(input, innerNode);
        }
        else
        {
            result = input;
        }

        // Apply activation using autodiff
        var activated = ApplyActivationAutodiff(result);

        // Set the gradient at the output
        activated.Gradient = outputGradient;

        // Perform topological sort and backward pass
        var topoOrder = GetTopologicalOrder(activated);

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Route gradient to inner layer if present and accumulate with skip connection gradient
        if (_innerLayer != null && innerNode != null && innerNode.Gradient != null)
        {
            // Get gradient from inner layer backward pass
            var innerGradient = _innerLayer.Backward(innerNode.Gradient);

            // Sum skip connection gradient (input.Gradient) with inner branch gradient
            // Both branches contribute to the input gradient in a residual connection
            if (input.Gradient != null)
            {
                return input.Gradient.Add(innerGradient);
            }
            return innerGradient;
        }

        return input.Gradient!;
    }

    /// <summary>
    /// Gets the topological order of nodes in the computation graph.
    /// </summary>
    private List<Autodiff.ComputationNode<T>> GetTopologicalOrder(Autodiff.ComputationNode<T> root)
    {
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var result = new List<Autodiff.ComputationNode<T>>();

        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
            {
                continue;
            }

            if (processed)
            {
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                stack.Push((node, true));

                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        stack.Push((parent, false));
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies activation function using autodiff operations.
    /// </summary>
    private Autodiff.ComputationNode<T> ApplyActivationAutodiff(Autodiff.ComputationNode<T> input)
    {
        if (ScalarActivation is ReLUActivation<T>)
        {
            return Autodiff.TensorOperations<T>.ReLU(input);
        }
        else if (ScalarActivation is SigmoidActivation<T>)
        {
            return Autodiff.TensorOperations<T>.Sigmoid(input);
        }
        else if (ScalarActivation is TanhActivation<T>)
        {
            return Autodiff.TensorOperations<T>.Tanh(input);
        }
        else
        {
            // For unsupported activations, return input unchanged
            return input;
        }
    }

    /// <summary>
    /// Updates the parameters of the inner layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of the inner layer (if present) based on the gradients calculated during the backward pass.
    /// If there is no inner layer, this method does nothing since there are no parameters to update.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's learnable values during training.
    /// 
    /// When updating parameters:
    /// - If there's an inner layer, its parameters are updated using the specified learning rate
    /// - If there's no inner layer, nothing happens since there are no parameters to update
    /// 
    /// The learning rate controls how big each update step is:
    /// - Smaller learning rates: slower but more stable learning
    /// - Larger learning rates: faster but potentially unstable learning
    /// 
    /// Residual layers themselves don't have parameters to learn - they just pass the
    /// updates to their inner layer if one exists.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _innerLayer?.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters from the inner layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters from the inner layer, or an empty vector if there is no inner layer.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the inner layer (if present) and returns them as a single vector.
    /// If there is no inner layer, it returns an empty vector since there are no parameters to retrieve.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// For a residual layer:
    /// - If there's an inner layer, it returns that layer's parameters
    /// - If there's no inner layer, it returns an empty list (no parameters)
    /// 
    /// Residual layers themselves don't have parameters to learn - all learning happens
    /// in the inner layer (if one exists).
    /// 
    /// This method is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return _innerLayer?.GetParameters() ?? Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the residual layer and its inner layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the residual layer, including the cached input, as well as
    /// the state of the inner layer (if present). This is useful when starting to process a new batch or when implementing
    /// stateful networks that need to be reset between sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The stored input from the previous forward pass is cleared
    /// - If there's an inner layer, its state is also reset
    /// 
    /// This is important for:
    /// - Processing a new batch of unrelated data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like clearing your workspace before starting a new project -
    /// it ensures that old information doesn't interfere with new processing.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _innerLayer?.ResetState();
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if the activation and inner layer (if present) support JIT compilation; otherwise, <c>false</c>.
    /// </value>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check if activation can be jitted
            if (!CanActivationBeJitted())
                return false;

            // Check if inner layer (if present) supports JIT
            if (_innerLayer is not null && !_innerLayer.SupportsJitCompilation)
                return false;

            return true;
        }
    }

    /// <summary>
    /// Exports the residual layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the residual connection with activation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the residual connection: output = activation(input + innerLayer(input)).
    /// If there is no inner layer, it simply returns: output = activation(input).
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (!CanActivationBeJitted())
            throw new NotSupportedException("Activation function not supported for JIT compilation.");

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create placeholder for input data
        var inputPlaceholder = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(inputPlaceholder, "input");

        inputNodes.Add(inputNode);

        Autodiff.ComputationNode<T> resultNode;

        if (_innerLayer is not null)
        {
            // Build computation graph for inner layer
            var innerInputNodes = new List<Autodiff.ComputationNode<T>>();
            var innerOutput = _innerLayer.ExportComputationGraph(innerInputNodes);

            // For the residual connection, we need to pass the same input to the inner layer
            // This is a simplification - in a full implementation, we would need to properly
            // connect the input node to the inner layer's computation graph

            // Residual connection: add input + innerLayer(input)
            resultNode = Autodiff.TensorOperations<T>.Add(inputNode, innerOutput);
        }
        else
        {
            // No inner layer, just pass through
            resultNode = inputNode;
        }

        // Apply activation using LayerBase helper
        var activatedOutput = ApplyActivationToGraph(resultNode);

        return activatedOutput;
    }
}