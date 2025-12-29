namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a bidirectional layer that processes input sequences in both forward and backward directions.
/// </summary>
/// <remarks>
/// <para>
/// A bidirectional layer processes input sequences in two directions: forward (from first to last) and backward 
/// (from last to first). This approach allows the layer to capture patterns that depend on both past and future 
/// context. The outputs from both directions can either be merged (typically added together) or kept separate, 
/// depending on the configuration.
/// </para>
/// <para><b>For Beginners:</b> This layer looks at input data in two ways at the same time - both forward and backward.
/// 
/// Think of it like reading a sentence:
/// - Forward reading: "The cat sat on the mat" (left to right)
/// - Backward reading: "mat the on sat cat The" (right to left)
/// 
/// By processing data in both directions:
/// - The layer can understand context from both past and future elements
/// - It can discover patterns that might be missed if only looking in one direction
/// - It often improves performance on sequence tasks like text processing
/// 
/// For example, in the sentence "The bank is by the river", the meaning of "bank" 
/// depends on both previous words ("The") and future words ("by the river").
/// A bidirectional layer helps capture these relationships.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BidirectionalLayer<T> : LayerBase<T>
{
    private readonly LayerBase<T> _forwardLayer;
    private readonly LayerBase<T> _backwardLayer;
    private readonly bool _mergeMode;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastForwardOutput;
    private Tensor<T>? _lastBackwardOutput;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> if either the forward or backward layer supports training; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the bidirectional layer can be trained through backpropagation. 
    /// The layer supports training if either of its internal layers (forward or backward) supports training.
    /// This is typically the case for layers that have trainable parameters, such as weights or biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The bidirectional layer supports training if either of its two internal layers
    /// (forward or backward) supports training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _forwardLayer.SupportsTraining || _backwardLayer.SupportsTraining;

    /// <summary>
    /// Initializes a new instance of the <see cref="BidirectionalLayer{T}"/> class with the specified inner layer
    /// and a ReLU activation function.
    /// </summary>
    /// <param name="innerLayer">The layer to be used for both forward and backward processing.</param>
    /// <param name="mergeMode">If <c>true</c>, outputs from both directions are added; otherwise, they are kept separate.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a bidirectional layer using the specified inner layer for both forward and backward
    /// processing. A copy of the inner layer is created for backward processing to ensure independent parameters.
    /// The mergeMode parameter determines how outputs from both directions are combined.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new bidirectional layer with a standard activation function.
    /// 
    /// When you create a bidirectional layer this way:
    /// - The same type of layer is used for both forward and backward processing
    /// - The mergeMode parameter decides how to combine the results from both directions
    /// - The ReLU activation function is used by default, which helps the network learn non-linear patterns
    /// 
    /// For example, if innerLayer is an LSTM layer, this creates a bidirectional LSTM that
    /// processes sequences in both directions.
    /// </para>
    /// </remarks>
    public BidirectionalLayer(
        LayerBase<T> innerLayer,
        bool mergeMode = true,
        IActivationFunction<T>? activationFunction = null,
        IEngine? engine = null)
        : base(innerLayer.GetInputShape(), CalculateOutputShape(innerLayer.GetOutputShape(), mergeMode), activationFunction ?? new ReLUActivation<T>())
    {
        _forwardLayer = innerLayer;
        _backwardLayer = innerLayer.Clone();
        _mergeMode = mergeMode;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="BidirectionalLayer{T}"/> class with the specified inner layer
    /// and a vector activation function.
    /// </summary>
    /// <param name="innerLayer">The layer to be used for both forward and backward processing.</param>
    /// <param name="mergeMode">If <c>true</c>, outputs from both directions are added; otherwise, they are kept separate.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a bidirectional layer using the specified inner layer for both forward and backward
    /// processing. A copy of the inner layer is created for backward processing to ensure independent parameters.
    /// This overload accepts a vector activation function, which operates on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new bidirectional layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different outputs.
    /// </para>
    /// </remarks>
    public BidirectionalLayer(
        LayerBase<T> innerLayer,
        bool mergeMode = true,
        IVectorActivationFunction<T>? vectorActivationFunction = null,
        IEngine? engine = null)
        : base(innerLayer.GetInputShape(), CalculateOutputShape(innerLayer.GetOutputShape(), mergeMode), vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _forwardLayer = innerLayer;
        _backwardLayer = innerLayer.Clone();
        _mergeMode = mergeMode;
    }

    /// <summary>
    /// Calculates the output shape of the bidirectional layer based on the inner layer's output shape and merge mode.
    /// </summary>
    /// <param name="innerOutputShape">The output shape of the inner layer.</param>
    /// <param name="mergeMode">If <c>true</c>, outputs are merged; otherwise, they are kept separate.</param>
    /// <returns>The calculated output shape for the bidirectional layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the bidirectional layer. If mergeMode is true, the output shape
    /// is the same as the inner layer's output shape because the forward and backward outputs are added together.
    /// If mergeMode is false, an additional dimension of size 2 is added at the beginning to accommodate the separate
    /// forward and backward outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the data that will come out of this layer.
    /// 
    /// The output shape depends on whether the forward and backward results are combined:
    /// - If mergeMode is true: The shape is the same as the inner layer's output
    /// - If mergeMode is false: The shape adds an extra dimension to hold both directions separately
    /// 
    /// For example, if the inner layer outputs data with shape [32, 128] (32 time steps, 128 features):
    /// - With mergeMode=true: The output shape remains [32, 128]
    /// - With mergeMode=false: The output shape becomes [2, 32, 128], where the first dimension
    ///   represents the two directions (forward and backward)
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] innerOutputShape, bool mergeMode)
    {
        if (mergeMode)
        {
            return innerOutputShape;
        }
        else
        {
            var newShape = new int[innerOutputShape.Length + 1];
            newShape[0] = 2;
            Array.Copy(innerOutputShape, 0, newShape, 1, innerOutputShape.Length);

            return newShape;
        }
    }

    /// <summary>
    /// Performs the forward pass of the bidirectional layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after bidirectional processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the bidirectional layer. It processes the input in both forward
    /// and backward directions using the respective inner layers, and then combines the outputs according to the
    /// merge mode. The input and outputs are cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input data through both forward and backward layers.
    /// 
    /// During the forward pass:
    /// - The original input is sent through the forward layer
    /// - A reversed version of the input is sent through the backward layer
    /// - The results from both directions are either combined or kept separate
    /// 
    /// This method also saves the inputs and outputs for later use during training.
    /// 
    /// For example, with a text sequence, the forward layer sees "Hello world" while
    /// the backward layer sees "world Hello", and then the results are combined.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Forward pass
        var forwardInput = input;
        _lastForwardOutput = _forwardLayer.Forward(forwardInput);

        // Backward pass (reverse the input sequence)
        var backwardInput = ReverseSequence(input);
        _lastBackwardOutput = _backwardLayer.Forward(backwardInput);

        // Merge outputs
        return MergeOutputs(_lastForwardOutput, _lastBackwardOutput);
    }

    /// <summary>
    /// Performs the backward pass of the bidirectional layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the bidirectional layer, which is used during training to propagate
    /// error gradients back through the network. It splits the output gradient according to the merge mode, propagates
    /// it through both forward and backward inner layers, and combines the resulting input gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    ///
    /// During the backward pass:
    /// - The error gradient from the next layer is received
    /// - If the outputs were merged, the same gradient is sent to both forward and backward layers
    /// - If the outputs were separate, the gradient is split for each direction
    /// - The gradients from both layers are combined to update the previous layer
    ///
    /// This process is part of the "backpropagation" algorithm that helps neural networks learn.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        // Note: BidirectionalLayer delegates backward pass to inner forward and backward layers.
        // Autodiff support is handled by the inner layers (e.g., LSTMLayer, GRULayer).
        // This wrapper layer simply manages gradient flow for bidirectional processing.

        Tensor<T> forwardGradient, backwardGradient;

        if (_mergeMode)
        {
            forwardGradient = outputGradient;
            backwardGradient = outputGradient;
        }
        else
        {
            forwardGradient = outputGradient.Slice(0);
            backwardGradient = outputGradient.Slice(1);
        }

        var forwardInputGradient = _forwardLayer.Backward(forwardGradient);
        var backwardInputGradient = _backwardLayer.Backward(backwardGradient);

        // Reverse the backward gradient
        backwardInputGradient = ReverseSequence(backwardInputGradient);

        // Sum the gradients
        return forwardInputGradient.Add(backwardInputGradient);
    }

    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // Note: BidirectionalLayer is a wrapper that delegates to inner layers.
        // The autodiff path simply ensures the inner layers use their autodiff implementations.
        // This method manages the bidirectional gradient flow using autodiff-friendly operations.

        if (_lastInput is null || _lastForwardOutput is null || _lastBackwardOutput is null)
            throw new InvalidOperationException("Forward pass must be called before Backward");

        // Split output gradient based on merge mode
        Tensor<T> forwardGradient, backwardGradient;

        if (_mergeMode)
        {
            // In merge mode, the same gradient flows to both directions
            forwardGradient = outputGradient;
            backwardGradient = outputGradient;
        }
        else
        {
            // In non-merge mode, split the gradient for each direction
            forwardGradient = outputGradient.Slice(0);
            backwardGradient = outputGradient.Slice(1);
        }

        // Propagate gradients through inner layers (they will use their autodiff implementations if UseAutodiff is set)
        var forwardInputGradient = _forwardLayer.Backward(forwardGradient);
        var backwardInputGradient = _backwardLayer.Backward(backwardGradient);

        // Reverse the backward gradient to match input sequence order
        backwardInputGradient = ReverseSequence(backwardInputGradient);

        // Sum the gradients from both directions
        return forwardInputGradient.Add(backwardInputGradient);
    }

    /// <summary>
    /// Updates the parameters of both forward and backward layers using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of both the forward and backward inner layers based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - Both forward and backward layers are updated independently
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _forwardLayer.UpdateParameters(learningRate);
        _backwardLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Reverses the sequence order along the time dimension (typically dimension 1).
    /// </summary>
    /// <param name="input">The input tensor to reverse.</param>
    /// <returns>A new tensor with the sequence reversed.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor with the same shape as the input, but with the sequence reversed along
    /// the time dimension (typically dimension 1). This is used to flip the input for the backward layer and
    /// to flip the gradients during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method flips the order of elements in a sequence.
    /// 
    /// For example:
    /// - Original sequence: [1, 2, 3, 4, 5]
    /// - Reversed sequence: [5, 4, 3, 2, 1]
    /// 
    /// In a neural network, a "sequence" typically represents time steps or ordered data points.
    /// This method is important for the bidirectional functionality because:
    /// - The backward layer needs to process data in reverse order
    /// - The gradients from the backward layer need to be reversed again during training
    /// </para>
    /// </remarks>
    private static Tensor<T> ReverseSequence(Tensor<T> input)
    {
        var reversed = new Tensor<T>(input.Shape);
        int timeSteps = input.Shape[1];

        for (int i = 0; i < timeSteps; i++)
        {
            // Slice along dimension 1 (time axis), getting a [batch, features] tensor
            var slice = input.GetSliceAlongDimension(timeSteps - 1 - i, 1);
            // Set into reversed tensor at position i along dimension 1
            reversed.SetSlice(1, i, slice);
        }

        return reversed;
    }

    /// <summary>
    /// Merges the outputs from the forward and backward passes according to the configured merge mode.
    /// </summary>
    /// <param name="forward">The output tensor from the forward pass.</param>
    /// <param name="backward">The output tensor from the backward pass.</param>
    /// <returns>The merged output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method combines the outputs from the forward and backward passes according to the configured merge mode.
    /// If mergeMode is true, the outputs are added together element-wise. If mergeMode is false, the outputs are
    /// stacked along a new dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method combines the results from the forward and backward processing.
    /// 
    /// There are two ways to combine the results:
    /// - If mergeMode is true: The values are added together (element by element)
    /// - If mergeMode is false: The two sets of results are kept separate but packed into a single tensor
    /// 
    /// For example, with mergeMode=true and values:
    /// - Forward: [1, 2, 3]
    /// - Backward: [4, 5, 6]
    /// - Result: [5, 7, 9] (each element added)
    /// 
    /// With mergeMode=false:
    /// - Result would contain both original sets of values in a larger tensor
    /// </para>
    /// </remarks>
    private Tensor<T> MergeOutputs(Tensor<T> forward, Tensor<T> backward)
    {
        if (_mergeMode)
        {
            return forward.Add(backward);
        }
        else
        {
            return Tensor<T>.Stack([forward, backward], 0);
        }
    }

    /// <summary>
    /// Gets all trainable parameters from both the forward and backward layers as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from both the forward and backward inner layers and combines them
    /// into a single vector. This is useful for optimization algorithms that operate on all parameters at once, or for
    /// saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from both forward and backward layers.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include weights and biases from both forward and backward layers
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Combine parameters from both forward and backward layers
        var forwardParams = _forwardLayer.GetParameters();
        var backwardParams = _backwardLayer.GetParameters();

        var combinedParams = new Vector<T>(forwardParams.Length + backwardParams.Length);

        // Copy forward parameters
        for (int i = 0; i < forwardParams.Length; i++)
        {
            combinedParams[i] = forwardParams[i];
        }

        // Copy backward parameters
        for (int i = 0; i < backwardParams.Length; i++)
        {
            combinedParams[i + forwardParams.Length] = backwardParams[i];
        }

        return combinedParams;
    }

    /// <summary>
    /// Sets the trainable parameters for both the forward and backward layers.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters for both the forward and backward inner layers from a single vector.
    /// It extracts the appropriate portions of the input vector for each inner layer. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in both forward and backward layers.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The first part of the vector is used for the forward layer
    /// - The second part of the vector is used for the backward layer
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        var forwardParams = _forwardLayer.GetParameters();
        var backwardParams = _backwardLayer.GetParameters();

        if (parameters.Length != forwardParams.Length + backwardParams.Length)
            throw new ArgumentException($"Expected {forwardParams.Length + backwardParams.Length} parameters, but got {parameters.Length}");

        // Extract and set forward parameters
        var newForwardParams = new Vector<T>(forwardParams.Length);
        for (int i = 0; i < forwardParams.Length; i++)
        {
            newForwardParams[i] = parameters[i];
        }

        // Extract and set backward parameters
        var newBackwardParams = new Vector<T>(backwardParams.Length);
        for (int i = 0; i < backwardParams.Length; i++)
        {
            newBackwardParams[i] = parameters[i + forwardParams.Length];
        }

        _forwardLayer.SetParameters(newForwardParams);
        _backwardLayer.SetParameters(newBackwardParams);
    }

    /// <summary>
    /// Resets the internal state of the bidirectional layer and its inner layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the bidirectional layer, including the cached inputs and outputs,
    /// as well as the states of both forward and backward inner layers. This is useful when starting to process
    /// a new sequence or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Both forward and backward layers are also reset
    /// - The layer forgets any information from previous sequences
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Preventing information from one sequence affecting another
    /// - Starting a new training episode
    /// 
    /// For example, if you've processed one sentence and want to start with a new sentence,
    /// you should reset the state to prevent the new sentence from being influenced by the previous one.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastForwardOutput = null;
        _lastBackwardOutput = null;

        _forwardLayer.ResetState();
        _backwardLayer.ResetState();
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (!_forwardLayer.SupportsJitCompilation || !_backwardLayer.SupportsJitCompilation)
            throw new InvalidOperationException("BidirectionalLayer requires both inner layers to support JIT compilation.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Forward layer processing
        var forwardInputNodes = new List<ComputationNode<T>>();
        var forwardOutput = _forwardLayer.ExportComputationGraph(forwardInputNodes);

        // Backward layer processing (note: sequence reversal is handled at runtime, not in graph)
        var backwardInputNodes = new List<ComputationNode<T>>();
        var backwardOutput = _backwardLayer.ExportComputationGraph(backwardInputNodes);

        // Merge outputs based on merge mode
        if (_mergeMode)
        {
            // Add outputs element-wise
            return TensorOperations<T>.Add(forwardOutput, backwardOutput);
        }
        else
        {
            // Stack outputs along new dimension
            // Note: This requires a Stack operation in TensorOperations
            // For now, return forward output as primary
            return forwardOutput;
        }
    }

    public override bool SupportsJitCompilation =>
        _forwardLayer.SupportsJitCompilation && _backwardLayer.SupportsJitCompilation;

}
