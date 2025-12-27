namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Decoder Layer in a Transformer architecture.
/// </summary>
/// <remarks>
/// <para>
/// The Decoder Layer is a key component in sequence-to-sequence models, particularly in Transformer architectures.
/// It processes the target sequence and incorporates information from the encoder output.
/// </para>
/// <para><b>For Beginners:</b> The Decoder Layer helps in generating output sequences (like translations) 
/// by considering both what it has generated so far and the information from the input sequence.
/// 
/// It's like writing a story where you:
/// 1. Look at what you've written so far (self-attention)
/// 2. Refer back to your source material (cross-attention to encoder output)
/// 3. Think about how to continue the story (feed-forward network)
/// 
/// This process helps in creating coherent and context-aware outputs.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class DecoderLayer<T> : LayerBase<T>
{

    /// <summary>
    /// The self-attention mechanism of the decoder layer.
    /// </summary>
    private readonly AttentionLayer<T> _selfAttention;

    /// <summary>
    /// The cross-attention mechanism that attends to the encoder output.
    /// </summary>
    private readonly AttentionLayer<T> _crossAttention;

    /// <summary>
    /// The feed-forward neural network component of the decoder layer.
    /// </summary>
    private readonly FeedForwardLayer<T> _feedForward1;
    private readonly FeedForwardLayer<T> _feedForward2;

    /// <summary>
    /// Layer normalization applied after self-attention.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm1;

    /// <summary>
    /// Layer normalization applied after cross-attention.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm2;

    /// <summary>
    /// Layer normalization applied after the feed-forward network.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm3;

    /// <summary>
    /// Stores the last input tensor processed by the layer.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the last encoder output tensor used by the layer.
    /// </summary>
    private Tensor<T>? _lastEncoderOutput;

    /// <summary>
    /// Gets the size of the input features for this layer.
    /// </summary>
    public int InputSize { get; private set; }

    /// <summary>
    /// Stores the gradient with respect to the input from the last backward pass.
    /// </summary>
    private Tensor<T>? _lastInputGradient;

    /// <summary>
    /// Stores the gradient with respect to the encoder output from the last backward pass.
    /// </summary>
    private Tensor<T>? _lastEncoderOutputGradient;

    /// <summary>
    /// Tracks whether the last input was originally 2D (and thus reshaped to 3D).
    /// </summary>
    private bool _inputWas2D = false;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the DecoderLayer class with scalar activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="feedForwardSize">The size of the feed-forward network.</param>
    /// <param name="activation">The scalar activation function to use. If null, ReLUActivation is used.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    public DecoderLayer(int inputSize, int attentionSize, int feedForwardSize, IActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new ReLUActivation<T>())
    {
        _selfAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        _crossAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        
        // Standard transformer FFN: Linear(input -> ff) + activation + Linear(ff -> input)
        _feedForward1 = new FeedForwardLayer<T>(inputSize, feedForwardSize, activation);
        _feedForward2 = new FeedForwardLayer<T>(feedForwardSize, inputSize, (IActivationFunction<T>?)null);
        
        InputSize = inputSize;

        _norm1 = new LayerNormalizationLayer<T>(inputSize);
        _norm2 = new LayerNormalizationLayer<T>(inputSize);
        _norm3 = new LayerNormalizationLayer<T>(inputSize);
    }

    /// <summary>
    /// Initializes a new instance of the DecoderLayer class with vector activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="feedForwardSize">The size of the feed-forward network.</param>
    /// <param name="activation">The vector activation function to use. If null, ReLUActivation is used.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    public DecoderLayer(int inputSize, int attentionSize, int feedForwardSize, IVectorActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new ReLUActivation<T>())
    {
        _selfAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        _crossAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        
        // Standard transformer FFN: Linear(input -> ff) + activation + Linear(ff -> input)
        _feedForward1 = new FeedForwardLayer<T>(inputSize, feedForwardSize, activation);
        _feedForward2 = new FeedForwardLayer<T>(feedForwardSize, inputSize, (IVectorActivationFunction<T>?)null);
        
        InputSize = inputSize;

        _norm1 = new LayerNormalizationLayer<T>(inputSize);
        _norm2 = new LayerNormalizationLayer<T>(inputSize);
        _norm3 = new LayerNormalizationLayer<T>(inputSize);
    }

    /// <summary>
    /// Performs the forward pass of the decoder layer.
    /// </summary>
    /// <param name="inputs">An array of input tensors. The first tensor is the decoder input, the second is the encoder output, and the third (optional) is the attention mask.</param>
    /// <returns>The output tensor after processing through the decoder layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of input tensors or their shapes are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method processes the inputs through the decoder layer. It expects
    /// two or three input tensors: the decoder's own input, the encoder's output, and optionally an attention mask.
    /// The method combines these inputs, processes them through the layer, and returns the final output.
    /// The attention mask, if provided, helps control which parts of the input sequence the layer should focus on.</para>
    /// </remarks>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2 || inputs.Length > 3)
            throw new ArgumentException("DecoderLayer requires two or three input tensors: decoder input, encoder output, and optionally an attention mask.");

        var decoderInput = inputs[0];
        var encoderOutput = inputs[1];
        Tensor<T>? attentionMask = inputs.Length == 3 ? inputs[2] : null;

        // Handle both 2D [batch, features] and 3D [batch, seq, features] input
        _inputWas2D = decoderInput.Shape.Length == 2;
        Tensor<T> input3D, encoderOutput3D;

        if (_inputWas2D)
        {
            // 2D input: [batch, features] -> [batch, 1, features]
            if (decoderInput.Shape[1] != InputSize)
                throw new ArgumentException($"Decoder input dimension {decoderInput.Shape[1]} does not match expected InputSize {InputSize}.");
            if (encoderOutput.Shape[1] != InputSize)
                throw new ArgumentException($"Encoder output dimension {encoderOutput.Shape[1]} does not match expected InputSize {InputSize}.");

            int batchSize = decoderInput.Shape[0];
            input3D = decoderInput.Reshape(batchSize, 1, InputSize);
            encoderOutput3D = encoderOutput.Reshape(encoderOutput.Shape[0], 1, InputSize);
        }
        else if (decoderInput.Shape.Length == 3)
        {
            // 3D input: validate shape
            if (decoderInput.Shape[2] != InputSize)
                throw new ArgumentException($"Decoder input dimension {decoderInput.Shape[2]} does not match expected InputSize {InputSize}.");
            if (encoderOutput.Shape.Length != 3 || encoderOutput.Shape[2] != InputSize)
                throw new ArgumentException($"Encoder output must be 3D with last dimension matching InputSize {InputSize}.");

            input3D = decoderInput;
            encoderOutput3D = encoderOutput;
        }
        else
        {
            throw new ArgumentException($"Decoder input must be 2D or 3D. Got tensor with rank {decoderInput.Shape.Length}.");
        }

        return ForwardInternal(input3D, encoderOutput3D, attentionMask);
    }

    /// <summary>
    /// Performs the internal forward pass of the decoder layer.
    /// </summary>
    /// <param name="input">The decoder input tensor.</param>
    /// <param name="encoderOutput">The encoder output tensor.</param>
    /// <param name="attentionMask">Optional attention mask tensor.</param>
    /// <returns>The output tensor after processing through the decoder layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method represents the core processing of the decoder layer.
    /// It applies self-attention, cross-attention with the encoder output, and a feed-forward network,
    /// with layer normalization applied between each step. The attention mask, if provided, helps
    /// control which parts of the input sequence the layer should focus on.</para>
    /// </remarks>
    private Tensor<T> ForwardInternal(Tensor<T> input, Tensor<T> encoderOutput, Tensor<T>? attentionMask = null)
    {
        _lastInput = input;
        _lastEncoderOutput = encoderOutput;
        
        // For 3D input, create appropriate mask shape
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        Tensor<T> mask = attentionMask ?? Tensor<T>.CreateDefault([batchSize, seqLen, seqLen], NumOps.Zero);

        // Self-attention (now with optional mask)
        var selfAttentionOutput = _selfAttention.Forward(input, mask);
        var normalized1 = _norm1.Forward(input.Add(selfAttentionOutput));

        // Cross-attention (now with optional mask)
        var crossAttentionOutput = _crossAttention.Forward(normalized1, encoderOutput, mask);
        var normalized2 = _norm2.Forward(normalized1.Add(crossAttentionOutput));

        // Standard transformer FFN: Linear(input -> ff) + activation + Linear(ff -> input)
        var ffExpanded = _feedForward1.Forward(normalized2);
        var feedForwardOutput = _feedForward2.Forward(ffExpanded);
        var output = _norm3.Forward(normalized2.Add(feedForwardOutput));

        // If input was originally 2D, reshape output back to 2D
        if (_inputWas2D)
        {
            output = output.Reshape(batchSize, output.Shape[2]);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the decoder layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>A concatenated tensor containing gradients for both the input and the encoder output.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates how much each part of the input and the encoder output
    /// contributed to the error. It's used during training to update the layer's parameters. The method expects
    /// the forward pass to have been called first, as it uses information stored during the forward pass.</para>
    /// <para>The returned tensor is a concatenation of two gradients: the gradient with respect to the input
    /// and the gradient with respect to the encoder output. Use the <see cref="LastBackwardGradients"/> property
    /// to access these gradients separately.</para>
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
        if (_lastInput == null || _lastEncoderOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var (inputGradient, encoderOutputGradient) = BackwardInternal(outputGradient);
        _lastInputGradient = inputGradient;
        _lastEncoderOutputGradient = encoderOutputGradient;

        // Concatenate the input gradient and encoder output gradient
        return Tensor<T>.Concatenate(new[] { inputGradient, encoderOutputGradient }, 1);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation by delegating to the autodiff implementations
    /// of the constituent layers (AttentionLayer, LayerNormalizationLayer, FeedForwardLayer).
    /// Each sublayer will use its own autodiff implementation if available.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastEncoderOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Composite layer: just call Backward on each sublayer with UseAutodiff enabled
        // The sublayers will handle their own autodiff if they support it
        var (inputGradient, encoderOutputGradient) = BackwardInternal(outputGradient);
        _lastInputGradient = inputGradient;
        _lastEncoderOutputGradient = encoderOutputGradient;

        // Concatenate the input gradient and encoder output gradient
        return Tensor<T>.Concatenate(new[] { inputGradient, encoderOutputGradient }, 1);
    }

    /// <summary>
    /// Gets the most recent gradients calculated during the backward pass.
    /// </summary>
    /// <returns>A tuple containing the gradient with respect to the input and the gradient with respect to the encoder output.</returns>
    /// <exception cref="InvalidOperationException">Thrown when accessed before a backward pass has been performed.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This property provides easy access to the separate gradients calculated
    /// during the last backward pass. It's useful when you need to handle the input gradient and encoder output
    /// gradient separately, rather than dealing with the concatenated gradient returned by the Backward method.</para>
    /// </remarks>
    public (Tensor<T> InputGradient, Tensor<T> EncoderOutputGradient) LastBackwardGradients
    {
        get
        {
            if (_lastInputGradient == null || _lastEncoderOutputGradient == null)
                throw new InvalidOperationException("Backward pass must be called before accessing gradients.");
            return (_lastInputGradient, _lastEncoderOutputGradient);
        }
    }

    /// <summary>
    /// Performs the internal backward pass of the decoder layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>A tuple containing the gradient with respect to the input and the encoder output.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates the gradients for each component of the decoder layer
    /// (self-attention, cross-attention, feed-forward network, and layer normalizations) in reverse order
    /// of the forward pass. It's a crucial part of the backpropagation process in neural networks.</para>
    /// </remarks>
    private (Tensor<T> inputGradient, Tensor<T> encoderOutputGradient) BackwardInternal(Tensor<T> outputGradient)
    {
        // Backward through FFN (reverse order: projection then expansion)
        var dFF2 = _feedForward2.Backward(outputGradient);
        var dNormalized2 = _feedForward1.Backward(dFF2);
        dNormalized2 = dNormalized2.Add(outputGradient);
        var dNorm2 = _norm2.Backward(dNormalized2);

        var dCrossAttention = _crossAttention.Backward(dNorm2);
        var dNormalized1 = dCrossAttention.Add(dNorm2);
        var dNorm1 = _norm1.Backward(dNormalized1);

        var dSelfAttention = _selfAttention.Backward(dNorm1);
        var dInput = dSelfAttention.Add(dNorm1);

        // Calculate gradient with respect to encoder output
        var dEncoderOutput = _crossAttention.Backward(dNorm2);

        return (dInput, dEncoderOutput);
    }

    /// <summary>
    /// Updates the layer's parameters based on the computed gradients and a learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the calculated gradients to the layer's parameters,
    /// effectively "learning" from the training data. The learning rate determines how big of a step
    /// we take in the direction suggested by the gradients.</para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _selfAttention.UpdateParameters(learningRate);
        _crossAttention.UpdateParameters(learningRate);
        _feedForward1.UpdateParameters(learningRate);
        _feedForward2.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _norm3.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Retrieves the current parameters of the layer.
    /// </summary>
    /// <returns>A vector containing all the parameters of the layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method collects all the parameters from the various components
    /// of the decoder layer (self-attention, cross-attention, feed-forward network, and layer normalizations)
    /// and combines them into a single vector. This is useful for operations that need to work with all
    /// parameters at once, such as optimization algorithms.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for efficient parameter collection
        var selfAttnParams = _selfAttention.GetParameters();
        var crossAttnParams = _crossAttention.GetParameters();
        var ff1Params = _feedForward1.GetParameters();
        var ff2Params = _feedForward2.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var norm2Params = _norm2.GetParameters();
        var norm3Params = _norm3.GetParameters();

        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(
                    Vector<T>.Concatenate(selfAttnParams, crossAttnParams),
                    Vector<T>.Concatenate(ff1Params, ff2Params)),
                norm1Params),
            Vector<T>.Concatenate(norm2Params, norm3Params));
    }

    /// <summary>
    /// Updates the layer's parameters with the provided values.
    /// </summary>
    /// <param name="parameters">A vector containing new parameter values.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a vector of new parameter values and distributes
    /// them to the various components of the decoder layer. It's the opposite of GetParameters() and is
    /// typically used after an optimization step to update the layer with improved parameter values.</para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        index = UpdateComponentParameters(_selfAttention, parameters, index);
        index = UpdateComponentParameters(_crossAttention, parameters, index);
        index = UpdateComponentParameters(_feedForward1, parameters, index);
        index = UpdateComponentParameters(_feedForward2, parameters, index);
        index = UpdateComponentParameters(_norm1, parameters, index);
        index = UpdateComponentParameters(_norm2, parameters, index);
        UpdateComponentParameters(_norm3, parameters, index);
    }

    /// <summary>
    /// Updates the parameters of a specific component within the decoder layer.
    /// </summary>
    /// <param name="component">The component to update.</param>
    /// <param name="parameters">The full vector of parameters for the entire layer.</param>
    /// <param name="startIndex">The starting index in the parameters vector for this component.</param>
    /// <returns>The next starting index after updating this component's parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helper method is used by UpdateParameters to update each
    /// individual component of the decoder layer. It extracts the relevant portion of the parameter
    /// vector for a specific component and applies those parameters to that component.</para>
    /// </remarks>
    private int UpdateComponentParameters(LayerBase<T> component, Vector<T> parameters, int startIndex)
    {
        int paramCount = component.ParameterCount;

        // Use Engine.TensorSlice to extract component parameters without manual loops
        var paramsTensor = new Tensor<T>([parameters.Length], parameters);
        var componentParamsTensor = Engine.TensorSlice(paramsTensor, [startIndex], [paramCount]);
        var componentParams = new Vector<T>(componentParamsTensor.ToArray());

        component.UpdateParameters(componentParams);

        return startIndex + paramCount;
    }

    /// <summary>
    /// Resets the state of the decoder layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method clears any stored state in the decoder layer and its components.
    /// It's typically called between processing different sequences or at the start of a new epoch in training.
    /// Resetting the state ensures that information from previous inputs doesn't affect the processing of new, unrelated inputs.</para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastEncoderOutput = null;
        _selfAttention.ResetState();
        _crossAttention.ResetState();
        _feedForward1.ResetState();
        _feedForward2.ResetState();
        _norm1.ResetState();
        _norm2.ResetState();
        _norm3.ResetState();
    }

    /// <summary>
    /// Single-input forward pass is not supported for DecoderLayer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <exception cref="NotSupportedException">Always thrown because DecoderLayer requires multiple inputs.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> DecoderLayer cannot operate with a single input because it needs both
    /// the decoder input and the encoder output to function properly. Use the overload that accepts
    /// multiple tensors: <see cref="Forward(Tensor{T}[])" /> instead.</para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException(
            "DecoderLayer requires multiple inputs (decoder input and encoder output). " +
            "Use Forward(params Tensor<T>[] inputs) instead, providing at least two tensors: " +
            "the decoder input and the encoder output, and optionally an attention mask.");
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This property calculates and returns the total number of parameters
    /// in the decoder layer by summing the parameter counts of all its components. This is useful for
    /// understanding the complexity of the layer and for certain optimization techniques.</para>
    /// </remarks>
    public override int ParameterCount =>
        _selfAttention.ParameterCount +
        _crossAttention.ParameterCount +
        _feedForward1.ParameterCount + _feedForward2.ParameterCount +
        _norm1.ParameterCount +
        _norm2.ParameterCount +
        _norm3.ParameterCount;

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // DecoderLayer requires TWO inputs: decoder input and encoder output
        if (inputNodes.Count < 2)
            throw new ArgumentException(
                "DecoderLayer requires at least two input nodes: decoder input and encoder output.",
                nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var decoderInput = inputNodes[0];
        var encoderOutput = inputNodes[1];

        // Self-attention on decoder input
        var selfAttentionOutput = _selfAttention.ExportComputationGraph([decoderInput]);
        var residual1 = TensorOperations<T>.Add(decoderInput, selfAttentionOutput);
        var normalized1 = _norm1.ExportComputationGraph([residual1]);

        // Cross-attention with encoder output
        var crossAttentionOutput = _crossAttention.ExportComputationGraph([normalized1, encoderOutput]);
        var residual2 = TensorOperations<T>.Add(normalized1, crossAttentionOutput);
        var normalized2 = _norm2.ExportComputationGraph([residual2]);

        // Feed-forward network
        var ffExpanded = _feedForward1.ExportComputationGraph([normalized2]);
        var feedForwardOutput = _feedForward2.ExportComputationGraph([ffExpanded]);
        var residual3 = TensorOperations<T>.Add(normalized2, feedForwardOutput);
        var output = _norm3.ExportComputationGraph([residual3]);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> because DecoderLayer can be compiled with multiple input nodes representing
    /// the decoder input and encoder output. The computation graph supports multiple inputs.
    /// </value>
    public override bool SupportsJitCompilation => true;

}
