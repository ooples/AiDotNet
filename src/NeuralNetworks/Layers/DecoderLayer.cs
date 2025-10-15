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
    private readonly AttentionLayer<T> _selfAttention = default!;

    /// <summary>
    /// The cross-attention mechanism that attends to the encoder output.
    /// </summary>
    private readonly AttentionLayer<T> _crossAttention = default!;

    /// <summary>
    /// The feed-forward neural network component of the decoder layer.
    /// </summary>
    private readonly FeedForwardLayer<T> _feedForward = default!;

    /// <summary>
    /// Layer normalization applied after self-attention.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm1 = default!;

    /// <summary>
    /// Layer normalization applied after cross-attention.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm2 = default!;

    /// <summary>
    /// Layer normalization applied after the feed-forward network.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm3 = default!;

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
    public override int InputSize { get; }

    /// <summary>
    /// Stores the gradient with respect to the input from the last backward pass.
    /// </summary>
    private Tensor<T>? _lastInputGradient;

    /// <summary>
    /// Stores the gradient with respect to the encoder output from the last backward pass.
    /// </summary>
    private Tensor<T>? _lastEncoderOutputGradient;

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
    public DecoderLayer(int inputSize, int attentionSize, int feedForwardSize, IActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new ReLUActivation<T>())
    {
        _selfAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        _crossAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        _feedForward = new FeedForwardLayer<T>(inputSize, feedForwardSize, activation);
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
    public DecoderLayer(int inputSize, int attentionSize, int feedForwardSize, IVectorActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new ReLUActivation<T>())
    {
        _selfAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        _crossAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);
        _feedForward = new FeedForwardLayer<T>(inputSize, feedForwardSize, activation);
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

        if (decoderInput.Shape.Length != 2 || decoderInput.Shape[1] != InputSize)
            throw new ArgumentException("Decoder input tensor must have shape [batch_size, input_size].");

        if (encoderOutput.Shape.Length != 2 || encoderOutput.Shape[1] != InputSize)
            throw new ArgumentException("Encoder output tensor must have shape [batch_size, input_size].");

        if (attentionMask != null && (attentionMask.Shape.Length != 2 || attentionMask.Shape[0] != decoderInput.Shape[0]))
            throw new ArgumentException("Attention mask tensor must have shape [batch_size, sequence_length] and match the batch size of the input.");

        return ForwardInternal(decoderInput, encoderOutput, attentionMask);
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
        Tensor<T> mask = attentionMask ?? Tensor<T>.CreateDefault(input.Shape, NumOps.One); // Default to all ones if no mask is provided

        // Self-attention (now with optional mask)
        var selfAttentionOutput = _selfAttention.Forward(input, mask);
        var normalized1 = _norm1.Forward(input.Add(selfAttentionOutput));

        // Cross-attention (now with optional mask)
        var crossAttentionOutput = _crossAttention.Forward(normalized1, encoderOutput, mask);
        var normalized2 = _norm2.Forward(normalized1.Add(crossAttentionOutput));

        // Feed-forward (unchanged)
        var feedForwardOutput = _feedForward.Forward(normalized2);
        var output = _norm3.Forward(normalized2.Add(feedForwardOutput));

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
        if (_lastInput == null || _lastEncoderOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

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
        var dNormalized2 = _feedForward.Backward(outputGradient);
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
        _feedForward.UpdateParameters(learningRate);
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
        var parameters = new List<T>();
        parameters.AddRange(_selfAttention.GetParameters());
        parameters.AddRange(_crossAttention.GetParameters());
        parameters.AddRange(_feedForward.GetParameters());
        parameters.AddRange(_norm1.GetParameters());
        parameters.AddRange(_norm2.GetParameters());
        parameters.AddRange(_norm3.GetParameters());
        return new Vector<T>(parameters.ToArray());
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
        index = UpdateComponentParameters(_feedForward, parameters, index);
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
        var componentParams = new Vector<T>(paramCount);
        Array.Copy(parameters.ToArray(), startIndex, componentParams.ToArray(), 0, paramCount);
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
        _feedForward.ResetState();
        _norm1.ResetState();
        _norm2.ResetState();
        _norm3.ResetState();
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new InvalidOperationException(
            "DecoderLayer requires both decoder input and encoder output. " +
            "Please use the Forward(decoderInput, encoderOutput, attentionMask) overload instead.");
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
        _feedForward.ParameterCount +
        _norm1.ParameterCount +
        _norm2.ParameterCount +
        _norm3.ParameterCount;
}