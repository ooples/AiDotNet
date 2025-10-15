namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an Attention Layer for focusing on relevant parts of input sequences.
/// </summary>
/// <remarks>
/// <para>
/// The Attention Layer is a mechanism that allows a neural network to focus on different parts of the input
/// sequence when producing each element of the output sequence. It computes a weighted sum of the input sequence,
/// where the weights (attention weights) are determined based on the relevance of each input element to the current output.
/// </para>
/// <para><b>For Beginners:</b> An Attention Layer helps the network focus on important parts of the input.
/// 
/// Think of it like reading a long document to answer a question:
/// - Instead of remembering every word, you focus on key sentences or phrases
/// - The attention mechanism does something similar for the neural network
/// - It helps the network decide which parts of the input are most relevant for the current task
/// 
/// Common applications include:
/// - Machine translation (focusing on relevant words when translating)
/// - Image captioning (focusing on relevant parts of an image when describing it)
/// - Speech recognition (focusing on important audio segments)
/// 
/// The key advantage is that it allows the network to handle long sequences more effectively
/// by focusing on the most relevant parts rather than trying to remember everything.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class AttentionLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight tensor for the query transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor transforms the input into a query representation.
    /// </para>
    /// <para><b>For Beginners:</b> This helps create a "question" from the current state.
    /// 
    /// Think of it as formulating what information we're looking for in the input sequence.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wq = default!;

    /// <summary>
    /// The weight tensor for the key transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor transforms the input into a key representation.
    /// </para>
    /// <para><b>For Beginners:</b> This helps create "labels" for each part of the input sequence.
    /// 
    /// Think of it as creating a way to identify or index different parts of the input.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wk = default!;

    /// <summary>
    /// The weight tensor for the value transformation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor transforms the input into a value representation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the actual content we'll use from each part of the input.
    /// 
    /// Think of it as extracting the useful information from each part of the input sequence.
    /// </para>
    /// </remarks>
    private Tensor<T> _Wv = default!;

    /// <summary>
    /// The size of the input features.
    /// </summary>
    private readonly int _inputSize;

    /// <summary>
    /// The size of the attention mechanism (typically smaller than the input size).
    /// </summary>
    private readonly int _attentionSize;

    /// <summary>
    /// The last input processed by the layer.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The last attention weights computed by the layer.
    /// </summary>
    private Tensor<T>? _lastAttentionWeights;

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property calculates the total number of trainable parameters in the Attention Layer,
    /// which includes all the weights for query, key, and value transformations.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many numbers the layer needs to learn.
    /// 
    /// It counts all the weights in the three transformation matrices (Wq, Wk, Wv).
    /// A higher number means the layer can potentially learn more complex patterns,
    /// but also requires more data and time to train effectively.
    /// </para>
    /// </remarks>
    public override int ParameterCount =>
        _attentionSize * _inputSize * 3; // Wq, Wk, Wv

    /// <summary>
    /// Gradient of the weight tensor for the value transformation.
    /// </summary>
    private Tensor<T>? _dWv;

    /// <summary>
    /// Gradient of the weight tensor for the key transformation.
    /// </summary>
    private Tensor<T>? _dWk;

    /// <summary>
    /// Gradient of the weight tensor for the query transformation.
    /// </summary>
    private Tensor<T>? _dWq;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates that the Attention Layer can be trained using backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you that the layer can learn and improve its performance over time.
    /// 
    /// When this is true, it means the layer can adjust its internal weights based on the errors it makes,
    /// allowing it to get better at its task as it sees more data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the AttentionLayer class with scalar activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="activation">The activation function to use. If null, SoftmaxActivation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an Attention Layer with scalar activation, allowing for element-wise application of the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Attention Layer with its initial values, using a scalar activation function.
    /// 
    /// The scalar activation means the same function is applied to each element independently.
    /// This is useful when you want to treat each attention score separately.
    /// </para>
    /// </remarks>
    public AttentionLayer(int inputSize, int attentionSize, IActivationFunction<T>? activation = null)
        : base([inputSize], [attentionSize], activation ?? new SoftmaxActivation<T>())
    {
        _inputSize = inputSize;
        _attentionSize = attentionSize;
        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _attentionSize));
        _Wq = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wk = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wv = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
    }

    /// <summary>
    /// Initializes a new instance of the AttentionLayer class with vector activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="activation">The vector activation function to use. If null, SoftmaxActivation is used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an Attention Layer with vector activation, allowing for operations on entire vectors or tensors.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Attention Layer with its initial values, using a vector activation function.
    /// 
    /// The vector activation means the function is applied to the entire set of attention scores at once.
    /// This can be more efficient and allows for more complex interactions between attention scores.
    /// </para>
    /// </remarks>
    public AttentionLayer(int inputSize, int attentionSize, IVectorActivationFunction<T>? activation = null)
        : base([inputSize], [attentionSize], activation ?? new SoftmaxActivation<T>())
    {
        _inputSize = inputSize;
        _attentionSize = attentionSize;
        T scale = NumOps.Sqrt(NumOps.FromDouble(1.0 / _attentionSize));
        _Wq = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wk = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
        _Wv = InitializeTensor(new[] { _attentionSize, _inputSize }, scale);
    }

    /// <summary>
    /// Initializes a tensor with random values scaled by a given factor.
    /// </summary>
    /// <param name="shape">The shape of the tensor to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <returns>A new tensor with randomly initialized values.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new tensor and fills it with random values from a uniform distribution,
    /// scaled by the provided factor. This helps in initializing the weights of the attention mechanism.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the starting values for the layer's internal weights.
    /// 
    /// The random initialization is important because it gives the network a starting point from which to learn.
    /// The scaling helps to ensure that these initial values are neither too large nor too small,
    /// which can affect how well the network learns.
    /// </para>
    /// </remarks>
    private Tensor<T> InitializeTensor(int[] shape, T scale)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
        }

        return tensor;
    }

    /// <summary>
    /// Performs the forward pass of the attention mechanism.
    /// </summary>
    /// <param name="input">The input tensor to the layer.</param>
    /// <returns>The output tensor after applying the attention mechanism.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core functionality of the attention mechanism. It transforms the input
    /// into query, key, and value representations, computes attention scores, applies scaling and activation,
    /// and produces the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the attention magic happens!
    /// 
    /// 1. The input is transformed into three different representations: Query (Q), Key (K), and Value (V).
    /// 2. Attention scores are computed by comparing Q and K.
    /// 3. These scores are scaled and activated (usually with softmax) to get attention weights.
    /// 4. The final output is produced by applying these weights to V.
    /// 
    /// This process allows the layer to focus on different parts of the input as needed.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        var Q = input.Multiply(_Wq);
        var K = input.Multiply(_Wk);
        var V = input.Multiply(_Wv);

        var attentionScores = Q.Multiply(K.Transpose([1, 0]));
        var scaleFactor = NumOps.Sqrt(NumOps.FromDouble(K.Shape[K.Shape.Length - 1]));
        attentionScores = attentionScores.Scale(NumOps.Divide(NumOps.One, scaleFactor));

        _lastAttentionWeights = ApplyActivation(attentionScores);

        var output = _lastAttentionWeights.Multiply(V);
        return output;
    }

    /// <summary>
    /// Performs the forward pass of the attention mechanism with multiple inputs.
    /// </summary>
    /// <param name="inputs">An array of input tensors. Based on the number of inputs:
    ///   - One input: Standard forward pass with just the input tensor
    ///   - Two inputs: The first tensor is the query input, the second is either the key/value input or an attention mask
    ///   - Three inputs: The first tensor is the query input, the second is the key/value input, and the third is the attention mask
    /// </param>
    /// <returns>The output tensor after applying the attention mechanism.</returns>
    /// <exception cref="ArgumentException">Thrown when the input array is empty.</exception>
    /// <remarks>
    /// <para>
    /// This method extends the attention mechanism to support multiple input tensors, which is useful
    /// for implementing cross-attention (as used in transformer decoder layers) and masked attention.
    /// </para>
    /// <para><b>For Beginners:</b> This method allows the attention layer to handle more complex scenarios:
    /// 
    /// 1. With one input: It works just like the standard attention (self-attention)
    /// 2. With two inputs: It can either:
    ///    - Perform cross-attention (where query comes from one source, and key/value from another)
    ///    - Apply a mask to self-attention to control which parts of the input to focus on
    /// 3. With three inputs: It performs masked cross-attention, which combines both features above
    /// 
    /// These capabilities are essential for transformer architectures, especially decoder layers
    /// that need to attend to both their own outputs and the encoder's outputs.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required for attention mechanism.");

        // Case 1: Standard self-attention with a single input
        if (inputs.Length == 1)
        {
            return Forward(inputs[0]);
        }
    
        // Case 2: Either cross-attention or masked self-attention
        else if (inputs.Length == 2)
        {
            var primaryInput = inputs[0];
            var secondInput = inputs[1];
        
            // Check if the second input could be a mask (typically has a different shape)
            if (primaryInput.Shape[1] == secondInput.Shape[1] && secondInput.Shape.Length == primaryInput.Shape.Length)
            {
                // This appears to be cross-attention (query from primaryInput, key/value from secondInput)
                return ForwardCrossAttention(primaryInput, secondInput, null);
            }
            else
            {
                // This appears to be masked self-attention
                return ForwardMaskedAttention(primaryInput, secondInput);
            }
        }
    
        // Case 3: Masked cross-attention with three inputs
        else if (inputs.Length == 3)
        {
            var queryInput = inputs[0];
            var keyValueInput = inputs[1];
            var attentionMask = inputs[2];
        
            return ForwardCrossAttention(queryInput, keyValueInput, attentionMask);
        }
    
        // Unsupported number of inputs
        else
        {
            throw new ArgumentException($"Unsupported number of inputs ({inputs.Length}) for attention mechanism. Expected 1-3 inputs.");
        }
    }

    /// <summary>
    /// Performs masked self-attention, where query, key, and value all come from the same input,
    /// but with an attention mask applied.
    /// </summary>
    /// <param name="input">The input tensor for query, key, and value.</param>
    /// <param name="mask">The attention mask tensor.</param>
    /// <returns>The output tensor after applying masked self-attention.</returns>
    private Tensor<T> ForwardMaskedAttention(Tensor<T> input, Tensor<T> mask)
    {
        _lastInput = input;

        var Q = input.Multiply(_Wq);
        var K = input.Multiply(_Wk);
        var V = input.Multiply(_Wv);

        var attentionScores = Q.Multiply(K.Transpose([1, 0]));
    
        // Apply scaling factor
        var scaleFactor = NumOps.Sqrt(NumOps.FromDouble(K.Shape[K.Shape.Length - 1]));
        attentionScores = attentionScores.Scale(NumOps.Divide(NumOps.One, scaleFactor));
    
        // Apply mask - typically mask values are 0 for positions to attend to and very negative (e.g., -10000) for positions to ignore
        attentionScores = attentionScores.Add(mask);

        _lastAttentionWeights = ApplyActivation(attentionScores);

        var output = _lastAttentionWeights.Multiply(V);
        return output;
    }

    /// <summary>
    /// Performs cross-attention, where query comes from one input and key/value come from another,
    /// optionally with an attention mask applied.
    /// </summary>
    /// <param name="queryInput">The input tensor for query.</param>
    /// <param name="keyValueInput">The input tensor for key and value.</param>
    /// <param name="mask">Optional attention mask tensor.</param>
    /// <returns>The output tensor after applying cross-attention.</returns>
    private Tensor<T> ForwardCrossAttention(Tensor<T> queryInput, Tensor<T> keyValueInput, Tensor<T>? mask)
    {
        _lastInput = queryInput;  // Store the query input for backward pass

        var Q = queryInput.Multiply(_Wq);
        var K = keyValueInput.Multiply(_Wk);
        var V = keyValueInput.Multiply(_Wv);

        var attentionScores = Q.Multiply(K.Transpose([1, 0]));
    
        // Apply scaling factor
        var scaleFactor = NumOps.Sqrt(NumOps.FromDouble(K.Shape[K.Shape.Length - 1]));
        attentionScores = attentionScores.Scale(NumOps.Divide(NumOps.One, scaleFactor));
    
        // Apply mask if provided
        if (mask != null)
        {
            attentionScores = attentionScores.Add(mask);
        }

        _lastAttentionWeights = ApplyActivation(attentionScores);

        var output = _lastAttentionWeights.Multiply(V);
        return output;
    }

    /// <summary>
    /// Performs the backward pass of the attention mechanism.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backpropagation algorithm for the attention mechanism. It computes
    /// the gradients of the loss with respect to the layer's parameters and input.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the layer learns from its mistakes.
    /// 
    /// The method takes the gradient of the error with respect to the layer's output and works backwards to figure out:
    /// 1. How much each weight contributed to the error (stored in _dWq, _dWk, _dWv)
    /// 2. How the input itself contributed to the error (the returned value)
    /// 
    /// This information is then used to update the weights and improve the layer's performance.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastAttentionWeights == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var dV = _lastAttentionWeights.Transpose([1, 0]).Multiply(outputGradient);
        var V = _lastInput.Multiply(_Wv);
        var dAttentionWeights = outputGradient.Multiply(V.Transpose([1, 0]));

        var dAttentionScores = _lastAttentionWeights.ElementwiseMultiply(
            dAttentionWeights.Subtract(_lastAttentionWeights.SumOverAxis(-1).Reshape(_lastAttentionWeights.Shape).ElementwiseMultiply(dAttentionWeights))
        );

        var scaleFactor = NumOps.Sqrt(NumOps.FromDouble(_Wk.Shape[_Wk.Shape.Length - 1]));
        dAttentionScores = dAttentionScores.Scale(NumOps.Divide(NumOps.One, scaleFactor));

        var dK = _lastInput.Transpose([1, 0]).Multiply(dAttentionScores);
        var dQ = dAttentionScores.Multiply(_lastInput);

        _dWv = _lastInput.Transpose([1, 0]).Multiply(dV);
        _dWk = _lastInput.Transpose([1, 0]).Multiply(dK);
        _dWq = _lastInput.Transpose([1, 0]).Multiply(dQ);

        var dinput = dQ.Multiply(_Wq.Transpose([1, 0]))
                    .Add(dK.Multiply(_Wk.Transpose([1, 0])))
                    .Add(dV.Multiply(_Wv.Transpose([1, 0])));

        return dinput;
    }

    /// <summary>
    /// Updates the layer's parameters based on the computed gradients and a learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <remarks>
    /// <para>
    /// This method applies the computed gradients to the layer's weights, scaled by the learning rate.
    /// This is typically called after the backward pass to adjust the layer's parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the layer actually improves its performance.
    /// 
    /// After figuring out how each weight contributed to the error (in the Backward method),
    /// this method adjusts those weights to reduce the error:
    /// - Weights that contributed to large errors are changed more.
    /// - The learning rate determines how big these changes are.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_dWq == null || _dWk == null || _dWv == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _Wq = _Wq.Subtract(_dWq.Scale(learningRate));
        _Wk = _Wk.Subtract(_dWk.Scale(learningRate));
        _Wv = _Wv.Subtract(_dWv.Scale(learningRate));
    }

    /// <summary>
    /// Updates the layer's parameters with the provided values.
    /// </summary>
    /// <param name="parameters">A vector containing new parameter values.</param>
    /// <remarks>
    /// <para>
    /// This method replaces the current values of the layer's weights with new values provided in the parameters vector.
    /// It's useful for setting the layer's state to a specific configuration, such as when loading a pre-trained model.
    /// </para>
    /// <para><b>For Beginners:</b> This allows you to directly set the layer's internal weights.
    /// 
    /// Instead of the layer learning these weights through training, you're providing them directly.
    /// This is often used when you want to use a pre-trained attention layer or set up the layer with specific initial values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
    
        // Update Wq
        for (int i = 0; i < _Wq.Length; i++)
        {
            _Wq[i] = parameters[startIndex++];
        }
    
        // Update Wk
        for (int i = 0; i < _Wk.Length; i++)
        {
            _Wk[i] = parameters[startIndex++];
        }
    
        // Update Wv
        for (int i = 0; i < _Wv.Length; i++)
        {
            _Wv[i] = parameters[startIndex++];
        }
    }

    /// <summary>
    /// Retrieves the current parameters of the layer.
    /// </summary>
    /// <returns>A vector containing all the parameters of the layer.</returns>
    /// <remarks>
    /// <para>
    /// This method collects all the weights of the attention layer (Wq, Wk, Wv) into a single vector.
    /// It's useful for operations that need to work with all the layer's parameters at once,
    /// such as certain optimization algorithms or when saving the model's state.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you all the layer's learned values in one list.
    /// 
    /// It's like taking a snapshot of everything the layer has learned.
    /// This can be useful for saving the layer's current state or for advanced training techniques.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Get Wq parameters
        for (int i = 0; i < _Wq.Length; i++)
        {
            parameters[index++] = _Wq[i];
        }

        // Get Wk parameters
        for (int i = 0; i < _Wk.Length; i++)
        {
            parameters[index++] = _Wk[i];
        }

        // Get Wv parameters
        for (int i = 0; i < _Wv.Length; i++)
        {
            parameters[index++] = _Wv[i];
        }

        return parameters;
    }

    /// <summary>
    /// Resets the state of the attention layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the attention layer. It clears the last input
    /// and attention weights, effectively preparing the layer for a new sequence or episode.
    /// </para>
    /// <para><b>For Beginners:</b> This is like clearing the layer's short-term memory.
    /// 
    /// In attention mechanisms, sometimes we want to start fresh, forgetting any previous inputs.
    /// This is especially useful when starting a new sequence or when you don't want the layer
    /// to consider past information anymore.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastAttentionWeights = null;
    }
}