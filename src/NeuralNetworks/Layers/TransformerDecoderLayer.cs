namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a transformer decoder layer that processes sequences using self-attention, cross-attention, and feed-forward networks.
/// </summary>
/// <remarks>
/// <para>
/// A transformer decoder layer is a fundamental building block of transformer-based models for sequence-to-sequence tasks.
/// It consists of three main components: a masked self-attention mechanism that processes the target sequence, a cross-attention
/// mechanism that attends to the encoder's output, and a feed-forward network for additional transformation. Each component
/// is followed by layer normalization and residual connections to facilitate training of deep networks.
/// </para>
/// <para><b>For Beginners:</b> This layer helps the network generate sequences while considering both what it has generated so far and input from another source.
/// 
/// Think of it like a writer who is translating a book:
/// - First, the writer looks at what they've translated so far to maintain consistency (self-attention)
/// - Then they look at the original text to understand what to translate next (cross-attention)
/// - Finally, they process all this information to produce the next part of the translation (feed-forward network)
/// 
/// For example, in machine translation, the decoder generates each word of the target language by:
/// - Looking at the words it has already generated (to maintain grammatical coherence)
/// - Looking at the encoded source sentence (to understand what content to translate)
/// - Combining this information to produce the most appropriate next word
/// 
/// This architecture is powerful for tasks like translation, summarization, and text generation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TransformerDecoderLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The size of the embeddings for queries, keys, values, and outputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the dimensionality of the embedding vectors used throughout the transformer decoder layer.
    /// It determines the size of the attention heads and the input/output dimensions of the various sublayers.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many features or dimensions each word or token has.
    /// 
    /// Think of it as the "richness" of information for each element in the sequence:
    /// - Larger values (like 768 or 1024) give more capacity to represent complex patterns
    /// - Common values range from 128 for simple tasks to 1024+ for complex language models
    /// - This value stays constant throughout the entire transformer layer
    /// 
    /// For example, in a language model, this might be 512 dimensions to capture various
    /// aspects of each word like its meaning, grammatical role, and context.
    /// </para>
    /// </remarks>
    private readonly int _embeddingSize;

    /// <summary>
    /// The number of attention heads for the self-attention and cross-attention mechanisms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of attention heads used in the multi-head attention layers. Multiple heads allow
    /// the model to jointly attend to information from different representation subspaces at different positions.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many different perspectives the model can consider simultaneously.
    /// 
    /// Multi-head attention is like having multiple people look at the same text:
    /// - Each "head" focuses on different relationships in the data
    /// - One head might focus on subject-verb relationships
    /// - Another might focus on adjective-noun relationships
    /// - Together they capture a richer understanding of the context
    /// 
    /// Common values range from 8-16 heads, with larger models using more heads.
    /// This allows the model to "pay attention" to different aspects of the input at the same time.
    /// </para>
    /// </remarks>
    private readonly int _numHeads;

    /// <summary>
    /// The inner dimension of the feed-forward network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the dimensionality of the inner layer in the feed-forward network. This is typically
    /// larger than the embedding size, allowing the network to project into a higher-dimensional space for more
    /// expressive transformations before projecting back to the embedding size.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how much information the network can process internally after attention.
    /// 
    /// The feed-forward network:
    /// - Takes the output from the attention layers
    /// - Expands it to this larger dimension for more complex processing
    /// - Then compresses it back to the original embedding size
    /// 
    /// It's typically 4 times larger than the embedding size (e.g., 2048 for a 512 embedding size).
    /// This expanded dimension gives the network more capacity to transform the information.
    /// </para>
    /// </remarks>
    private readonly int _feedForwardDim;

    /// <summary>
    /// The maximum length of the input and output sequences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the maximum sequence length that the transformer decoder layer can process. It determines
    /// the size of position embeddings and attention matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This represents the maximum number of elements (like words) the model can process at once.
    /// 
    /// Sequence length determines:
    /// - How much context the model can consider at once
    /// - The maximum length of inputs and outputs
    /// - Memory usage during processing (longer sequences use more memory)
    /// 
    /// For example, in a language model, this might be 512 tokens (roughly equivalent to paragraphs of text).
    /// This limit exists because the attention mechanism needs to compare each element with every other element,
    /// which becomes computationally expensive for very long sequences.
    /// </para>
    /// </remarks>
    private readonly int _sequenceLength;

    /// <summary>
    /// The self-attention mechanism for processing the decoder input sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the multi-head self-attention layer that allows the decoder to attend to its own previous outputs.
    /// In practice, this is usually implemented with masking to prevent attending to future positions.
    /// </para>
    /// <para><b>For Beginners:</b> This component helps the model consider what it has generated so far.
    /// 
    /// Self-attention works by:
    /// - Looking at the sequence the decoder has generated so far
    /// - Determining which previous elements are most relevant to the current position
    /// - Using this information to maintain consistency and coherence
    /// 
    /// In text generation, this is how the model makes sure that what it's generating now makes sense
    /// with what it has already generated, maintaining grammar, style, and topic consistency.
    /// </para>
    /// </remarks>
    private MultiHeadAttentionLayer<T> _selfAttention = default!;

    /// <summary>
    /// The layer normalization applied after self-attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the layer normalization that is applied to the output of the self-attention sublayer
    /// (after adding the residual connection). Layer normalization helps stabilize the learning process.
    /// </para>
    /// <para><b>For Beginners:</b> This component helps keep the values in a reasonable range after self-attention.
    /// 
    /// Layer normalization:
    /// - Adjusts the scale of the values to prevent them from growing too large or too small
    /// - Helps the network train more stably and quickly
    /// - Is applied after each major component in the transformer
    /// 
    /// Think of it like keeping the volume at a consistent level while listening to music -
    /// it prevents sudden spikes or drops that might disrupt the learning process.
    /// </para>
    /// </remarks>
    private LayerNormalizationLayer<T> _norm1 = default!;

    /// <summary>
    /// The cross-attention mechanism for attending to the encoder output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the multi-head cross-attention layer that allows the decoder to attend to the output of the encoder.
    /// This enables the decoder to focus on relevant parts of the input sequence when generating the output sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This component helps the model consider the input sequence (from the encoder) when generating output.
    /// 
    /// Cross-attention works by:
    /// - Looking at the entire input sequence (processed by the encoder)
    /// - Determining which parts of the input are most relevant for generating the current output
    /// - Focusing on those relevant parts to produce appropriate output
    /// 
    /// For example, in translation, this is how the model decides which words in the source language
    /// are most important when generating the next word in the target language.
    /// </para>
    /// </remarks>
    private MultiHeadAttentionLayer<T> _crossAttention = default!;

    /// <summary>
    /// The layer normalization applied after cross-attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the layer normalization that is applied to the output of the cross-attention sublayer
    /// (after adding the residual connection). Layer normalization helps stabilize the learning process.
    /// </para>
    /// <para><b>For Beginners:</b> This component helps keep the values in a reasonable range after cross-attention.
    /// 
    /// Like the first normalization layer, this:
    /// - Standardizes the output of the cross-attention component
    /// - Prevents extreme values that could disrupt training
    /// - Helps the network converge faster during training
    /// 
    /// Each normalization layer in the transformer acts as a stabilizing force,
    /// helping maintain consistent signal strength throughout the network.
    /// </para>
    /// </remarks>
    private LayerNormalizationLayer<T> _norm2 = default!;

    /// <summary>
    /// The feed-forward network for additional transformation of the sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the feed-forward network that applies a non-linear transformation to each position in the sequence independently.
    /// It consists of two linear transformations with an activation function in between.
    /// </para>
    /// <para><b>For Beginners:</b> This component processes the attended information to produce the final output for each position.
    /// 
    /// The feed-forward network:
    /// - Processes each position independently (unlike attention, which looks across positions)
    /// - Applies a more complex transformation with non-linearity
    /// - Helps the network learn more abstract patterns and relationships
    /// 
    /// This is where the model does its "thinking" after gathering information from self-attention
    /// and cross-attention, making final decisions about what to output.
    /// </para>
    /// </remarks>
    private FeedForwardLayer<T> _feedForward = default!;

    /// <summary>
    /// The layer normalization applied after the feed-forward network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the layer normalization that is applied to the output of the feed-forward sublayer
    /// (after adding the residual connection). Layer normalization helps stabilize the learning process.
    /// </para>
    /// <para><b>For Beginners:</b> This component helps keep the values in a reasonable range after the feed-forward processing.
    /// 
    /// Like the previous normalization layers, this:
    /// - Standardizes the output of the feed-forward component
    /// - Ensures the final output of the decoder layer has a consistent scale
    /// - Makes it easier for the next layer to process the output
    /// 
    /// This final normalization helps prepare the output for either the next decoder layer
    /// or for the final output projection in the complete transformer model.
    /// </para>
    /// </remarks>
    private LayerNormalizationLayer<T> _norm3 = default!;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the decoder input tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the decoder input (target sequence) from the last calculation.
    /// 
    /// Storing the input is necessary because:
    /// - During training, the layer needs to remember what decoder input it processed
    /// - This helps calculate the correct gradients during backpropagation
    /// - It's part of the layer's "memory" for the learning process
    /// 
    /// This cached input helps the layer understand how to adjust its parameters
    /// to improve its performance on future inputs.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The encoder output tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the encoder output tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients for the cross-attention mechanism.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the encoder's output (source sequence information) from the last calculation.
    /// 
    /// Storing the encoder output is necessary because:
    /// - The cross-attention mechanism needs this information during both forward and backward passes
    /// - It represents the source sequence that the decoder is conditioning its generation on
    /// - It needs to be available during training to compute proper gradients
    /// 
    /// For example, in translation, this would contain the processed representation of
    /// the source language sentence that the decoder is translating.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastEncoderOutput;

    /// <summary>
    /// The output tensor of the self-attention sublayer from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor of the self-attention sublayer from the most recent forward pass,
    /// which is needed during the backward pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the result after the self-attention step from the last calculation.
    /// 
    /// Storing the self-attention output:
    /// - Helps track the information flow through the network
    /// - Is used to compute accurate gradients during training
    /// - Forms part of the computational graph needed for backpropagation
    /// 
    /// These intermediate results are necessary for the layer to learn effectively,
    /// as they show how each component contributed to the final output.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastSelfAttentionOutput;

    /// <summary>
    /// The output tensor after the first normalization from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor after the first layer normalization from the most recent forward pass,
    /// which is needed during the backward pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the result after normalizing the self-attention output.
    /// 
    /// Like other intermediate results, this:
    /// - Tracks the data flow through the layer
    /// - Helps compute the correct gradients during training
    /// - Is part of the chain of calculations needed for learning
    /// 
    /// These saved states form a kind of "memory" that allows the network to understand
    /// how it arrived at its final output, so it can learn to improve.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastNormalized1;

    /// <summary>
    /// The output tensor of the cross-attention sublayer from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor of the cross-attention sublayer from the most recent forward pass,
    /// which is needed during the backward pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the result after the cross-attention step from the last calculation.
    /// 
    /// Similar to the self-attention output, this:
    /// - Records how the decoder attended to the encoder output
    /// - Is used in training to calculate how parameters should be updated
    /// - Helps track information flow between the encoder and decoder
    /// 
    /// This represents the information extracted from the source sequence that
    /// was deemed relevant for generating the target sequence.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastCrossAttentionOutput;

    /// <summary>
    /// The output tensor after the second normalization from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor after the second layer normalization from the most recent forward pass,
    /// which is needed during the backward pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the result after normalizing the cross-attention output.
    /// 
    /// This intermediate result:
    /// - Captures the state after cross-attention and before feed-forward processing
    /// - Forms another link in the chain of computations
    /// - Helps the network understand how information flowed through the layer
    /// 
    /// During training, these stored states help the network reconstruct exactly
    /// how it arrived at its final output, enabling precise learning.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastNormalized2;

    /// <summary>
    /// The output tensor of the feed-forward sublayer from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor of the feed-forward sublayer from the most recent forward pass,
    /// which is needed during the backward pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the result after the feed-forward processing step from the last calculation.
    /// 
    /// Like other intermediate results, this:
    /// - Tracks how the final transformation affected the output
    /// - Helps compute gradients for the feed-forward network during training
    /// - Contributes to the layer's ability to learn from its mistakes
    /// 
    /// This represents the final processing step before the last normalization
    /// and output of the complete decoder layer.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastFeedForwardOutput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, as it contains trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the transformer decoder layer can be trained through backpropagation.
    /// Since this layer has trainable parameters in its sublayers, it supports training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// For this layer, the value is always true because it contains multiple sublayers
    /// with trainable parameters that need to be optimized during training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerDecoderLayer{T}"/> class with scalar activation function.
    /// </summary>
    /// <param name="embeddingSize">The size of the embeddings. Default is 512.</param>
    /// <param name="numHeads">The number of attention heads. Default is 8.</param>
    /// <param name="feedForwardDim">The dimension of the feed-forward network. Default is 2048.</param>
    /// <param name="sequenceLength">The maximum sequence length. Default is 512.</param>
    /// <param name="ffnActivation">The activation function for the feed-forward network. Default is GELU.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a transformer decoder layer with the specified dimensions and a scalar activation function
    /// for the feed-forward network. It initializes all the sublayers needed for the transformer decoder architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new transformer decoder layer with standard settings.
    /// 
    /// The parameters you provide determine:
    /// - embeddingSize: How rich the representation of each token is (more = more expressive)
    /// - numHeads: How many different "perspectives" the attention mechanism can have
    /// - feedForwardDim: How much processing capacity the feed-forward network has
    /// - sequenceLength: The maximum number of tokens the model can process
    /// - ffnActivation: The mathematical function used in the feed-forward network
    /// 
    /// These settings control the capacity, expressiveness, and computational requirements of the decoder.
    /// The default values (512 embedding size, 8 heads, etc.) are similar to those used in the original
    /// transformer paper and work well for many language tasks.
    /// </para>
    /// </remarks>
    public TransformerDecoderLayer(int embeddingSize = 512, 
        int numHeads = 8, 
        int feedForwardDim = 2048, 
        int sequenceLength = 512,
        IActivationFunction<T>? ffnActivation = null)
        : base([embeddingSize], [embeddingSize])
    {
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;
        _sequenceLength = sequenceLength;

        var activation = ffnActivation ?? new GELUActivation<T>();
         
        // Self-attention layer (no activation)
        _selfAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm1 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Cross-attention layer (no activation)
        _crossAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm2 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Feed-forward layer (with activation)
        _feedForward = new FeedForwardLayer<T>(_embeddingSize, _feedForwardDim, activation);
        _norm3 = new LayerNormalizationLayer<T>(_embeddingSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerDecoderLayer{T}"/> class with vector activation function.
    /// </summary>
    /// <param name="embeddingSize">The size of the embeddings. Default is 512.</param>
    /// <param name="numHeads">The number of attention heads. Default is 8.</param>
    /// <param name="feedForwardDim">The dimension of the feed-forward network. Default is 2048.</param>
    /// <param name="sequenceLength">The maximum sequence length. Default is 512.</param>
    /// <param name="ffnVectorActivation">The vector activation function for the feed-forward network. Default is GELU.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a transformer decoder layer with the specified dimensions and a vector activation function
    /// for the feed-forward network. It initializes all the sublayers needed for the transformer decoder architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is similar to the previous one, but uses vector activations.
    /// 
    /// Vector<double> activations:
    /// - Process entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements
    /// - Allow for more complex transformations
    /// 
    /// This version is useful when you need more sophisticated processing that considers
    /// how different features relate to each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public TransformerDecoderLayer(int embeddingSize = 512, 
        int numHeads = 8, 
        int feedForwardDim = 2048, 
        int sequenceLength = 512,
        IVectorActivationFunction<T>? ffnVectorActivation = null)
        : base([embeddingSize], [embeddingSize])
    {
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;
        _sequenceLength = sequenceLength;

        var activation = ffnVectorActivation ?? new GELUActivation<T>();

        // Self-attention layer (no activation)
        _selfAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm1 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Cross-attention layer (no activation)
        _crossAttention = new MultiHeadAttentionLayer<T>(_sequenceLength, _embeddingSize, _numHeads, activation);
        _norm2 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Feed-forward layer (with vector activation)
        _feedForward = new FeedForwardLayer<T>(_embeddingSize, _feedForwardDim, activation);
        _norm3 = new LayerNormalizationLayer<T>(_embeddingSize);
    }

    /// <summary>
    /// Not supported for this layer. Use Forward(Tensor<double>&lt;T&gt; input, Tensor<double>&lt;T&gt; encoderOutput) instead.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>Never returns as this method throws an exception.</returns>
    /// <exception cref="InvalidOperationException">Always thrown, as this method is not supported for this layer.</exception>
    /// <remarks>
    /// <para>
    /// This method is not supported for the transformer decoder layer, as it requires both a decoder input and an encoder output.
    /// Use the overloaded Forward method that accepts both inputs instead.
    /// </para>
    /// <para><b>For Beginners:</b> This method is a placeholder that shows an error if used incorrectly.
    /// 
    /// The transformer decoder needs two inputs:
    /// - The decoder's own input (what it has generated so far)
    /// - The encoder's output (information from the source sequence)
    /// 
    /// This method exists only to satisfy the base class requirements, but will show an error
    /// if someone tries to use it. The correct method to use is the one that accepts both inputs.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new InvalidOperationException("Use Forward(Tensor<T> input, Tensor<T> encoderOutput) for TransformerDecoderLayer.");
    }

    /// <summary>
    /// Performs the forward pass of the transformer decoder layer.
    /// </summary>
    /// <param name="input">The decoder input tensor.</param>
    /// <param name="encoderOutput">The encoder output tensor.</param>
    /// <returns>The output tensor after processing through the transformer decoder layer.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the transformer decoder layer. It processes the decoder input through
    /// the self-attention mechanism, applies layer normalization and a residual connection, then passes the result through
    /// the cross-attention mechanism (attending to the encoder output), applies another layer normalization and residual
    /// connection, and finally processes the result through the feed-forward network followed by a final layer normalization
    /// and residual connection.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the inputs through all components of the decoder layer.
    /// 
    /// The forward pass follows these steps:
    /// 
    /// 1. Self-Attention:
    ///    - The decoder looks at its own input to understand the context of what it has generated so far
    ///    - The result is added to the original input (residual connection)
    ///    - Layer normalization is applied to stabilize the values
    /// 
    /// 2. Cross-Attention:
    ///    - The decoder looks at the encoder output to gather information from the source sequence
    ///    - The result is added to the output from step 1 (residual connection)
    ///    - Layer normalization is applied again
    /// 
    /// 3. Feed-Forward Network:
    ///    - The output from step 2 is processed through a feed-forward network
    ///    - The result is added to the output from step 2 (residual connection)
    ///    - A final layer normalization is applied
    /// 
    /// These steps allow the decoder to generate output that is coherent with both
    /// what it has generated so far and the information from the source sequence.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> encoderOutput)
    {
        _lastInput = input;
        _lastEncoderOutput = encoderOutput;

        _lastSelfAttentionOutput = _selfAttention.Forward(input);
        _lastNormalized1 = _norm1.Forward(input + _lastSelfAttentionOutput);
        _lastCrossAttentionOutput = _crossAttention.Forward(_lastNormalized1, encoderOutput);
        _lastNormalized2 = _norm2.Forward(_lastNormalized1 + _lastCrossAttentionOutput);
        _lastFeedForwardOutput = _feedForward.Forward(_lastNormalized2);
        var output = _norm3.Forward(_lastNormalized2 + _lastFeedForwardOutput);

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the transformer decoder layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the transformer decoder layer, which is used during training to
    /// propagate error gradients back through the network. It computes gradients for each sublayer in reverse order
    /// of the forward pass, ensuring that residual connections are properly handled.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the layer's inputs should change to reduce errors.
    /// 
    /// During the backward pass, we go through the same steps as the forward pass, but in reverse order:
    /// 
    /// 1. Final Layer Normalization:
    ///    - Compute how the normalization's input should change based on output errors
    /// 
    /// 2. Feed-Forward Network:
    ///    - Determine how the feed-forward network's input should change
    ///    - Account for the residual connection by adding gradients
    /// 
    /// 3. Second Layer Normalization:
    ///    - Compute how the second normalization's input should change
    /// 
    /// 4. Cross-Attention:
    ///    - Determine how the cross-attention's inputs should change
    ///    - Account for the residual connection
    /// 
    /// 5. First Layer Normalization:
    ///    - Compute how the first normalization's input should change
    /// 
    /// 6. Self-Attention:
    ///    - Determine how the self-attention's input should change
    ///    - Account for the final residual connection
    /// 
    /// This reverse flow of gradients allows each component to learn how it contributed to any errors.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var gradNorm3 = _norm3.Backward(outputGradient);
        var gradFeedForward = _feedForward.Backward(gradNorm3);
        var gradNorm2 = _norm2.Backward(gradFeedForward + gradNorm3);
        var gradCrossAttention = _crossAttention.Backward(gradNorm2);
        var gradNorm1 = _norm1.Backward(gradCrossAttention + gradNorm2);
        var gradSelfAttention = _selfAttention.Backward(gradNorm1);

        return gradSelfAttention + gradNorm1;
    }

    /// <summary>
    /// Updates the parameters of all sublayers using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of all sublayers in the transformer decoder layer based on the gradients
    /// calculated during the backward pass. It delegates the update process to each sublayer, passing the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts all the internal values of the layer to improve its performance.
    /// 
    /// During parameter updates:
    /// - The learning rate controls how big each adjustment is
    /// - Every sublayer gets updated based on what was learned in the backward pass
    /// - This helps the entire decoder layer gradually improve its performance
    /// 
    /// Think of it like fine-tuning all the components of the decoder based on feedback:
    /// - The self-attention mechanism learns to focus on more relevant parts of what's been generated
    /// - The cross-attention mechanism learns to extract more useful information from the source
    /// - The feed-forward network learns to better transform this information into the next output
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _selfAttention.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _crossAttention.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _feedForward.UpdateParameters(learningRate);
        _norm3.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters of the transformer decoder layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters from all sublayers.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from all sublayers of the transformer decoder layer and combines
    /// them into a single vector. This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from all parts of the decoder.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include weights from attention mechanisms, normalization layers, and the feed-forward network
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// A transformer decoder layer typically has millions of parameters, all of which
    /// contribute to its ability to generate high-quality sequences.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Collect parameters from all sublayers
        var selfAttentionParams = _selfAttention.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var crossAttentionParams = _crossAttention.GetParameters();
        var norm2Params = _norm2.GetParameters();
        var feedForwardParams = _feedForward.GetParameters();
        var norm3Params = _norm3.GetParameters();
    
        // Calculate total parameter count
        int totalParamCount = selfAttentionParams.Length + 
                              norm1Params.Length + 
                              crossAttentionParams.Length + 
                              norm2Params.Length + 
                              feedForwardParams.Length + 
                              norm3Params.Length;
    
        // Create a vector to hold all parameters
        var parameters = new Vector<T>(totalParamCount);
    
        // Copy all parameters into the combined vector
        int currentIndex = 0;
    
        // Copy self-attention parameters
        for (int i = 0; i < selfAttentionParams.Length; i++)
            parameters[currentIndex++] = selfAttentionParams[i];
    
        // Copy norm1 parameters
        for (int i = 0; i < norm1Params.Length; i++)
            parameters[currentIndex++] = norm1Params[i];
    
        // Copy cross-attention parameters
        for (int i = 0; i < crossAttentionParams.Length; i++)
            parameters[currentIndex++] = crossAttentionParams[i];
    
        // Copy norm2 parameters
        for (int i = 0; i < norm2Params.Length; i++)
            parameters[currentIndex++] = norm2Params[i];
    
        // Copy feed-forward parameters
        for (int i = 0; i < feedForwardParams.Length; i++)
            parameters[currentIndex++] = feedForwardParams[i];
    
        // Copy norm3 parameters
        for (int i = 0; i < norm3Params.Length; i++)
            parameters[currentIndex++] = norm3Params[i];
    
        return parameters;
    }

    /// <summary>
    /// Resets the internal state of the transformer decoder layer and all its sublayers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the transformer decoder layer and all its sublayers. It clears the cached
    /// tensors from the forward pass and delegates the reset operation to each sublayer.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - All sublayers are reset to their initial condition
    /// - Stored inputs and outputs are cleared
    /// - The layer forgets all intermediate results from previous processing
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Starting a new training episode
    /// - Testing the layer with fresh inputs
    /// 
    /// Think of it like clearing the entire team's mind before starting a completely new task,
    /// ensuring no residual information affects the processing of new inputs.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Reset all sublayers
        _selfAttention.ResetState();
        _norm1.ResetState();
        _crossAttention.ResetState();
        _norm2.ResetState();
        _feedForward.ResetState();
        _norm3.ResetState();
    
        // Clear cached tensors
        _lastInput = null;
        _lastEncoderOutput = null;
        _lastSelfAttentionOutput = null;
        _lastNormalized1 = null;
        _lastCrossAttentionOutput = null;
        _lastNormalized2 = null;
        _lastFeedForwardOutput = null;
    }
}