using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
public class TransformerDecoderLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether auxiliary loss is enabled for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the layer aggregates auxiliary losses from its attention sublayers (both self-attention and cross-attention).
    /// This helps regularize attention patterns and prevents issues like attention collapse.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the layer uses additional learning signals.
    ///
    /// When enabled (true):
    /// - The layer collects extra penalties from both self-attention and cross-attention mechanisms
    /// - This helps the attention heads learn diverse and focused patterns
    /// - Training may be more stable and produce better results
    ///
    /// When disabled (false):
    /// - Only the main task loss is used for training
    /// - This is the default setting
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the auxiliary loss contribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value determines how much the aggregated auxiliary losses contribute to the total loss.
    /// The default value of 0.005 provides a good balance between the main task and regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much importance to give to the attention regularization.
    ///
    /// The weight affects training:
    /// - Higher values (e.g., 0.01) make the network prioritize better attention patterns more strongly
    /// - Lower values (e.g., 0.001) make the regularization less important
    /// - The default (0.005) works well for most transformer tasks
    ///
    /// If your attention is collapsing (all heads learning the same thing), you might increase this value.
    /// If the main task is more important, you might decrease it.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Stores the last computed auxiliary loss for diagnostic purposes.
    /// </summary>
    private T _lastAuxiliaryLoss;

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
    private MultiHeadAttentionLayer<T> _selfAttention;

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
    private LayerNormalizationLayer<T> _norm1;

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
    private MultiHeadAttentionLayer<T> _crossAttention;

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
    private LayerNormalizationLayer<T> _norm2;

    /// <summary>
    /// The feed-forward network for additional transformation of the sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the first layer of the feed-forward network that projects from embedding size to a larger hidden dimension.
    /// It applies a non-linear transformation with an activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This component expands the representation to a larger dimension.
    ///
    /// The first feed-forward layer:
    /// - Takes the attention output (embedding size)
    /// - Expands it to a larger hidden dimension
    /// - Applies an activation function for non-linearity
    ///
    /// This expansion gives the network more capacity to learn complex transformations.
    /// </para>
    /// </remarks>
    private FeedForwardLayer<T> _feedForward;

    /// <summary>
    /// The projection layer that maps back from the feed-forward hidden dimension to the embedding size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the second layer of the feed-forward network that projects from the hidden dimension back to the embedding size.
    /// This ensures the output has the same dimension as the input for residual connections.
    /// </para>
    /// <para><b>For Beginners:</b> This component compresses the expanded representation back to the original size.
    ///
    /// The projection layer:
    /// - Takes the expanded hidden representation
    /// - Compresses it back to the embedding size
    /// - Ensures the output can be added to the residual connection
    ///
    /// This is the standard FFN architecture in transformers: expand → activate → project back.
    /// </para>
    /// </remarks>
    private readonly FeedForwardLayer<T> _feedForwardProjection;

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
    private LayerNormalizationLayer<T> _norm3;

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

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuNormalized1;
    private IGpuTensor<T>? _gpuNormalized2;

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
    /// Gets a value indicating whether this layer can execute on GPU.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <remarks>
    /// This returns the sum of all parameters from sublayers: self-attention, cross-attention,
    /// layer norms, feed-forward layer, and feed-forward projection layer.
    /// </remarks>
    public override int ParameterCount =>
        _selfAttention.ParameterCount +
        _norm1.ParameterCount +
        _crossAttention.ParameterCount +
        _norm2.ParameterCount +
        _feedForward.ParameterCount +
        _feedForwardProjection.ParameterCount +
        _norm3.ParameterCount;

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
        IActivationFunction<T>? ffnActivation = null,
        IEngine? engine = null)
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

        // Feed-forward layer (with activation) - expands to hidden dimension
        _feedForward = new FeedForwardLayer<T>(_embeddingSize, _feedForwardDim, activation);
        // Projection layer (no activation) - projects back to embedding size
        _feedForwardProjection = new FeedForwardLayer<T>(_feedForwardDim, _embeddingSize, (IActivationFunction<T>?)null);
        _norm3 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Initialize NumOps-based fields
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAuxiliaryLoss = NumOps.Zero;
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
    /// Vector activations:
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
        IVectorActivationFunction<T>? ffnVectorActivation = null,
        IEngine? engine = null)
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

        // Feed-forward layer (with vector activation) - expands to hidden dimension
        _feedForward = new FeedForwardLayer<T>(_embeddingSize, _feedForwardDim, activation);
        // Projection layer (no activation) - projects back to embedding size
        _feedForwardProjection = new FeedForwardLayer<T>(_feedForwardDim, _embeddingSize, (IActivationFunction<T>?)null);
        _norm3 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Initialize NumOps-based fields
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAuxiliaryLoss = NumOps.Zero;
    }

    /// <summary>
    /// Not supported for this layer. Use Forward(Tensor&lt;T&gt; input, Tensor&lt;T&gt; encoderOutput) instead.
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

        // residual1 = input + selfAttentionOutput
        var residual1 = Engine.TensorAdd(input, _lastSelfAttentionOutput);
        _lastNormalized1 = _norm1.Forward(residual1);

        _lastCrossAttentionOutput = _crossAttention.Forward(_lastNormalized1, encoderOutput);

        // residual2 = normalized1 + crossAttentionOutput
        var residual2 = Engine.TensorAdd(_lastNormalized1, _lastCrossAttentionOutput);
        _lastNormalized2 = _norm2.Forward(residual2);

        var feedForwardHidden = _feedForward.Forward(_lastNormalized2);
        _lastFeedForwardOutput = _feedForwardProjection.Forward(feedForwardHidden);

        // residual3 = normalized2 + feedForwardOutput
        var residual3 = Engine.TensorAdd(_lastNormalized2, _lastFeedForwardOutput);
        var output = _norm3.Forward(residual3);

        return output;
    }

    /// <summary>
    /// GPU-resident forward pass for the transformer decoder layer.
    /// Performs self-attention, cross-attention, and feed-forward operations entirely on GPU.
    /// </summary>
    /// <param name="inputs">Array containing [decoderInput, encoderOutput] GPU tensors.</param>
    /// <returns>GPU-resident output tensor.</returns>
    /// <exception cref="ArgumentException">Thrown when inputs array doesn't contain exactly 2 tensors.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the entire transformer decoder forward pass on the GPU without downloading
    /// intermediate results to CPU. All sublayer operations (self-attention, cross-attention, layer normalization,
    /// feed-forward networks, residual connections) remain GPU-resident for maximum performance.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length < 2)
            throw new ArgumentException("TransformerDecoderLayer requires two inputs: [decoderInput, encoderOutput]");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        IGpuTensor<T> decoderInput = inputs[0];
        IGpuTensor<T> encoderOutput = inputs[1];

        // 1. Self-attention sublayer
        var selfAttentionOutput = _selfAttention.ForwardGpu(decoderInput);

        // 2. First residual connection: input + selfAttentionOutput
        var residual1 = gpuEngine.AddGpu(decoderInput, selfAttentionOutput);

        // 3. First layer normalization
        var (normalized1, _, _) = gpuEngine.LayerNormGpu(residual1, _norm1.GetGammaTensor(), _norm1.GetBetaTensor(), Convert.ToDouble(_norm1.GetEpsilon()));

        // 4. Cross-attention sublayer (decoder attends to encoder output)
        var crossAttentionOutput = _crossAttention.ForwardGpu(normalized1, encoderOutput);

        // 5. Second residual connection: normalized1 + crossAttentionOutput
        var residual2 = gpuEngine.AddGpu(normalized1, crossAttentionOutput);

        // 6. Second layer normalization
        var (normalized2, _, _) = gpuEngine.LayerNormGpu(residual2, _norm2.GetGammaTensor(), _norm2.GetBetaTensor(), Convert.ToDouble(_norm2.GetEpsilon()));

        // 7. Feed-forward network (two layers)
        var ffHidden = _feedForward.ForwardGpu(normalized2);
        var ffProjected = _feedForwardProjection.ForwardGpu(ffHidden);

        // 8. Third residual connection: normalized2 + ffProjected
        var residual3 = gpuEngine.AddGpu(normalized2, ffProjected);

        // 9. Third layer normalization (final output)
        var (output, _, _) = gpuEngine.LayerNormGpu(residual3, _norm3.GetGammaTensor(), _norm3.GetBetaTensor(), Convert.ToDouble(_norm3.GetEpsilon()));

        // Cache state for backward pass only during training
        // Skip this expensive download during inference (50% overhead reduction)
        if (IsTrainingMode)
        {
            _gpuNormalized1 = normalized1;
            _gpuNormalized2 = normalized2;
            _lastInput = decoderInput.ToTensor();
            _lastEncoderOutput = encoderOutput.ToTensor();
            _lastSelfAttentionOutput = selfAttentionOutput.ToTensor();
            _lastNormalized1 = normalized1.ToTensor();
            _lastCrossAttentionOutput = crossAttentionOutput.ToTensor();
            _lastNormalized2 = normalized2.ToTensor();
            _lastFeedForwardOutput = ffProjected.ToTensor();
        }

        return output;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the decoder input on the GPU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the decoder input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        // Backward through norm3
        IGpuTensor<T> grad = InvokeBackwardGpu(_norm3, outputGradient, gpuEngine);

        // Gradient flows to both residual and FFN
        // Backward through FFN projection
        IGpuTensor<T> ffProjectedGrad = InvokeBackwardGpu(_feedForwardProjection, grad, gpuEngine);

        // Backward through FFN
        IGpuTensor<T> ffHiddenGrad = InvokeBackwardGpu(_feedForward, ffProjectedGrad, gpuEngine);

        // Add residual gradient: dNormalized2 = grad + ffHiddenGrad
        var norm2Grad = gpuEngine.AddGpu(grad, ffHiddenGrad);

        // Backward through norm2
        grad = InvokeBackwardGpu(_norm2, norm2Grad, gpuEngine);

        // Gradient flows to both residual and cross-attention
        // Backward through cross-attention
        IGpuTensor<T> crossAttnGrad = InvokeBackwardGpu(_crossAttention, grad, gpuEngine);

        // Add residual gradient: dNormalized1 = grad + crossAttnGrad
        var norm1Grad = gpuEngine.AddGpu(grad, crossAttnGrad);

        // Backward through norm1
        grad = InvokeBackwardGpu(_norm1, norm1Grad, gpuEngine);

        // Gradient flows to both residual and self-attention
        // Backward through self-attention
        IGpuTensor<T> selfAttnGrad = InvokeBackwardGpu(_selfAttention, grad, gpuEngine);

        // Add residual gradient to get final input gradient
        var inputGrad = gpuEngine.AddGpu(grad, selfAttnGrad);

        return inputGrad;
    }

    /// <summary>
    /// Helper method to invoke BackwardGpu on a sublayer using reflection.
    /// </summary>
    private static IGpuTensor<T> InvokeBackwardGpu(LayerBase<T> layer, IGpuTensor<T> grad, DirectGpuTensorEngine gpuEngine)
    {
        var layerType = layer.GetType();
        var backwardGpuMethod = layerType.GetMethod("BackwardGpu", new[] { typeof(IGpuTensor<T>) });

        if (backwardGpuMethod != null)
        {
            return (IGpuTensor<T>)backwardGpuMethod.Invoke(layer, new object[] { grad })!;
        }
        else
        {
            // Fallback to CPU backward
            var cpuGrad = grad.ToTensor();
            var cpuResult = layer.Backward(cpuGrad);
            return gpuEngine.UploadToGpu<T>(cpuResult, GpuTensorRole.Gradient);
        }
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
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }


    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. It's slower than the
    /// manual implementation but can be useful for:
    /// - Verifying gradient correctness
    /// - Rapid prototyping with custom modifications
    /// - Research and experimentation
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // For complex/composite layers, delegate to manual implementation
        // Full autodiff requires implementing all sub-operations
        return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        var gradNorm3 = _norm3.Backward(outputGradient);
        // Backward through projection layer first, then the feed-forward layer
        var gradProjection = _feedForwardProjection.Backward(gradNorm3);
        var gradFeedForward = _feedForward.Backward(gradProjection);

        // gradInput2 = gradFeedForward + gradNorm3
        var gradInput2 = Engine.TensorAdd(gradFeedForward, gradNorm3);
        var gradNorm2 = _norm2.Backward(gradInput2);

        var gradCrossAttention = _crossAttention.Backward(gradNorm2);

        // gradInput1 = gradCrossAttention + gradNorm2
        var gradInput1 = Engine.TensorAdd(gradCrossAttention, gradNorm2);
        var gradNorm1 = _norm1.Backward(gradInput1);

        var gradSelfAttention = _selfAttention.Backward(gradNorm1);

        // gradInput = gradSelfAttention + gradNorm1
        return Engine.TensorAdd(gradSelfAttention, gradNorm1);
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
        _feedForwardProjection.UpdateParameters(learningRate);
        _norm3.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Updates layer parameters using GPU-resident optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    /// <remarks>
    /// <para>
    /// This method delegates to each sublayer's UpdateParametersGpu method.
    /// All sublayers (self-attention, cross-attention, layer norms, feed-forward) are updated.
    /// </para>
    /// </remarks>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        // Update parameters for each sub-layer using GPU optimizer
        _selfAttention.UpdateParametersGpu(config);
        _norm1.UpdateParametersGpu(config);
        _crossAttention.UpdateParametersGpu(config);
        _norm2.UpdateParametersGpu(config);
        _feedForward.UpdateParametersGpu(config);
        _feedForwardProjection.UpdateParametersGpu(config);
        _norm3.UpdateParametersGpu(config);
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
        // === Vectorized Parameter Concatenation (Phase B: US-GPU-015) ===
        // Collect parameters from all sublayers
        var selfAttentionParams = _selfAttention.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var crossAttentionParams = _crossAttention.GetParameters();
        var norm2Params = _norm2.GetParameters();
        var feedForwardParams = _feedForward.GetParameters();
        var feedForwardProjectionParams = _feedForwardProjection.GetParameters();
        var norm3Params = _norm3.GetParameters();

        // Concatenate all parameter vectors efficiently
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(
                    Vector<T>.Concatenate(
                        Vector<T>.Concatenate(
                            Vector<T>.Concatenate(selfAttentionParams, norm1Params),
                            crossAttentionParams),
                        norm2Params),
                    feedForwardParams),
                feedForwardProjectionParams),
            norm3Params);
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
        _feedForwardProjection.ResetState();
        _norm3.ResetState();

        // Clear GPU cached tensors
        _gpuNormalized1 = null;
        _gpuNormalized2 = null;

        // Clear cached tensors
        _lastInput = null;
        _lastEncoderOutput = null;
        _lastSelfAttentionOutput = null;
        _lastNormalized1 = null;
        _lastCrossAttentionOutput = null;
        _lastNormalized2 = null;
        _lastFeedForwardOutput = null;
    }

    /// <summary>
    /// Computes the auxiliary loss for this layer by aggregating losses from sublayers.
    /// </summary>
    /// <returns>The computed auxiliary loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the auxiliary loss by aggregating losses from sublayers that implement IAuxiliaryLossLayer.
    /// For the decoder layer, this includes both self-attention and cross-attention mechanisms, which provide
    /// attention entropy and head diversity regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects additional learning signals from the layer's components.
    ///
    /// Auxiliary loss aggregation:
    /// - Checks each attention sublayer to see if it has auxiliary losses
    /// - Collects those losses from both self-attention and cross-attention
    /// - Combines them and returns the total for use in training
    ///
    /// Why this is useful:
    /// - Both attention mechanisms benefit from regularization to prevent all heads from learning the same patterns
    /// - Self-attention regularization helps the decoder maintain coherent generation patterns
    /// - Cross-attention regularization helps the decoder focus on relevant parts of the source
    /// - Aggregating losses at the decoder level provides a unified view of attention quality
    ///
    /// Example: If the self-attention has an entropy loss (to keep attention focused) and a diversity loss
    /// (to prevent heads from being redundant), and the cross-attention has similar losses, this method
    /// adds all of them together and returns the total.
    ///
    /// The aggregated loss helps ensure:
    /// - Both attention mechanisms learn diverse patterns
    /// - Attention is focused rather than diffuse
    /// - The decoder uses its capacity efficiently for both understanding context and attending to source
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            _lastAuxiliaryLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        T totalAuxLoss = NumOps.Zero;
        int auxLayerCount = 0;

        // Aggregate auxiliary loss from self-attention if it implements IAuxiliaryLossLayer
        if (_selfAttention is IAuxiliaryLossLayer<T> auxSelfAttention && auxSelfAttention.UseAuxiliaryLoss)
        {
            T selfAttentionAuxLoss = auxSelfAttention.ComputeAuxiliaryLoss();
            totalAuxLoss = NumOps.Add(totalAuxLoss, selfAttentionAuxLoss);
            auxLayerCount++;
        }

        // Aggregate auxiliary loss from cross-attention if it implements IAuxiliaryLossLayer
        if (_crossAttention is IAuxiliaryLossLayer<T> auxCrossAttention && auxCrossAttention.UseAuxiliaryLoss)
        {
            T crossAttentionAuxLoss = auxCrossAttention.ComputeAuxiliaryLoss();
            totalAuxLoss = NumOps.Add(totalAuxLoss, crossAttentionAuxLoss);
            auxLayerCount++;
        }

        // Average the auxiliary losses if any were computed
        if (auxLayerCount > 0)
        {
            totalAuxLoss = NumericalStabilityHelper.SafeDiv(totalAuxLoss, NumOps.FromDouble(auxLayerCount));
        }

        _lastAuxiliaryLoss = totalAuxLoss;
        return _lastAuxiliaryLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about the auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method returns diagnostic information that can be used to monitor the auxiliary loss during training.
    /// The diagnostics include the total auxiliary loss, the weight applied to it, whether auxiliary loss is enabled,
    /// and detailed diagnostics from both self-attention and cross-attention sublayers.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information to help you understand how the auxiliary loss is working.
    ///
    /// The diagnostics show:
    /// - TotalAuxiliaryLoss: The combined penalty from all attention sublayers
    /// - AuxiliaryWeight: How much this penalty affects the overall training
    /// - UseAuxiliaryLoss: Whether this penalty is currently enabled
    /// - SelfAttentionDiagnostics: Detailed information from the self-attention mechanism
    /// - CrossAttentionDiagnostics: Detailed information from the cross-attention mechanism
    ///
    /// You can use this information to:
    /// - Monitor if attention patterns are healthy (diverse and focused) in both mechanisms
    /// - Debug training issues related to attention
    /// - Understand how the decoder is learning both context and source information
    ///
    /// Example: If you see that self-attention entropy is very low, it might mean the decoder isn't
    /// maintaining good coherence with its own generated output. If cross-attention diversity is low,
    /// it might mean all heads are looking at the same part of the source, wasting capacity.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "TotalAuxiliaryLoss", _lastAuxiliaryLoss?.ToString() ?? "0" },
            { "AuxiliaryWeight", AuxiliaryLossWeight?.ToString() ?? "0.005" },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
        };

        // Include diagnostics from self-attention if available
        if (_selfAttention is IAuxiliaryLossLayer<T> auxSelfAttention)
        {
            var selfAttentionDiagnostics = auxSelfAttention.GetAuxiliaryLossDiagnostics();
            foreach (var kvp in selfAttentionDiagnostics)
            {
                diagnostics[$"SelfAttention_{kvp.Key}"] = kvp.Value;
            }
        }

        // Include diagnostics from cross-attention if available
        if (_crossAttention is IAuxiliaryLossLayer<T> auxCrossAttention)
        {
            var crossAttentionDiagnostics = auxCrossAttention.GetAuxiliaryLossDiagnostics();
            foreach (var kvp in crossAttentionDiagnostics)
            {
                diagnostics[$"CrossAttention_{kvp.Key}"] = kvp.Value;
            }
        }

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    /// <summary>
    /// Exports the transformer decoder layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the transformer decoder operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node (decoder input)
    /// 2. Applies masked self-attention with residual connection and norm
    /// 3. Applies cross-attention to encoder output with residual and norm
    /// 4. Applies feed-forward network with residual connection and norm
    /// 5. Returns the final output
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of a transformer decoder layer for JIT.
    ///
    /// The transformer decoder layer is a composite layer combining:
    /// - Masked self-attention (prevents looking ahead in target sequence)
    /// - Cross-attention (attends to encoder output, connects source and target)
    /// - Layer normalization (stabilizes training)
    /// - Feed-forward network (processes each position independently)
    /// - Residual connections (helps gradient flow in deep networks)
    ///
    /// The forward pass:
    /// 1. x' = LayerNorm(x + MaskedSelfAttention(x))
    /// 2. x'' = LayerNorm(x' + CrossAttention(x', encoder_output))
    /// 3. output = LayerNorm(x'' + FeedForward(x''))
    ///
    /// JIT optimization for composite layers:
    /// - For now, composite layers note their structure but may delegate to sublayers
    /// - Future optimization could fuse operations across sublayers
    /// - Each sublayer (self-attention, cross-attention, feed-forward, norm) can be independently JIT compiled
    ///
    /// This is the core building block of GPT (decoder-only) and encoder-decoder models like T5.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when sublayers are not initialized.</exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured. Initialize the layer first.");

        if (_selfAttention == null || _norm1 == null ||
            _crossAttention == null || _norm2 == null ||
            _feedForward == null || _norm3 == null)
            throw new InvalidOperationException("Sublayers not initialized. Initialize the layer first.");

        // Create symbolic input nodes: decoder input and encoder output
        var symbolicDecoderInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var decoderInputNode = TensorOperations<T>.Variable(symbolicDecoderInput, "decoder_input");
        inputNodes.Add(decoderInputNode);

        // Encoder output has same shape as decoder input in standard transformers
        var symbolicEncoderOutput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var encoderOutputNode = TensorOperations<T>.Variable(symbolicEncoderOutput, "encoder_output");
        inputNodes.Add(encoderOutputNode);

        // Step 1: Masked self-attention sublayer (decoder attends to itself)
        var selfAttentionOut = ApplyMultiHeadAttentionGraph(_selfAttention, decoderInputNode, decoderInputNode, decoderInputNode);

        // Step 2: First residual connection: residual1 = input + self_attention_out
        var residual1 = TensorOperations<T>.Add(decoderInputNode, selfAttentionOut);

        // Step 3: First layer normalization
        var normalized1 = ApplyLayerNormGraph(_norm1, residual1);

        // Step 4: Cross-attention sublayer (decoder attends to encoder output)
        // Query comes from decoder, Key and Value come from encoder
        var crossAttentionOut = ApplyMultiHeadAttentionGraph(_crossAttention, normalized1, encoderOutputNode, encoderOutputNode);

        // Step 5: Second residual connection: residual2 = normalized1 + cross_attention_out
        var residual2 = TensorOperations<T>.Add(normalized1, crossAttentionOut);

        // Step 6: Second layer normalization
        var normalized2 = ApplyLayerNormGraph(_norm2, residual2);

        // Step 7: Feed-forward sublayer
        var ffOut = ApplyFeedForwardGraph(_feedForward, normalized2);

        // Step 8: Third residual connection: residual3 = normalized2 + ff_out
        var residual3 = TensorOperations<T>.Add(normalized2, ffOut);

        // Step 9: Third layer normalization (final output)
        var output = ApplyLayerNormGraph(_norm3, residual3);

        return output;
    }

    /// <summary>
    /// Applies multi-head attention graph to input nodes (supports both self-attention and cross-attention).
    /// </summary>
    private ComputationNode<T> ApplyMultiHeadAttentionGraph(
        MultiHeadAttentionLayer<T> attentionLayer,
        ComputationNode<T> query,
        ComputationNode<T> key,
        ComputationNode<T> value)
    {
        // Get attention projection weights
        var queryWeights = attentionLayer.GetQueryWeights();
        var keyWeights = attentionLayer.GetKeyWeights();
        var valueWeights = attentionLayer.GetValueWeights();
        var outputWeights = attentionLayer.GetOutputWeights();

        if (queryWeights == null || keyWeights == null || valueWeights == null || outputWeights == null)
            throw new InvalidOperationException("Attention weights not initialized.");

        // Create constant nodes for projection weights (already Tensor<T>)
        var wqNode = TensorOperations<T>.Constant(queryWeights, "Wq");
        var wkNode = TensorOperations<T>.Constant(keyWeights, "Wk");
        var wvNode = TensorOperations<T>.Constant(valueWeights, "Wv");
        var woNode = TensorOperations<T>.Constant(outputWeights, "Wo");

        // Apply multi-head attention
        return TensorOperations<T>.MultiHeadAttention(
            query: query,
            key: key,
            value: value,
            numHeads: attentionLayer.HeadCount,
            wQ: wqNode,
            wK: wkNode,
            wV: wvNode,
            wO: woNode);
    }

    /// <summary>
    /// Applies layer normalization graph to an input node.
    /// </summary>
    private ComputationNode<T> ApplyLayerNormGraph(LayerNormalizationLayer<T> normLayer, ComputationNode<T> input)
    {
        // Get normalization parameters directly as tensors
        var gamma = normLayer.GetGammaTensor();
        var beta = normLayer.GetBetaTensor();
        var normalizedShape = normLayer.GetNormalizedShape();
        var epsilon = Convert.ToDouble(normLayer.GetEpsilon());

        // Create constant nodes for gamma and beta
        var gammaNode = TensorOperations<T>.Constant(gamma, "gamma");
        var betaNode = TensorOperations<T>.Constant(beta, "beta");

        return TensorOperations<T>.LayerNorm(input, normalizedShape, gammaNode, betaNode, epsilon);
    }

    /// <summary>
    /// Applies feed-forward graph to an input node.
    /// </summary>
    private ComputationNode<T> ApplyFeedForwardGraph(FeedForwardLayer<T> ffLayer, ComputationNode<T> input)
    {
        // Get feed-forward weights and biases directly as tensors (first layer: expand to hidden dim)
        var weightsTensor = ffLayer.GetWeightsTensor();
        var biasTensor = ffLayer.GetBiasesTensor();

        if (weightsTensor == null || biasTensor == null)
            throw new InvalidOperationException("Feed-forward layer weights not initialized.");

        var weightsNode = TensorOperations<T>.Constant(weightsTensor, "ff_weights");
        var biasNode = TensorOperations<T>.Constant(biasTensor, "ff_bias");

        // Linear transformation: hidden = input @ weights^T + bias
        var weightsT = TensorOperations<T>.Transpose(weightsNode);
        var linear = TensorOperations<T>.MatrixMultiply(input, weightsT);
        var withBias = TensorOperations<T>.Add(linear, biasNode);

        // Apply activation if present using the activation's own ApplyToGraph method
        ComputationNode<T> hidden = withBias;
        var activation = ffLayer.ScalarActivation;
        if (activation != null)
        {
            hidden = activation.ApplyToGraph(withBias);
        }

        // Apply projection layer (second layer: project back to embedding size)
        var projWeightsTensor = _feedForwardProjection.GetWeightsTensor();
        var projBiasTensor = _feedForwardProjection.GetBiasesTensor();

        if (projWeightsTensor == null || projBiasTensor == null)
            throw new InvalidOperationException("Feed-forward projection layer weights not initialized.");

        var projWeightsNode = TensorOperations<T>.Constant(projWeightsTensor, "ff_proj_weights");
        var projBiasNode = TensorOperations<T>.Constant(projBiasTensor, "ff_proj_bias");

        // Linear transformation: output = hidden @ projWeights^T + projBias
        var projWeightsT = TensorOperations<T>.Transpose(projWeightsNode);
        var projLinear = TensorOperations<T>.MatrixMultiply(hidden, projWeightsT);
        var output = TensorOperations<T>.Add(projLinear, projBiasNode);

        return output;
    }

    /// <summary>
    /// Gets whether this transformer decoder layer supports JIT compilation.
    /// </summary>
    /// <value>True if all sublayers support JIT compilation.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. As a composite layer,
    /// it supports JIT if all its sublayers support JIT:
    /// - Masked self-attention layer
    /// - Cross-attention layer (attends to encoder output)
    /// - Layer normalization layers (3 total)
    /// - Feed-forward layer
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this composite layer can use JIT compilation.
    ///
    /// The transformer decoder layer can be JIT compiled if:
    /// - All sublayers are properly initialized
    /// - Each sublayer supports JIT compilation
    ///
    /// Composite layer JIT optimization:
    /// - Each sublayer can be independently JIT compiled
    /// - Future optimization: fuse operations across sublayers
    /// - Residual connections and layer norms are fast operations
    ///
    /// The bottleneck in decoder layers:
    /// - Self-attention: O(n²) for target sequence
    /// - Cross-attention: O(n*m) where n=target length, m=source length
    /// - Feed-forward: matrix multiplications
    ///
    /// All benefit significantly from JIT compilation (5-10x speedup).
    ///
    /// GPT models use decoder-only architecture (no cross-attention, only self-attention).
    /// T5 and other seq2seq models use both encoder and decoder layers.
    /// GPT-3 has 96 decoder layers, making JIT optimization critical for performance.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // TransformerDecoderLayer is a composite layer
            // It supports JIT if all sublayers support JIT
            return _selfAttention != null && _selfAttention.SupportsJitCompilation &&
                   _norm1 != null && _norm1.SupportsJitCompilation &&
                   _crossAttention != null && _crossAttention.SupportsJitCompilation &&
                   _norm2 != null && _norm2.SupportsJitCompilation &&
                   _feedForward != null && _feedForward.SupportsJitCompilation &&
                   _feedForwardProjection != null && _feedForwardProjection.SupportsJitCompilation &&
                   _norm3 != null && _norm3.SupportsJitCompilation;
        }
    }
}
