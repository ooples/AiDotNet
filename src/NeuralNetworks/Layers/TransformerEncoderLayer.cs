using AiDotNet.NeuralNetworks.Layers.PositionalEncoding;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a transformer encoder layer that processes sequences using self-attention and feed-forward networks.
/// </summary>
/// <remarks>
/// <para>
/// A transformer encoder layer is a fundamental building block of transformer-based models for sequence processing tasks.
/// It consists of two main components: a self-attention mechanism that allows each position in a sequence to attend to all
/// positions, and a feed-forward network that processes each position independently. Each component is followed by layer
/// normalization and residual connections to facilitate training of deep networks.
/// </para>
/// <para><b>For Beginners:</b> This layer helps a neural network understand relationships between different elements in a sequence.
/// 
/// Think of it like a careful reader analyzing a paragraph:
/// - First, the reader looks at how each word relates to every other word (self-attention)
/// - Then, the reader processes this information to understand the meaning (feed-forward network)
/// 
/// For example, in the sentence "The animal didn't cross the street because it was too wide":
/// - The self-attention helps the network understand that "it" refers to "the street" (not "the animal")
/// - The feed-forward network processes this contextual information for each word
/// 
/// This architecture is powerful for tasks like understanding text, analyzing time series, or processing any data
/// where the relationships between elements matter.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TransformerEncoderLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The size of the embeddings for queries, keys, values, and outputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the dimensionality of the embedding vectors used throughout the transformer encoder layer.
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
    /// The number of attention heads for the self-attention mechanism.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of attention heads used in the multi-head self-attention layer. Multiple heads allow
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
    /// - Takes the output from the attention layer
    /// - Expands it to this larger dimension for more complex processing
    /// - Then compresses it back to the original embedding size
    /// 
    /// It's typically 4 times larger than the embedding size (e.g., 2048 for a 512 embedding size).
    /// This expanded dimension gives the network more capacity to transform the information.
    /// </para>
    /// </remarks>
    private readonly int _feedForwardDim;

    /// <summary>
    /// The self-attention mechanism for processing the input sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the multi-head self-attention layer that allows each position in the input sequence to attend
    /// to all positions, capturing the relationships between different elements in the sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This component helps the model understand how different parts of the sequence relate to each other.
    /// 
    /// Self-attention works by:
    /// - Looking at the entire sequence
    /// - For each position, determining which other positions are most relevant
    /// - Creating a weighted combination of all positions for the output
    /// 
    /// For example, in a sentence, this helps the model understand that "it" in "The dog chased the ball because it was round"
    /// refers to "the ball" rather than "the dog" by attending strongly to the word "ball".
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
    /// The feed-forward network for additional transformation of the sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains the feed-forward network that applies a non-linear transformation to each position in the sequence independently.
    /// It consists of two linear transformations with an activation function (typically GELU) in between.
    /// </para>
    /// <para><b>For Beginners:</b> This component processes the attended information to produce the final output for each position.
    /// 
    /// The feed-forward network:
    /// - Processes each position independently (unlike attention, which looks across positions)
    /// - Applies a more complex transformation with non-linearity
    /// - Helps the network learn more abstract patterns and relationships
    /// 
    /// This is where the model does its "thinking" after gathering information through self-attention,
    /// making final decisions about what features to extract from each position.
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
    /// <para><b>For Beginners:</b> This component helps keep the values in a reasonable range after feed-forward processing.
    /// 
    /// Like the first normalization layer, this:
    /// - Standardizes the output of the feed-forward component
    /// - Prevents extreme values that could disrupt training
    /// - Helps the network converge faster during training
    /// 
    /// This final normalization helps prepare the output for either the next encoder layer
    /// or for use by a decoder in the complete transformer model.
    /// </para>
    /// </remarks>
    private LayerNormalizationLayer<T> _norm2 = default!;

    /// <summary>
    /// The dropout rate applied to attention and feed-forward layers.
    /// </summary>
    private readonly double _dropoutRate;

    /// <summary>
    /// The number of encoder layers to stack.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// The positional encoding applied to input embeddings.
    /// </summary>
    private readonly IPositionalEncoding<T> _positionalEncoding = default!;

    /// <summary>
    /// Additional encoder layers when numLayers > 1.
    /// </summary>
    private List<TransformerEncoderLayer<T>>? _additionalLayers;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, as it contains trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the transformer encoder layer can be trained through backpropagation.
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
    /// Initializes a new instance of the <see cref="TransformerEncoderLayer{T}"/> class with basic configuration.
    /// </summary>
    /// <param name="embeddingSize">The size of the embeddings.</param>
    /// <param name="numHeads">The number of attention heads.</param>
    /// <param name="feedForwardDim">The dimension of the feed-forward network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a basic transformer encoder layer with the specified dimensions. It automatically
    /// uses sensible defaults: no dropout (0.0), single layer, and sinusoidal positional encoding with max sequence length of 8192.
    /// </para>
    /// <para><b>For Beginners:</b> This is the simplest way to create a transformer encoder layer.
    /// 
    /// The parameters you provide determine:
    /// - embeddingSize: How rich the representation of each token is (more = more expressive)
    /// - numHeads: How many different "perspectives" the attention mechanism can have
    /// - feedForwardDim: How much processing capacity the feed-forward network has
    /// 
    /// This constructor automatically includes:
    /// - No dropout (good for inference or when you don't want regularization)
    /// - Classic sinusoidal positional encoding (proven and widely used)
    /// - Single encoder layer (not stacked)
    /// 
    /// Typical values might be 512 for embedding size, 8 attention heads, and 2048 for the feed-forward dimension,
    /// similar to those used in the original transformer paper.
    /// </para>
    /// </remarks>
    public TransformerEncoderLayer(int embeddingSize, int numHeads, int feedForwardDim)
        : base([embeddingSize], [embeddingSize])
    {
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;

        // Initialize new fields with default values
        _dropoutRate = 0.0; // No dropout for this basic constructor
        _numLayers = 1; // Single layer
        _positionalEncoding = new SinusoidalPositionalEncoding<T>(8192, embeddingSize); // Default sinusoidal encoding

        int sequenceLength = 1; // Default to 1
        _selfAttention = new MultiHeadAttentionLayer<T>(
            sequenceLength, 
            _embeddingSize, 
            _numHeads, 
            new GELUActivation<T>() as IActivationFunction<T>);
            
        _norm1 = new LayerNormalizationLayer<T>(_embeddingSize);
        
        _feedForward = new FeedForwardLayer<T>(
            _embeddingSize, 
            _feedForwardDim, 
            new GELUActivation<T>() as IActivationFunction<T>);
            
        _norm2 = new LayerNormalizationLayer<T>(_embeddingSize);
        
        // No additional layers for this basic constructor
        _additionalLayers = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerEncoderLayer{T}"/> class with dropout and optional positional encoding.
    /// </summary>
    /// <param name="embeddingSize">The size of the embeddings.</param>
    /// <param name="numHeads">The number of attention heads.</param>
    /// <param name="feedForwardDim">The dimension of the feed-forward network.</param>
    /// <param name="dropoutRate">The dropout rate to apply after attention and feed-forward layers.</param>
    /// <param name="positionalEncoding">The positional encoding to apply to the input embeddings. If null, uses SinusoidalPositionalEncoding with a default max sequence length of 8192.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a single transformer encoder layer with the specified dimensions and dropout rate.
    /// It's a simpler version for users who don't need to stack multiple layers.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simplified constructor for creating a single transformer encoder layer.
    /// 
    /// Use this constructor when you want:
    /// - A single transformer encoder layer (not stacked)
    /// - Dropout regularization for better training
    /// - Optional custom positional encoding (or automatic default)
    /// 
    /// If you don't specify a positional encoding, the constructor will use the classic
    /// sinusoidal positional encoding which works well for most tasks.
    /// </para>
    /// </remarks>
    public TransformerEncoderLayer(
        int embeddingSize,
        int numHeads,
        int feedForwardDim,
        double dropoutRate,
        IPositionalEncoding<T>? positionalEncoding = null)
        : this(embeddingSize, numHeads, feedForwardDim, dropoutRate, 1, positionalEncoding)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerEncoderLayer{T}"/> class with additional parameters.
    /// </summary>
    /// <param name="embeddingSize">The size of the embeddings.</param>
    /// <param name="numHeads">The number of attention heads.</param>
    /// <param name="feedForwardDim">The dimension of the feed-forward network.</param>
    /// <param name="dropoutRate">The dropout rate to apply after attention and feed-forward layers.</param>
    /// <param name="numLayers">The number of transformer encoder layers to stack.</param>
    /// <param name="positionalEncoding">The positional encoding to apply to the input embeddings. If null, uses SinusoidalPositionalEncoding with a default max sequence length of 8192.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a transformer encoder layer with the specified dimensions and additional parameters
    /// for dropout, layer stacking, and positional encoding. It initializes the self-attention, layer normalization,
    /// and feed-forward sublayers with appropriate dimensions and activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a more configurable transformer encoder layer.
    /// 
    /// The additional parameters provide more control:
    /// - dropoutRate: Controls regularization by randomly dropping some values during training
    /// - numLayers: Determines how many identical encoder layers to stack for deeper processing
    /// - positionalEncoding: Adds information about position in the sequence to each element
    /// 
    /// If you don't specify a positional encoding, the constructor will automatically use the classic
    /// sinusoidal positional encoding from the original Transformer paper, which works well for most tasks.
    /// 
    /// These settings allow for more sophisticated transformer architectures that can handle
    /// more complex tasks and potentially achieve better performance.
    /// </para>
    /// </remarks>
    public TransformerEncoderLayer(
        int embeddingSize,
        int numHeads,
        int feedForwardDim,
        double dropoutRate,
        int numLayers,
        IPositionalEncoding<T>? positionalEncoding = null)
        : base([embeddingSize], [embeddingSize])
    {
        _embeddingSize = embeddingSize;
        _numHeads = numHeads;
        _feedForwardDim = feedForwardDim;

        // Store additional parameters as fields
        _dropoutRate = dropoutRate;
        _numLayers = numLayers;
        
        // Use default sinusoidal positional encoding if none provided
        _positionalEncoding = positionalEncoding ?? new SinusoidalPositionalEncoding<T>(8192, embeddingSize);

        // Default sequence length (can be adjusted based on input)
        int sequenceLength = 1024;

        // Create the self-attention layer with dropout
        _selfAttention = new MultiHeadAttentionLayer<T>(
            sequenceLength,
            _embeddingSize,
            _numHeads,
            new GELUActivation<T>() as IActivationFunction<T>,
            dropoutRate);

        _norm1 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Create the feed-forward layer with dropout
        _feedForward = new FeedForwardLayer<T>(
            _embeddingSize,
            _feedForwardDim,
            new GELUActivation<T>() as IActivationFunction<T>,
            dropoutRate);

        _norm2 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Initialize additional layers if numLayers > 1
        if (numLayers > 1)
        {
            _additionalLayers = new List<TransformerEncoderLayer<T>>(numLayers - 1);
            for (int i = 0; i < numLayers - 1; i++)
            {
                _additionalLayers.Add(new TransformerEncoderLayer<T>(
                    embeddingSize,
                    numHeads,
                    feedForwardDim));
            }
        }
    }

    /// <summary>
    /// Performs the forward pass of the transformer encoder layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through the transformer encoder layer.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the transformer encoder layer. It processes the input through
    /// the self-attention mechanism, applies layer normalization and a residual connection, then passes the result
    /// through the feed-forward network followed by another layer normalization and residual connection.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input sequence through all components of the encoder layer.
    /// 
    /// The forward pass follows these steps:
    /// 
    /// 1. Self-Attention:
    ///    - The encoder looks at how each element in the sequence relates to all other elements
    ///    - This creates contextual representations that capture relationships in the data
    /// 
    /// 2. Add & Normalize:
    ///    - The original input is added to the attention output (residual connection)
    ///    - Layer normalization stabilizes the values
    /// 
    /// 3. Feed-Forward Network:
    ///    - Each position is processed independently through the feed-forward network
    ///    - This transforms the representations further
    /// 
    /// 4. Final Add & Normalize:
    ///    - The output from step 2 is added to the feed-forward output (another residual connection)
    ///    - Final layer normalization is applied
    /// 
    /// These steps allow the encoder to create rich representations of the input sequence
    /// that capture both the content of each element and its relationships to other elements.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Apply positional encoding if available
        var encodedInput = input;
        if (_positionalEncoding != null)
        {
            encodedInput = _positionalEncoding.AddPositionalEncoding(input);
        }

        // Process through the first layer (this instance)
        var attention = _selfAttention.Forward(encodedInput);
        var normalized1 = _norm1.Forward(encodedInput + attention);
        var feedForward = _feedForward.Forward(normalized1);
        var output = _norm2.Forward(normalized1 + feedForward);

        // Process through additional layers if any
        if (_additionalLayers != null && _additionalLayers.Count > 0)
        {
            foreach (var layer in _additionalLayers)
            {
                output = layer.Forward(output);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the transformer encoder layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the transformer encoder layer, which is used during training to
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
    /// 3. First Layer Normalization:
    ///    - Compute how the first normalization's input should change
    /// 
    /// 4. Self-Attention:
    ///    - Determine how the self-attention's input should change
    ///    - Account for the residual connection
    /// 
    /// This reverse flow of gradients allows each component to learn how it contributed to any errors.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward pass through the second normalization layer
        var dNorm2 = _norm2.Backward(outputGradient);
    
        // Split the gradient for the residual connection
        var dFeedForward = dNorm2;
        var dNormalized1 = dNorm2;

        // Backward pass through the feed-forward layer
        var dFeedForwardInput = _feedForward.Backward(dFeedForward);
        dNormalized1 += dFeedForwardInput;

        // Backward pass through the first normalization layer
        var dNorm1 = _norm1.Backward(dNormalized1);

        // Split the gradient for the residual connection
        var dAttention = dNorm1;
        var dInput = dNorm1;

        // Backward pass through the self-attention layer
        var dSelfAttentionInput = _selfAttention.Backward(dAttention);
        dInput += dSelfAttentionInput;

        return dInput;
    }

    /// <summary>
    /// Updates the parameters of all sublayers using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of all sublayers in the transformer encoder layer based on the gradients
    /// calculated during the backward pass. It delegates the update process to each sublayer, passing the learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts all the internal values of the layer to improve its performance.
    /// 
    /// During parameter updates:
    /// - The learning rate controls how big each adjustment is
    /// - Every sublayer gets updated based on what was learned in the backward pass
    /// - This helps the entire encoder layer gradually improve its performance
    /// 
    /// Think of it like fine-tuning all the components of the encoder based on feedback:
    /// - The self-attention mechanism learns to focus on more relevant relationships
    /// - The feed-forward network learns to better transform the information
    /// - The normalization layers learn to keep values in the optimal range
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update parameters for each sub-layer
        _selfAttention.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _feedForward.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters of the transformer encoder layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters from all sublayers.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from all sublayers of the transformer encoder layer and combines
    /// them into a single vector. This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from all parts of the encoder.
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
    /// A transformer encoder layer typically has millions of parameters, all of which
    /// contribute to its ability to understand complex sequences.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Collect parameters from all sublayers
        var selfAttentionParams = _selfAttention.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var feedForwardParams = _feedForward.GetParameters();
        var norm2Params = _norm2.GetParameters();
    
        // Calculate total parameter count
        int totalParamCount = selfAttentionParams.Length + 
                              norm1Params.Length + 
                              feedForwardParams.Length + 
                              norm2Params.Length;
    
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
    
        // Copy feed-forward parameters
        for (int i = 0; i < feedForwardParams.Length; i++)
            parameters[currentIndex++] = feedForwardParams[i];
    
        // Copy norm2 parameters
        for (int i = 0; i < norm2Params.Length; i++)
            parameters[currentIndex++] = norm2Params[i];
    
        return parameters;
    }

    /// <summary>
    /// Resets the internal state of the transformer encoder layer and all its sublayers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the transformer encoder layer and all its sublayers. It delegates
    /// the reset operation to each sublayer, ensuring that any cached state is cleared.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - All sublayers are reset to their initial condition
    /// - Any cached information from previous processing is cleared
    /// - The layer is ready to process new, unrelated sequences
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Starting a new training episode
    /// - Testing the layer with fresh inputs
    /// 
    /// Think of it like clearing your mind before starting a completely new task,
    /// ensuring no information from previous tasks affects your current thinking.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Reset all sublayers
        _selfAttention.ResetState();
        _norm1.ResetState();
        _feedForward.ResetState();
        _norm2.ResetState();
    }
}