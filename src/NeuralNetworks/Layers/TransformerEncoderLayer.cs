using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

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
public class TransformerEncoderLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets a value indicating whether auxiliary loss is enabled for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the layer aggregates auxiliary losses from its sublayers, particularly the self-attention mechanism.
    /// This helps regularize attention patterns and prevent issues like attention collapse.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the layer uses additional learning signals.
    ///
    /// When enabled (true):
    /// - The layer collects extra penalties from the self-attention mechanism
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
    /// Tracks whether the last input was originally 2D (and thus reshaped to 3D).
    /// </summary>
    private bool _inputWas2D = false;

    /// <summary>
    /// Stores the original input shape for restoring higher-rank tensor output.
    /// </summary>
    private int[]? _originalInputShape;

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
    private readonly FeedForwardLayer<T> _feedForward1;

    /// <summary>
    /// The second (projection) layer of the feed-forward network.
    /// </summary>
    /// <remarks>
    /// Projects the expanded representation back to the original embedding size.
    /// A proper transformer FFN has two layers: expansion and projection.
    /// </remarks>
    private readonly FeedForwardLayer<T> _feedForward2;

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
    private LayerNormalizationLayer<T> _norm2;

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

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
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <remarks>
    /// This returns the sum of all parameters from sublayers: self-attention, layer norms, and feed-forward layers.
    /// </remarks>
    public override int ParameterCount =>
        _selfAttention.ParameterCount +
        _norm1.ParameterCount +
        _feedForward1.ParameterCount +
        _feedForward2.ParameterCount +
        _norm2.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerEncoderLayer{T}"/> class.
    /// </summary>
    /// <param name="embeddingSize">The size of the embeddings.</param>
    /// <param name="numHeads">The number of attention heads.</param>
    /// <param name="feedForwardDim">The dimension of the feed-forward network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a transformer encoder layer with the specified dimensions. It initializes the
    /// self-attention, layer normalization, and feed-forward sublayers with appropriate dimensions and activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new transformer encoder layer with the specified settings.
    /// 
    /// The parameters you provide determine:
    /// - embeddingSize: How rich the representation of each token is (more = more expressive)
    /// - numHeads: How many different "perspectives" the attention mechanism can have
    /// - feedForwardDim: How much processing capacity the feed-forward network has
    /// 
    /// These settings control the capacity, expressiveness, and computational requirements of the encoder.
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

        int sequenceLength = 1; // Default to 1
        _selfAttention = new MultiHeadAttentionLayer<T>(
            sequenceLength,
            _embeddingSize,
            _numHeads,
            new GELUActivation<T>() as IActivationFunction<T>);

        _norm1 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Standard transformer FFN: Linear(embed -> ff) + GELU + Linear(ff -> embed)
        _feedForward1 = new FeedForwardLayer<T>(
            _embeddingSize,
            _feedForwardDim,
            new GELUActivation<T>() as IActivationFunction<T>);

        _feedForward2 = new FeedForwardLayer<T>(
            _feedForwardDim,
            _embeddingSize,
            (IActivationFunction<T>?)null); // No activation on projection layer

        _norm2 = new LayerNormalizationLayer<T>(_embeddingSize);

        // Initialize NumOps-based fields
        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastAuxiliaryLoss = NumOps.Zero;
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
        // Handle any rank >= 2: last 2 dims are [seq, embed], earlier dims are batch-like
        int rank = input.Shape.Length;
        _inputWas2D = rank == 2;
        _originalInputShape = input.Shape;

        Tensor<T> input3D;
        if (_inputWas2D)
        {
            input3D = input.Reshape(1, input.Shape[0], input.Shape[1]);
        }
        else if (rank == 3)
        {
            input3D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            input3D = input.Reshape(flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]);
        }

        var attention = _selfAttention.Forward(input3D);
        // Residual connection: input + attention
        var residual1 = Engine.TensorAdd(input3D, attention);

        // Layer norm now supports any-rank tensors (normalizes over last dimension)
        var normalized1 = _norm1.Forward(residual1);

        // Standard transformer FFN: Linear(embed -> ff) + GELU + Linear(ff -> embed)
        var ffExpanded = _feedForward1.Forward(normalized1);
        var ffProjected = _feedForward2.Forward(ffExpanded);

        // Residual connection: normalized1 + ffProjected
        var residual2 = Engine.TensorAdd(normalized1, ffProjected);
        var output = _norm2.Forward(residual2);

        // Restore original tensor shape
        if (_inputWas2D)
        {
            output = output.Reshape(output.Shape[1], output.Shape[2]);
        }
        else if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 2; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 2] = output.Shape[1];
            outputShape[_originalInputShape.Length - 1] = output.Shape[2];
            output = output.Reshape(outputShape);
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU-resident output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the entire transformer encoder forward pass on the GPU without downloading
    /// intermediate results to CPU. All sublayer operations (self-attention, layer normalization,
    /// feed-forward networks, residual connections) remain GPU-resident for maximum performance.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];

        // Get dimensions from input shape
        int[] inputShape = input.Shape;
        int rank = inputShape.Length;

        IGpuTensor<T> input3D;
        int[] originalShape = inputShape;
        bool was2D = rank == 2;

        // Cache shape info for backward pass consistency with CPU Forward
        // (even though backward uses gradient shape, this maintains API consistency)
        if (IsTrainingMode)
        {
            _inputWas2D = was2D;
            _originalInputShape = inputShape;
        }

        if (was2D)
        {
            // 2D: [seqLen, embedDim] -> add batch dim
            input3D = gpuEngine.ReshapeGpu(input, [1, inputShape[0], inputShape[1]]);
        }
        else if (rank == 3)
        {
            // Standard 3D: [batch, seqLen, embedDim]
            input3D = input;
        }
        else if (rank > 3)
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= inputShape[d];
            input3D = gpuEngine.ReshapeGpu(input, [flatBatch, inputShape[rank - 2], inputShape[rank - 1]]);
        }
        else
        {
            throw new ArgumentException($"TransformerEncoderLayer requires at least 2D input, got {rank}D");
        }

        // 1. Self-attention sublayer
        var attention = _selfAttention.ForwardGpu(input3D);

        // 2. First residual connection: input + attention
        var residual1 = gpuEngine.AddGpu(input3D, attention);

        // 3. First layer normalization
        var (normalized1, _, _) = gpuEngine.LayerNormGpu(residual1, _norm1.GetGammaTensor(), _norm1.GetBetaTensor(), Convert.ToDouble(_norm1.GetEpsilon()));

        // 4. Feed-forward network (two layers)
        var ffExpanded = _feedForward1.ForwardGpu(normalized1);
        var ffProjected = _feedForward2.ForwardGpu(ffExpanded);

        // 5. Second residual connection: normalized1 + ffProjected
        var residual2 = gpuEngine.AddGpu(normalized1, ffProjected);

        // 6. Second layer normalization
        var (output, _, _) = gpuEngine.LayerNormGpu(residual2, _norm2.GetGammaTensor(), _norm2.GetBetaTensor(), Convert.ToDouble(_norm2.GetEpsilon()));

        // Restore original tensor shape
        if (was2D)
        {
            output = gpuEngine.ReshapeGpu(output, [originalShape[0], originalShape[1]]);
        }
        else if (rank > 3)
        {
            // Restore original batch dimensions for higher-rank input
            int[] newShape = new int[rank];
            for (int d = 0; d < rank - 2; d++)
                newShape[d] = originalShape[d];
            newShape[rank - 2] = output.Shape[1];
            newShape[rank - 1] = output.Shape[2];
            output = gpuEngine.ReshapeGpu(output, newShape);
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
        // If forward received 2D input, the output gradient will also be 2D
        // We need to reshape it to 3D to match internal processing shapes
        Tensor<T> grad3D;
        bool gradWas2D = outputGradient.Shape.Length == 2;

        if (gradWas2D)
        {
            // 2D gradient: [seq, embed] -> [1, seq, embed]
            grad3D = outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1]);
        }
        else
        {
            grad3D = outputGradient;
        }

        // Backward pass through the second normalization layer
        var dNorm2 = _norm2.Backward(grad3D);

        // Split the gradient for the residual connection (copy gradient to both paths)
        var dFeedForward = dNorm2;
        var dNormalized1 = dNorm2; // Residual gradient flows directly

        // Backward pass through FFN (reverse order: projection then expansion)
        var dFF2 = _feedForward2.Backward(dFeedForward);
        var dFeedForwardInput = _feedForward1.Backward(dFF2);
        // Add gradients at the join point: dNormalized1 = dNorm2 + dFeedForwardInput
        dNormalized1 = Engine.TensorAdd(dNormalized1, dFeedForwardInput);

        // Backward pass through the first normalization layer
        var dNorm1 = _norm1.Backward(dNormalized1);

        // Split the gradient for the residual connection
        var dAttention = dNorm1;
        var dInput = dNorm1; // Residual gradient flows directly

        // Backward pass through the self-attention layer
        var dSelfAttentionInput = _selfAttention.Backward(dAttention);
        // Add gradients at the join point: dInput = dNorm1 + dSelfAttentionInput
        dInput = Engine.TensorAdd(dInput, dSelfAttentionInput);

        // If input was originally 2D, reshape gradient back to 2D
        if (gradWas2D)
        {
            dInput = dInput.Reshape(dInput.Shape[1], dInput.Shape[2]);
        }

        return dInput;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. Since this is a composite layer,
    /// it delegates to its sublayers which will use autodiff when their UseAutodiff flags are enabled.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // Composite layer - delegates to sublayers which use autodiff if enabled
        return BackwardManual(outputGradient);
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
        _feedForward1.UpdateParameters(learningRate);
        _feedForward2.UpdateParameters(learningRate);
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
        // === Vectorized Parameter Concatenation (Phase B: US-GPU-015) ===
        // Collect parameters from all sublayers and concatenate them
        var selfAttentionParams = _selfAttention.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var ff1Params = _feedForward1.GetParameters();
        var ff2Params = _feedForward2.GetParameters();
        var norm2Params = _norm2.GetParameters();

        // Concatenate all parameter vectors at once
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(
                    Vector<T>.Concatenate(selfAttentionParams, norm1Params),
                    ff1Params),
                ff2Params),
            norm2Params);
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
        _feedForward1.ResetState();
        _feedForward2.ResetState();
        _norm2.ResetState();
    }

    /// <summary>
    /// Computes the auxiliary loss for this layer by aggregating losses from sublayers.
    /// </summary>
    /// <returns>The computed auxiliary loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the auxiliary loss by aggregating losses from sublayers that implement IAuxiliaryLossLayer.
    /// Currently, this includes the self-attention mechanism which provides attention entropy and head diversity regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects additional learning signals from the layer's components.
    ///
    /// Auxiliary loss aggregation:
    /// - Checks each sublayer to see if it has auxiliary losses
    /// - Collects those losses and combines them
    /// - Returns the total for use in training
    ///
    /// Why this is useful:
    /// - The self-attention mechanism can benefit from regularization to prevent all heads from learning the same patterns
    /// - Aggregating losses at the encoder level provides a unified view of attention quality
    /// - This helps the entire encoder learn better representations
    ///
    /// Example: If the self-attention has an entropy loss (to keep attention focused) and a diversity loss
    /// (to prevent heads from being redundant), this method adds them together and returns the total.
    ///
    /// The aggregated loss helps ensure:
    /// - Attention heads learn diverse patterns
    /// - Attention is focused rather than diffuse
    /// - The encoder uses its capacity efficiently
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            _lastAuxiliaryLoss = NumOps.Zero;
            return _lastAuxiliaryLoss;
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
    /// and detailed diagnostics from sublayers.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information to help you understand how the auxiliary loss is working.
    ///
    /// The diagnostics show:
    /// - TotalAuxiliaryLoss: The combined penalty from all sublayers
    /// - AuxiliaryWeight: How much this penalty affects the overall training
    /// - UseAuxiliaryLoss: Whether this penalty is currently enabled
    /// - SelfAttentionDiagnostics: Detailed information from the self-attention mechanism
    ///
    /// You can use this information to:
    /// - Monitor if attention patterns are healthy (diverse and focused)
    /// - Debug training issues related to attention
    /// - Understand how the encoder is learning
    ///
    /// Example: If you see that attention entropy is very low, it might mean attention is too diffuse.
    /// If head diversity is very low, it might mean all heads are learning the same thing and capacity is wasted.
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
    /// Exports the transformer encoder layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the transformer encoder operation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a symbolic computation graph for JIT compilation:
    /// 1. Creates a symbolic input node
    /// 2. Applies multi-head self-attention with residual connection and norm
    /// 3. Applies feed-forward network with residual connection and norm
    /// 4. Returns the final output
    /// </para>
    /// <para><b>For Beginners:</b> This method builds a symbolic representation of a transformer encoder layer for JIT.
    ///
    /// The transformer encoder layer is a composite layer combining:
    /// - Multi-head self-attention (captures relationships between positions)
    /// - Layer normalization (stabilizes training)
    /// - Feed-forward network (processes each position independently)
    /// - Residual connections (helps gradient flow in deep networks)
    ///
    /// The forward pass:
    /// 1. x' = LayerNorm(x + MultiHeadAttention(x))
    /// 2. output = LayerNorm(x' + FeedForward(x'))
    ///
    /// JIT optimization for composite layers:
    /// - For now, composite layers note their structure but may delegate to sublayers
    /// - Future optimization could fuse operations across sublayers
    /// - Each sublayer (attention, feed-forward, norm) can be independently JIT compiled
    ///
    /// This is the core building block of BERT (12-24 encoder layers), GPT uses decoder layers.
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

        if (_selfAttention == null || _norm1 == null || _feedForward1 == null || _feedForward2 == null || _norm2 == null)
            throw new InvalidOperationException("Sublayers not initialized. Initialize the layer first.");

        // Create symbolic input node with batch dimension
        // InputShape is [sequenceLength, embeddingDimension]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "encoder_input");
        inputNodes.Add(inputNode);

        // Step 1: Self-attention sublayer using MultiHeadAttention operation
        var attentionOut = ApplyMultiHeadAttentionGraph(_selfAttention, inputNode);

        // Step 2: First residual connection: residual1 = input + attention_out
        var residual1 = TensorOperations<T>.Add(inputNode, attentionOut);

        // Step 3: First layer normalization
        var normalized1 = ApplyLayerNormGraph(_norm1, residual1);

        // Step 4: Feed-forward sublayer (two linear layers)
        var ffExpanded = ApplyFeedForwardGraph(_feedForward1, normalized1);
        var ffApplied = ApplyFeedForwardGraph(_feedForward2, ffExpanded);

        // Step 5: Second residual connection: residual2 = normalized1 + ff_out
        var residual2 = TensorOperations<T>.Add(normalized1, ffApplied);

        // Step 6: Second layer normalization
        var output = ApplyLayerNormGraph(_norm2, residual2);

        return output;
    }

    /// <summary>
    /// Applies multi-head attention graph to an input node.
    /// </summary>
    private ComputationNode<T> ApplyMultiHeadAttentionGraph(MultiHeadAttentionLayer<T> attentionLayer, ComputationNode<T> input)
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

        // Apply multi-head attention (self-attention: query, key, value all from same input)
        return TensorOperations<T>.MultiHeadAttention(
            query: input,
            key: input,
            value: input,
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
        // Get feed-forward weights and biases directly as tensors
        var weightsTensor = ffLayer.GetWeightsTensor();
        var biasTensor = ffLayer.GetBiasesTensor();

        if (weightsTensor == null || biasTensor == null)
            throw new InvalidOperationException("Feed-forward layer weights not initialized.");

        var weightsNode = TensorOperations<T>.Constant(weightsTensor, "ff_weights");
        var biasNode = TensorOperations<T>.Constant(biasTensor, "ff_bias");

        // Linear transformation: output = input @ weights + bias
        var weightsT = TensorOperations<T>.Transpose(weightsNode);
        var linear = TensorOperations<T>.MatrixMultiply(input, weightsT);
        var withBias = TensorOperations<T>.Add(linear, biasNode);

        // Apply activation if present using the activation's own ApplyToGraph method
        // This follows OCP - each activation knows how to export itself to a graph
        var activation = ffLayer.ScalarActivation;
        if (activation != null)
        {
            return activation.ApplyToGraph(withBias);
        }

        return withBias;
    }

    /// <summary>
    /// Gets whether this transformer encoder layer supports JIT compilation.
    /// </summary>
    /// <value>True if all sublayers support JIT compilation.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be JIT compiled. As a composite layer,
    /// it supports JIT if all its sublayers support JIT:
    /// - Multi-head self-attention layer
    /// - Layer normalization layers
    /// - Feed-forward layer
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if this composite layer can use JIT compilation.
    ///
    /// The transformer encoder layer can be JIT compiled if:
    /// - All sublayers are properly initialized
    /// - Each sublayer supports JIT compilation
    ///
    /// Composite layer JIT optimization:
    /// - Each sublayer can be independently JIT compiled
    /// - Future optimization: fuse operations across sublayers
    /// - Residual connections and layer norms are fast operations
    ///
    /// The bottleneck in transformers is typically the attention mechanism (O(n²)),
    /// which benefits most from JIT compilation. The feed-forward networks are also
    /// computationally expensive (matrix multiplications).
    ///
    /// BERT and other transformers stack 12-24 of these encoder layers, so optimizing
    /// each layer compounds to significant speedup for the full model.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // TransformerEncoderLayer is a composite layer
            // It supports JIT if all sublayers support JIT
            return _selfAttention != null && _selfAttention.SupportsJitCompilation &&
                   _norm1 != null && _norm1.SupportsJitCompilation &&
                   _feedForward1 != null && _feedForward1.SupportsJitCompilation &&
                   _feedForward2 != null && _feedForward2.SupportsJitCompilation &&
                   _norm2 != null && _norm2.SupportsJitCompilation;
        }
    }
}
