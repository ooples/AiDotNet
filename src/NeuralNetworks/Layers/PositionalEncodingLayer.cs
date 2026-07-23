using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that adds positional encodings to input sequences.
/// </summary>
/// <remarks>
/// <para>
/// The PositionalEncodingLayer adds position-dependent signals to input embeddings, which helps
/// sequence models like Transformers understand the order of elements in a sequence. Since
/// attention-based models have no inherent notion of sequence order, positional encodings
/// provide this critical information. The encodings use sine and cosine functions of different
/// frequencies to create unique position-dependent patterns.
/// </para>
/// <para><b>For Beginners:</b> This layer adds information about position to your sequence data.
/// 
/// Think of it like numbering the words in a sentence:
/// - Without position information, a model only knows which words are in the sentence
/// - With position information, it knows which word comes first, second, third, etc.
/// 
/// For example, the sentences "dog bites man" and "man bites dog" contain the same words
/// but have completely different meanings because of word order. Positional encoding
/// helps models understand this difference.
/// 
/// The layer uses a clever mathematical pattern of sine and cosine waves to encode positions.
/// This approach has several advantages:
/// - It creates a unique pattern for each position
/// - Similar positions have similar encodings (helpful for generalization)
/// - It can potentially handle sequences longer than those seen during training
/// - The encodings have consistent patterns that models can learn from
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Positional)]
[LayerTask(LayerTask.PositionalEncoding)]
[LayerProperty(IsTrainable = false, TestInputShape = "16, 8", TestConstructorArgs = "16, 8")]
public class PositionalEncodingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The maximum sequence length that this layer can handle.
    /// </summary>
    /// <remarks>
    /// This field is automatically extended when longer sequences are encountered.
    /// </remarks>
    private int maxSequenceLength;

    /// <summary>
    /// The size of each embedding vector.
    /// </summary>
    /// <remarks>
    /// This field is automatically adapted when input embedding dimensions differ.
    /// </remarks>
    private int embeddingSize;

    /// <summary>
    /// The pre-computed positional encodings tensor.
    /// </summary>
    /// <remarks>
    /// This tensor stores the pre-computed positional encodings for all possible positions
    /// up to maxSequenceLength. The encodings are calculated once during initialization
    /// and reused for all forward passes.
    /// </remarks>
    private Tensor<T> encodings;

    private readonly object _encodingLock = new();

    /// <summary>
    /// The computation engine (CPU or GPU) for vectorized operations.
    /// </summary>

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because the PositionalEncodingLayer supports backpropagation, even though it has no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer supports backpropagation during training. Although
    /// the PositionalEncodingLayer has no trainable parameters, it still supports the backward pass to propagate
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
    public override bool SupportsTraining => false; // Precomputed constants, no trainable parameters

    /// <summary>
    /// Initializes a new instance of the <see cref="PositionalEncodingLayer{T}"/> class with the specified maximum sequence length and embedding size.
    /// </summary>
    /// <param name="maxSequenceLength">The maximum sequence length that this layer can handle.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a PositionalEncodingLayer with the specified maximum sequence length and embedding size.
    /// It initializes the positional encodings using sine and cosine functions of different frequencies, following
    /// the formula from the "Attention Is All You Need" paper.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary dimensions.
    /// 
    /// When creating a PositionalEncodingLayer, you need to specify:
    /// - maxSequenceLength: The longest sequence your model will handle (e.g., 512 for text processing)
    /// - embeddingSize: The size of your embedding vectors (e.g., 512 or 768 dimensions)
    /// 
    /// During initialization, the layer pre-calculates all the positional encodings using
    /// the sine/cosine formula. This is more efficient than calculating them each time.
    /// 
    /// The formula alternates between sine and cosine functions across the embedding dimensions,
    /// with different frequencies for different dimensions. This creates a unique pattern for each
    /// position that the model can learn to recognize.
    /// </para>
    /// </remarks>
    public PositionalEncodingLayer(int maxSequenceLength, int embeddingSize)
        : base([maxSequenceLength, embeddingSize], [maxSequenceLength, embeddingSize])
    {
        this.maxSequenceLength = maxSequenceLength;
        this.embeddingSize = embeddingSize;
        encodings = new Tensor<T>([maxSequenceLength, embeddingSize]);
        InitializeEncodings();
    }

    /// <summary>
    /// Initializes the positional encodings using sine and cosine functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the positional encodings tensor using sine and cosine functions
    /// of different frequencies. For each position and embedding dimension, it calculates the
    /// appropriate value based on the formula from the "Attention Is All You Need" paper:
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the pattern of numbers that encodes position information.
    /// 
    /// The method uses a specific mathematical formula to create a unique pattern for each position:
    /// - Even-indexed dimensions (0, 2, 4, ...) use sine functions
    /// - Odd-indexed dimensions (1, 3, 5, ...) use cosine functions
    /// - Different dimensions use different frequencies
    /// 
    /// This creates a unique "fingerprint" for each position that:
    /// - Changes smoothly as you move along the sequence
    /// - Has different patterns across different dimensions
    /// - Can be easily learned by neural networks
    /// 
    /// The formula with 10000 and sine/cosine was carefully chosen by researchers
    /// to have good mathematical properties for representing sequence positions.
    /// </para>
    /// </remarks>
    private void InitializeEncodings()
    {
        // Vectorized positional encoding computation using IEngine
        // Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        //          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        // Pre-compute division terms: 1 / 10000^(2i/d_model) for each dimension pair
        int halfEmbedding = embeddingSize / 2;
        var divTerms = new Tensor<T>(new[] { halfEmbedding });
        for (int i = 0; i < halfEmbedding; i++)
        {
            double exponent = NumericalStabilityHelper.SafeDiv(2.0 * i, embeddingSize);
            divTerms[i] = NumOps.FromDouble(1.0 / Math.Pow(10000, exponent));
        }

        // Create position tensor [maxSequenceLength]
        var positions = new Tensor<T>(new[] { maxSequenceLength });
        for (int pos = 0; pos < maxSequenceLength; pos++)
        {
            positions[pos] = NumOps.FromDouble(pos);
        }

        // Compute all angles: angles[pos, i] = pos * divTerms[i]
        // This is an outer product: [maxSequenceLength] x [halfEmbedding] -> [maxSequenceLength, halfEmbedding]
        var angles = Engine.TensorOuter(positions, divTerms);

        // Apply vectorized sin and cos using IEngine
        var sinValues = Engine.TensorSin(angles);
        var cosValues = Engine.TensorCos(angles);

        // Interleave sin and cos values into encodings tensor
        // Even indices get sin, odd indices get cos
        for (int pos = 0; pos < maxSequenceLength; pos++)
        {
            for (int i = 0; i < halfEmbedding; i++)
            {
                encodings[pos, 2 * i] = sinValues[pos, i];         // Even: sin
                encodings[pos, 2 * i + 1] = cosValues[pos, i];     // Odd: cos
            }
        }

        // Handle odd embeddingSize (last dimension uses sin if odd) - vectorized
        if (embeddingSize % 2 == 1)
        {
            int lastDimIdx = embeddingSize - 1;
            double exponent = NumericalStabilityHelper.SafeDiv(2.0 * (lastDimIdx / 2.0), embeddingSize);
            T divTerm = NumOps.FromDouble(1.0 / Math.Pow(10000, exponent));

            // Vectorized: angles = positions * divTerm, then sin(angles)
            var lastAngles = Engine.TensorMultiplyScalar(positions, divTerm);
            var lastSinValues = Engine.TensorSin(lastAngles);

            // Copy vectorized results to encodings
            for (int pos = 0; pos < maxSequenceLength; pos++)
            {
                encodings[pos, lastDimIdx] = lastSinValues[pos];
            }
        }
    }

    /// <summary>
    /// Performs the forward pass of the positional encoding layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor with positional encodings added.</returns>
    /// <exception cref="ArgumentException">Thrown when the input sequence length exceeds the maximum sequence length.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the positional encoding layer. It first checks that
    /// the input sequence length does not exceed the maximum allowed length. Then, it slices the
    /// pre-computed encodings tensor to match the input sequence length and adds the encodings to
    /// the input tensor element-wise.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds the position information to your input data.
    /// 
    /// During the forward pass:
    /// - The method checks that your sequence isn't too long
    /// - It takes the appropriate slice of the pre-computed encodings
    ///   (matching the length of your input sequence)
    /// - It adds these encodings directly to your input data
    /// 
    /// The addition operation combines your original data (like word embeddings)
    /// with the position information, allowing the model to use both.
    /// 
    /// For example, if your input is word embeddings for "The cat sat on the mat",
    /// after this layer, each word's embedding will also contain information about
    /// which position in the sentence it occupies.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Handle 1D input by treating it as [1, embed] (single position with embedding)
        bool was1D = input.Shape.Length == 1;
        Tensor<T> workingInput = input;

        if (was1D)
        {
            // Reshape [embed] -> [1, embed]
            workingInput = Engine.Reshape(input, [1, input.Shape[0]]);
        }

        // Handle any rank >= 2: last dim is embed, second-to-last is sequence
        int rank = workingInput.Shape.Length;
        int seqLength = workingInput.Shape[rank - 2];
        int inputEmbedDim = workingInput.Shape[rank - 1];

        Tensor<T> currentEncodings;
        int currentEmbeddingSize;

        lock (_encodingLock)
        {
            // Dynamically adjust embedding size if needed
            if (inputEmbedDim != embeddingSize)
            {
                embeddingSize = inputEmbedDim;
                encodings = new Tensor<T>([maxSequenceLength, embeddingSize]);

                for (int pos = 0; pos < maxSequenceLength; pos++)
                {
                    for (int i = 0; i < embeddingSize / 2; i++)
                    {
                        double exponent = NumericalStabilityHelper.SafeDiv(2.0 * i, embeddingSize);
                        double divTerm = 1.0 / Math.Pow(10000, exponent);
                        encodings[pos, 2 * i] = NumOps.FromDouble(Math.Sin(pos * divTerm));
                        encodings[pos, 2 * i + 1] = NumOps.FromDouble(Math.Cos(pos * divTerm));
                    }
                    if (embeddingSize % 2 == 1)
                    {
                        int lastDimIdx = embeddingSize - 1;
                        double exponent = NumericalStabilityHelper.SafeDiv(2.0 * (lastDimIdx / 2.0), embeddingSize);
                        double divTerm = 1.0 / Math.Pow(10000, exponent);
                        encodings[pos, lastDimIdx] = NumOps.FromDouble(Math.Sin(pos * divTerm));
                    }
                }
            }

            // Dynamically extend encodings if needed (support any sequence length)
            if (seqLength > maxSequenceLength)
            {
                int oldMaxSeq = maxSequenceLength;
                maxSequenceLength = seqLength;
                var newEncodings = new Tensor<T>([maxSequenceLength, embeddingSize]);

                for (int pos = 0; pos < oldMaxSeq; pos++)
                {
                    for (int e = 0; e < embeddingSize; e++)
                    {
                        newEncodings[pos, e] = encodings[pos, e];
                    }
                }

                for (int pos = oldMaxSeq; pos < maxSequenceLength; pos++)
                {
                    for (int i = 0; i < embeddingSize / 2; i++)
                    {
                        double exponent = NumericalStabilityHelper.SafeDiv(2.0 * i, embeddingSize);
                        double divTerm = 1.0 / Math.Pow(10000, exponent);
                        newEncodings[pos, 2 * i] = NumOps.FromDouble(Math.Sin(pos * divTerm));
                        newEncodings[pos, 2 * i + 1] = NumOps.FromDouble(Math.Cos(pos * divTerm));
                    }
                    if (embeddingSize % 2 == 1)
                    {
                        int lastDimIdx = embeddingSize - 1;
                        double exponent = NumericalStabilityHelper.SafeDiv(2.0 * (lastDimIdx / 2.0), embeddingSize);
                        double divTerm = 1.0 / Math.Pow(10000, exponent);
                        newEncodings[pos, lastDimIdx] = NumOps.FromDouble(Math.Sin(pos * divTerm));
                    }
                }

                encodings = newEncodings;
            }

            currentEncodings = encodings;
            currentEmbeddingSize = embeddingSize;
        }

        // Slice encodings to match input sequence length: [seq, embed]
        var slicedEncodings = currentEncodings.Slice(0, 0, seqLength, currentEmbeddingSize);

        Tensor<T> result;
        if (rank == 2)
        {
            // For 2D input [seq, embed], add directly
            result = Engine.TensorAdd(workingInput, slicedEncodings);
        }
        else
        {
            // For higher-rank input [..., seq, embed], broadcast encoding across leading dimensions
            // Create broadcast shape: [1, 1, ..., 1, seq, embed] with (rank-2) leading 1s
            var broadcastShape = new int[rank];
            for (int d = 0; d < rank - 2; d++)
                broadcastShape[d] = 1;
            broadcastShape[rank - 2] = seqLength;
            broadcastShape[rank - 1] = currentEmbeddingSize;

            var reshapedEncodings = Engine.Reshape(slicedEncodings, broadcastShape);
            result = Engine.TensorBroadcastAdd(workingInput, reshapedEncodings);
        }

        // If input was 1D, reshape output back to 1D
        if (was1D)
        {
            // Reshape [1, embed] -> [embed]
            result = Engine.Reshape(result, [result.Shape[result.Shape.Length - 1]]);
        }

        return result;
    }

    /// <summary>
    /// Updates the parameters of the positional encoding layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is part of the training process, but since PositionalEncodingLayer has no trainable parameters,
    /// this method does nothing. The positional encodings are fixed and do not change during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally update a layer's internal values during training.
    /// 
    /// However, since PositionalEncodingLayer uses fixed encodings that are calculated once at initialization
    /// and don't change during training, this method is empty.
    /// 
    /// This is different from layers like Dense or Convolutional layers, which have weights and biases
    /// that get updated during training. The positional encodings are based on a mathematical formula
    /// rather than learned from data.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all trainable parameters from the positional encoding layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since PositionalEncodingLayer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. Since PositionalEncodingLayer
    /// has no trainable parameters, it returns an empty vector. The positional encodings are fixed values
    /// determined by a mathematical formula, not learnable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the learnable values in the layer.
    /// 
    /// Since PositionalEncodingLayer:
    /// - Uses fixed encodings based on a mathematical formula
    /// - Has no weights, biases, or other learnable parameters
    /// - The method returns an empty list
    /// 
    /// This is different from layers like Dense layers, which would return their weights and biases.
    /// The positional encodings are fixed by design and don't need to be learned from data.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // PositionalEncodingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the positional encoding layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is intended to reset any internal state that might change during training or inference.
    /// However, since PositionalEncodingLayer has no state that changes (the encodings are fixed),
    /// this method does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally clear the layer's memory to start fresh.
    /// 
    /// However, since PositionalEncodingLayer doesn't maintain any changing state during processing
    /// (the encodings are fixed at initialization and don't change), this method is empty.
    /// 
    /// The encodings tensor is a fixed part of the layer that remains constant throughout
    /// the lifetime of the layer, so there's nothing to reset.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // No state to reset in this layer
        // The encodings are fixed and don't change during training
    }

    /// <summary>
    /// Gets whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs the forward pass on GPU, adding positional encodings to input embeddings.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensors (uses first input).</param>
    /// <returns>GPU-resident output tensor with positional encodings added.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];
        var inputShape = input._shape;
        int rank = inputShape.Length;

        // Handle 1D input by treating as [1, embed]
        bool was1D = rank == 1;
        Tensor<T> workingInput = input;

        if (was1D)
        {
            workingInput = gpuEngine.ReshapeGpu(input, [1, inputShape[0]]);
            rank = 2;
        }

        // Last dim is embed, second-to-last is sequence
        int seqLength = workingInput.Shape[rank - 2];
        int inputEmbedDim = workingInput.Shape[rank - 1];

        Tensor<T> currentEncodings;
        int currentEmbeddingSize;

        lock (_encodingLock)
        {
            // Dynamically adjust embedding size if needed
            if (inputEmbedDim != embeddingSize)
            {
                embeddingSize = inputEmbedDim;
                encodings = new Tensor<T>([maxSequenceLength, embeddingSize]);
                InitializeEncodings();
            }

            // Dynamically extend encodings if needed
            if (seqLength > maxSequenceLength)
            {
                int oldMaxSeq = maxSequenceLength;
                maxSequenceLength = seqLength;
                var newEncodings = new Tensor<T>([maxSequenceLength, embeddingSize]);

                for (int pos = 0; pos < oldMaxSeq; pos++)
                {
                    for (int e = 0; e < embeddingSize; e++)
                    {
                        newEncodings[pos, e] = encodings[pos, e];
                    }
                }

                for (int pos = oldMaxSeq; pos < maxSequenceLength; pos++)
                {
                    for (int i = 0; i < embeddingSize / 2; i++)
                    {
                        double exponent = NumericalStabilityHelper.SafeDiv(2.0 * i, embeddingSize);
                        double divTerm = 1.0 / Math.Pow(10000, exponent);
                        newEncodings[pos, 2 * i] = NumOps.FromDouble(Math.Sin(pos * divTerm));
                        newEncodings[pos, 2 * i + 1] = NumOps.FromDouble(Math.Cos(pos * divTerm));
                    }
                    if (embeddingSize % 2 == 1)
                    {
                        int lastDimIdx = embeddingSize - 1;
                        double exponent = NumericalStabilityHelper.SafeDiv(2.0 * (lastDimIdx / 2.0), embeddingSize);
                        double divTerm = 1.0 / Math.Pow(10000, exponent);
                        newEncodings[pos, lastDimIdx] = NumOps.FromDouble(Math.Sin(pos * divTerm));
                    }
                }

                encodings = newEncodings;
            }

            currentEncodings = encodings;
            currentEmbeddingSize = embeddingSize;
        }

        // Slice encodings to match input sequence length
        var slicedEncodings = currentEncodings.Slice(0, 0, seqLength, currentEmbeddingSize);

        // Upload encodings to GPU
        var gpuEncodings = gpuEngine.UploadToGpu(slicedEncodings, GpuTensorRole.Activation);

        Tensor<T> result;
        if (rank == 2)
        {
            // Direct add for 2D input
            result = gpuEngine.AddGpu(workingInput, gpuEncodings);
        }
        else
        {
            // For higher-rank input, we need to broadcast
            // Reshape encodings to match broadcast shape: [1, 1, ..., 1, seq, embed]
            var broadcastShape = new int[rank];
            for (int d = 0; d < rank - 2; d++)
                broadcastShape[d] = 1;
            broadcastShape[rank - 2] = seqLength;
            broadcastShape[rank - 1] = currentEmbeddingSize;

            var reshapedEncodings = gpuEngine.ReshapeGpu(gpuEncodings, broadcastShape);

            // Tile encodings to match input's batch dimensions
            int totalBatchSize = 1;
            for (int d = 0; d < rank - 2; d++)
                totalBatchSize *= workingInput.Shape[d];

            var tiledEncodings = gpuEngine.TileBatchGpu(reshapedEncodings, totalBatchSize);
            var finalEncodings = gpuEngine.ReshapeGpu(tiledEncodings, workingInput._shape);

            result = gpuEngine.AddGpu(workingInput, finalEncodings);
        }

        // Reshape back to 1D if needed
        if (was1D)
        {
            result = gpuEngine.ReshapeGpu(result, [result.Shape[^1]]);
        }

        return result;
    }
}
