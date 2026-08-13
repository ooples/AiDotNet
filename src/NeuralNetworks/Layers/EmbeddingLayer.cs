using AiDotNet.Helpers;
using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an embedding layer that converts discrete token indices into dense vector representations.
/// </summary>
/// <remarks>
/// <para>
/// An embedding layer maps discrete tokens (represented as indices) to continuous vector representations.
/// This is particularly useful for natural language processing tasks where words or tokens need to be
/// represented as dense vectors that capture semantic relationships. Each token is assigned a unique
/// vector in a high-dimensional space, allowing the model to learn meaningful representations.
/// </para>
/// <para><b>For Beginners:</b> An embedding layer turns words or other symbols into lists of numbers that capture their meaning.
///
/// Imagine you have a dictionary where:
/// - Each word has an ID number (like "cat" = 5, "dog" = 10)
/// - The embedding layer gives each ID a unique "coordinate" in a multi-dimensional space
/// - Words with similar meanings end up with similar coordinates
///
/// For example:
/// - "Cat" might become [0.2, -0.5, 0.1, 0.8]
/// - "Kitten" might become [0.25, -0.4, 0.15, 0.7]
/// - "Computer" might become [-0.8, 0.2, 0.5, -0.3]
///
/// The embedding layer learns these representations during training, so that:
/// - Similar words end up close to each other
/// - Related concepts form clusters
/// - The vectors capture meaningful semantic relationships
///
/// This allows neural networks to work with text and other discrete tokens in a way
/// that captures their meaning and relationships.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This layer is not thread-safe. Each layer instance maintains internal state
/// during forward and backward passes. If you need concurrent execution, use separate layer instances
/// per thread or synchronize access to shared instances.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "100, 16")]
[TensorPort("input", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenIds, MaxExclusiveMember = "_vocabularySize")]
[TensorPort("output", TensorPortDirection.Output, LayerInputDomainKind.Continuous,
    Role = TensorPortRole.Features)]
// An embedding is an index lookup: [Time] or [Batch, Time] in, with one embedding vector appended.
// Continuous feature projection is deliberately represented by DenseLayer, whose feature-last shape
// and continuous input-domain contracts are different. Keeping the operations as separate types makes
// a layer's parameter set, input domain, and output rank statically unambiguous.
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class EmbeddingLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>, ITokenEmbedding<T>, IShapeContract
{
    /// <summary>
    /// The embedding tensor that stores vector representations for each token in the vocabulary.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the learnable embedding vectors for each token in the vocabulary. The rows
    /// correspond to token indices, and the columns represent the dimensions of the embedding space.
    /// Each row in the tensor is the embedding vector for the corresponding token.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "dictionary" that maps each token ID to its vector representation.
    ///
    /// The embedding tensor works like this:
    /// - Each row corresponds to one token (word, character, etc.)
    /// - Each column is one dimension of the embedding space
    /// - If you have 10,000 words and 300 dimensions, the tensor will be 10,000 × 300
    ///
    /// For example, with a vocabulary of 5 words and 4 dimensions:
    /// ```
    /// Word ID | Embedding Vector
    /// --------|-----------------
    /// 0       | [0.1, 0.2, -0.3, 0.5]
    /// 1       | [-0.5, 0.8, 0.1, -0.2]
    /// 2       | [0.4, -0.1, -0.7, 0.3]
    /// 3       | [0.2, 0.5, 0.6, -0.4]
    /// 4       | [-0.3, -0.2, 0.4, 0.8]
    /// ```
    ///
    /// During training, these values are adjusted to make similar tokens have similar vectors.
    /// </para>
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Embeddings,
        Shape = "_vocabularySize, _embeddingDimension")]
    private Tensor<T> _embeddingTensor;

    /// <summary>
    /// Cached vocabulary size and embedding dimension. Stored as fields rather
    /// than read from <see cref="_embeddingTensor"/>'s shape so they remain
    /// authoritative when the tensor is in its zero-sized lazy-init placeholder
    /// state. Embedding tensors at transformer scale (e.g., BERT vocab 30,522 ×
    /// dim 768 = ~187 MB per instance) are by far the largest single allocation
    /// in BGE/SGPT/Matryoshka model construction; deferring it cuts test-time
    /// memory by hundreds of MB until the first Forward() pass actually runs.
    /// </summary>
    private readonly int _vocabularySize;
    private readonly int _embeddingDimension;
    private bool _embeddingInitialized = true;

    /// <inheritdoc />
    /// <remarks>
    /// Large embedding tables deliberately start as a zero-sized placeholder. Reporting the base
    /// class's unconditional <c>true</c> made the model manifest call that placeholder materialized;
    /// chunk enumeration then allocated the real table and appeared to change the parameter count.
    /// Readiness now describes the actual storage lifecycle without allocating it.
    /// </remarks>
    public override bool IsInitialized => _embeddingInitialized;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var table = _embeddingTensor;
        if (table is null || table.Rank < 2) return null;
        int embeddingDim = table.Shape[1];
        if (embeddingDim <= 0) return null;

        return inputRank switch
        {
            1 => new[]
            {
                new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(embeddingDim)),
            },
            2 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(embeddingDim)),
            },
            _ => null,
        };
    }

    // GPU-resident cached tensors for GPU training pipeline
    private Tensor<T>? _lastInputGpu;
    private int[]? _lastInputGpuShape;
    private Tensor<int>? _lastIndicesForGpu;

    /// <summary>
    /// The gradients for the embedding tensor, computed during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each element in the embedding tensor.
    /// These gradients are used to update the embeddings during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust each embedding value.
    ///
    /// During training:
    /// - The network calculates how each embedding vector contributed to errors
    /// - These gradients show how to change each value to improve performance
    /// - Larger gradients mean bigger adjustments are needed
    ///
    /// For example, if the network predicts incorrectly using the embedding for "cat",
    /// the gradients will indicate how to adjust that specific embedding vector to
    /// improve future predictions.
    ///
    /// Only the embeddings for tokens that were actually used in the current batch
    /// will receive gradient updates.
    /// </para>
    /// </remarks>
    private Tensor<T>? _embeddingGradient;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input indices received during the last forward pass. These indices are
    /// necessary for computing the gradients during the backward pass, as they indicate which embeddings
    /// were accessed.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers which token IDs were processed in the latest calculation.
    /// 
    /// During training:
    /// - The layer needs to remember which tokens it looked up
    /// - This helps when calculating how to improve the embeddings
    /// - Only the embeddings for these specific tokens will be updated
    /// 
    /// For example, if the input was the sequence [5, 10, 3] (representing three tokens),
    /// only the embeddings for token IDs 5, 10, and 3 will receive updates during training.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the last computed embedding regularization loss for diagnostics.
    /// </summary>
    private T _lastEmbeddingRegularizationLoss;

    /// <summary>
    /// Gets or sets whether to use auxiliary loss (embedding regularization) during training.
    /// Default is false. Enable to prevent embeddings from becoming too large or collapsing.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for embedding regularization.
    /// Default is 0.0001. Controls L2 regularization strength on embedding weights.
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because this layer has trainable parameters (the embedding matrix).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the embedding layer supports training through backpropagation.
    /// The layer has trainable embeddings that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its embeddings during training
    /// - It will improve its representations as it sees more data
    /// - It has parameters (the embedding matrix) that are updated to make better predictions
    /// 
    /// Unlike static word embeddings (like pre-trained word vectors), these embeddings
    /// adapt and improve specifically for your task during training.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// When <c>true</c>, the token-embedding lookup output is multiplied by
    /// <c>sqrt(embeddingDimension)</c> (Vaswani et al. 2017 §3.4). This scales the
    /// (small, Glorot-initialised) embeddings up so they are not drowned out by the
    /// fixed-magnitude sinusoidal positional encoding that is added immediately after,
    /// preserving token identity through the encoder. Opt-in (default <c>false</c>) so
    /// existing models that use the embedding as a plain lookup are unaffected; the
    /// transformer builder sets it for embeddings paired with positional encoding.
    /// Applies to the token-lookup output before it is combined with positional encoding.
    /// </summary>
    public bool ScaleBySqrtDimension { get; set; } = false;

    /// <summary>
    /// Gets a value indicating whether this layer can execute on GPU.
    /// </summary>
    /// <value>
    /// <c>true</c> because embedding lookup has efficient GPU support.
    /// </value>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Returns layer-specific metadata for serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata stores the vocabulary size so the layer can be rebuilt
    /// correctly when loading a saved model.
    /// </para>
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        // Use the stored fields (not _embeddingTensor.Shape) because the
        // tensor may be a [0,0] lazy placeholder before first Forward.
        metadata["VocabularySize"] = _vocabularySize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["EmbeddingDimension"] = _embeddingDimension.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ScaleBySqrtDimension"] = ScaleBySqrtDimension ? "true" : "false";
        return metadata;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EmbeddingLayer{T}"/> class.
    /// </summary>
    /// <param name="vocabularySize">The number of unique tokens in the vocabulary.</param>
    /// <param name="embeddingDimension">The dimension of the embedding vectors.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new embedding layer with the specified vocabulary size and embedding dimension.
    /// The embedding matrix is initialized with small random values scaled to help with training convergence.
    /// The input shape is set to [1] because the layer expects token indices, and the output shape is
    /// set to [embeddingDimension] as each token is mapped to a vector of that dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the embedding layer with the vocabulary size and embedding dimensions you need.
    ///
    /// When creating an embedding layer, you need to specify:
    /// - Vocabulary size: How many different tokens (words, characters, etc.) your model will handle
    /// - Embedding dimension: How many numbers to use for each token's representation
    ///
    /// For example:
    /// ```csharp
    /// // Create an embedding layer for 10,000 words with 300-dimensional embeddings
    /// var wordEmbedding = new EmbeddingLayer<float>(10000, 300);
    ///
    /// // Create an embedding layer for 128 characters with 50-dimensional embeddings
    /// var charEmbedding = new EmbeddingLayer<float>(128, 50);
    /// ```
    ///
    /// Typical embedding dimensions:
    /// - For words: 100-300 dimensions
    /// - For characters: 25-100 dimensions
    /// - For special tokens: 50-200 dimensions
    ///
    /// Larger dimensions can capture more information but require more computation and memory.
    /// </para>
    /// </remarks>
    public EmbeddingLayer(
        [LayerState] int vocabularySize,
        [LayerState] int embeddingDimension)
        : base([1], [embeddingDimension])
    {
        AuxiliaryLossWeight = NumOps.FromDouble(0.0001);
        _lastEmbeddingRegularizationLoss = NumOps.Zero;

        _vocabularySize = vocabularySize;
        _embeddingDimension = embeddingDimension;

        // Lazy initialization by default — the embedding tensor for transformer-
        // scale vocabularies dominates per-instance memory cost (BERT 30K vocab ×
        // 768 dim ≈ 187 MB). Stays at [0,0] until EnsureEmbeddingInitialized()
        // runs on the first Forward / GetParameters / SetParameters call.
        _embeddingTensor = new Tensor<T>([0, 0]);
        _embeddingInitialized = false;

    }

    /// <summary>
    /// Materializes the embedding tensor on first access. Cheap no-op after the
    /// first call. Initialization runs the same SimdRandom-scaled fill the
    /// constructor used to do eagerly, then registers the tensor with the
    /// engine for GPU persistence.
    /// </summary>
    /// <summary>
    /// Allocates the embedding table, whose shape is
    /// <c>[_vocabularySize, _embeddingDimension]</c> — both fixed at construction.
    /// </summary>
    /// <remarks>
    /// Because allocation happens lazily on first use, a freshly constructed layer
    /// offered one placeholder tensor of zero length, and a restore arrived with 3,072 values for a
    /// layer reporting none. The underlying routine treats an already-materialized table as
    /// authoritative trained state, so running it from here cannot overwrite a restore.
    /// </remarks>
    protected override void EnsureInitialized()
    {
        EnsureEmbeddingInitialized();
        base.EnsureInitialized();
    }

    private void EnsureEmbeddingInitialized()
    {
        if (_embeddingInitialized) return;

        lock (InitializationLock)
        {
            if (_embeddingInitialized) return;

            // A deserializer, ParameterBuffer, or copy-on-write clone can install a
            // fully materialized embedding tensor before this fresh layer has ever
            // executed Forward. In that case the tensor is the authoritative trained
            // state. Treat it exactly like PyTorch treats a materialized lazy module:
            // synchronize the runtime latch/registration without allocating over it.
            // Reinitializing here silently replaced the COW-shared trained table on a
            // clone's first prediction (UniAudio Clone_AfterTraining).
            if (WeightsAlreadyAllocated(_embeddingTensor, _vocabularySize, _embeddingDimension))
            {
                RegisterTrainableParameter(_embeddingTensor, PersistentTensorRole.Embeddings);
                _embeddingInitialized = true;
                return;
            }

            // Streaming-aware allocation: PaLM-E-scale models have
            // vocab × embed embedding matrices in the multi-GB range
            // (e.g. 256K × 8192 fp32 = 8 GB). Routing through
            // AllocateLazyWeight lets the streaming pool pre-evict
            // before the GC byte[] lands. Falls back to plain
            // new Tensor<T>(shape) for non-streaming models.
            _embeddingTensor = AllocateLazyWeight([_vocabularySize, _embeddingDimension]);
            InitializeParameters();
            RegisterTrainableParameter(_embeddingTensor, PersistentTensorRole.Embeddings);
            _embeddingInitialized = true;
        }
    }

    /// <inheritdoc/>
    public Matrix<T> GetTokenEmbeddings(IReadOnlyList<int> tokenIds)
    {
        if (tokenIds is null)
        {
            throw new ArgumentNullException(nameof(tokenIds));
        }

        EnsureEmbeddingInitialized();

        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];
        var result = new Matrix<T>(tokenIds.Count, embeddingDim);

        for (int i = 0; i < tokenIds.Count; i++)
        {
            int tokenId = tokenIds[i];
            if (tokenId < 0 || tokenId >= vocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {tokenId} is out of range 0..{vocabSize - 1}.");
            }

            for (int d = 0; d < embeddingDim; d++)
            {
                result[i, d] = _embeddingTensor[tokenId, d];
            }
        }

        return result;
    }

    /// <summary>
    /// Initializes the embedding tensor with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the embedding tensor with small random values scaled by a factor
    /// that depends on the embedding dimension. This scaling helps in achieving good convergence
    /// during training by preventing the initial values from being too large or too small.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial random values for all embeddings.
    ///
    /// Before training begins:
    /// - Each embedding needs some starting value
    /// - We use small random values, centered around zero
    /// - The values are scaled based on the embedding dimension
    ///
    /// This initialization is important because:
    /// - Too large values could cause training instability
    /// - Too small values could slow down learning
    /// - The scaling factor helps find a good middle ground
    ///
    /// As training progresses, these random initial values will gradually be replaced
    /// with meaningful representations learned from data.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];

        // Initialize embedding tensor with small random values using Engine operations
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(NumOps.FromDouble(1.0), NumOps.FromDouble(embeddingDim)));

        // Initialize with SimdRandom: random [0,1] → shift to [-0.5, 0.5] → scale.
        // For double/float, write directly to avoid NumOps.FromDouble virtual dispatch
        // on every element (23M elements for BERT vocab). The embedding tensor was
        // already allocated by the constructor; reuse it in place rather than allocating
        // a second tensor that shadows the original reference.
        //
        // Honour the layer-level deterministic seed when set
        // (LayerBase<T>.RandomSeed is the per-layer-seed mechanism wired
        // by LayerHelper.Wire from architecture.RandomSeed). Without this
        // hook, the architecture's RandomSeed wouldn't reach the embedding
        // fill — closes review-comment #1270.vhmx.
        SimdRandom rng = RandomSeed.HasValue
            ? new SimdRandom(RandomSeed.Value)
            : new SimdRandom();
        var span = _embeddingTensor.Data.Span;
        int total = span.Length;
        if (total == 0) return; // zero-sized embedding (no vocab or zero dim): nothing to fill
        double scaleD = NumOps.ToDouble(scale);

        // Write via a temp array + array-level reinterpret so the SIMD-batched
        // xoshiro256** fill path still applies. See MultiHeadAttentionLayer for full
        // rationale (Span<T> can't be reinterpreted across generic T without a struct
        // constraint, which we don't have, and CreateSpan isn't on net471).
        if (typeof(T) == typeof(double))
        {
            var buffer = new double[total];
            rng.NextDoubles(buffer.AsSpan());
            for (int i = 0; i < total; i++)
                buffer[i] = (buffer[i] - 0.5) * scaleD;
            var reinterpreted = System.Runtime.CompilerServices.Unsafe.As<double[], T[]>(ref buffer);
            reinterpreted.AsSpan(0, total).CopyTo(span);
        }
        else if (typeof(T) == typeof(float))
        {
            var buffer = new float[total];
            rng.NextFloats(buffer.AsSpan());
            float scaleF = (float)scaleD;
            for (int i = 0; i < total; i++)
                buffer[i] = (buffer[i] - 0.5f) * scaleF;
            var reinterpreted = System.Runtime.CompilerServices.Unsafe.As<float[], T[]>(ref buffer);
            reinterpreted.AsSpan(0, total).CopyTo(span);
        }
        else
        {
            const int batchSize = 4096;
            var tempBuf = new double[Math.Min(total, batchSize)];
            int offset = 0;
            while (offset < total)
            {
                int chunk = Math.Min(batchSize, total - offset);
                rng.NextDoubles(tempBuf.AsSpan(0, chunk));
                for (int j = 0; j < chunk; j++)
                    span[offset + j] = NumOps.FromDouble((tempBuf[j] - 0.5) * scaleD);
                offset += chunk;
            }
        }
    }

    /// <summary>
    /// The original input shape, saved for backward pass.
    /// </summary>
    private int[] _originalInputShape = [];

    /// <summary>
    /// Performs the forward pass of the embedding layer, converting token indices to vector representations.
    /// </summary>
    /// <param name="input">The input tensor containing token indices. Supports any-rank tensors:
    /// - 1D: [seqLen] - single sequence
    /// - 2D: [batch, seqLen] - batch of sequences (industry standard)
    /// - 3D: [batch, seqLen, 1] - compatible with legacy format
    /// </param>
    /// <returns>The output tensor containing embedding vectors with the same leading dimensions plus embeddingDim.</returns>
    /// <remarks>
    /// <para>
    /// <b>Industry Standard:</b> Like PyTorch's nn.Embedding, this layer supports any-rank input tensors.
    /// The indices in the last dimension(s) are looked up in the embedding table, and the result has
    /// the same shape with the last dimension replaced by the embedding dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method looks up the vector for each token ID in your input.
    ///
    /// The forward pass works like this:
    /// 1. Take a sequence of token IDs as input (like [5, 10, 3])
    /// 2. For each ID, look up its corresponding row in the embedding matrix
    /// 3. Copy that row (the embedding vector) to the output
    ///
    /// For example, with an input sequence [5, 10, 3]:
    /// - Look up row 5 in the embedding matrix -> output row 1
    /// - Look up row 10 in the embedding matrix -> output row 2
    /// - Look up row 3 in the embedding matrix -> output row 3
    ///
    /// The result is a sequence of embedding vectors, one for each input token.
    /// This transforms your discrete tokens into continuous vectors that the neural
    /// network can process more effectively.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Materialize the embedding tensor before any lookup runs. Lazy by default
        // so unused embedding layers in test construction don't pay the multi-MB
        // allocation up front.
        EnsureEmbeddingInitialized();

        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)
        _originalInputShape = input._shape;

        int embeddingDim = _embeddingTensor.Shape[1];
        int vocabularySize = _embeddingTensor.Shape[0];

        // The layer is an index lookup by construction. Continuous feature projection belongs in
        // DenseLayer, so invalid values cannot silently change this layer's output rank or parameters.
        ValidateIndicesOrThrow(input, vocabularySize);

        Tensor<T> flatOutput;
        // Standard embedding lookup for integer token indices.
        // AiDotNet#1331: route lookups through the float-indices overload
        // when a graph-mode lazy trace is active. The legacy
        // <c>TensorEmbeddingLookup&lt;T, int&gt;</c> snapshots the int
        // indices array INSIDE the lazy node — but the int array is built
        // here in C# from a flat <c>Tensor&lt;int&gt;</c> instance that is
        // NOT a leaf of the lazy graph. On subsequent <c>plan.Step()</c>
        // calls the snapshot never refreshes, so every replay gathers
        // rows for the FIRST batch's tokens regardless of the current
        // float input data — the model converges to a uniform output
        // (loss ≈ ln(V)) instead of learning the input→target mapping.
        //
        // The float-indices variant captures the float input tensor by
        // reference and converts to int at execute time. The caller can
        // update the float input's data in place between Step() calls and
        // the next replay sees the new indices.
        //
        // Tape-awareness: same backward as the int path (dL/dE scatter
        // back to the embedding table), but the scatter uses fresh
        // indices read at backward time too — keeping forward gather and
        // backward scatter aligned on every Step.
        if (AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive)
        {
            flatOutput = Engine.TensorEmbeddingLookupFromFloatIndices(_embeddingTensor, input);
        }
        else
        {
            // Eager / inference fast path: direct row gather avoids the
            // O(N*V) one-hot matmul. Pre-issue #1208 / pre-#1331 default.
            int totalIndices = input.Length;
            var flatIndices = new Tensor<int>([totalIndices]);
            for (int i = 0; i < totalIndices; i++)
            {
                int index = Convert.ToInt32(NumOps.ToDouble(input.Data.Span[i]));
                flatIndices[i] = index;
            }
            flatOutput = Engine.TensorEmbeddingLookup<T, int>(_embeddingTensor, flatIndices);
        }

        // Calculate output shape
        int[] outputShape;
        if (input.Rank == 1)
        {
            // [seqLen] -> [seqLen, embeddingDim]
            outputShape = [input.Shape[0], embeddingDim];
        }
        else if (input.Rank == 2)
        {
            // [batch, seqLen] -> [batch, seqLen, embeddingDim]
            outputShape = [input.Shape[0], input.Shape[1], embeddingDim];
        }
        else if (input.Rank == 3 && input.Shape[2] == 1)
        {
            // Legacy format [batch, seqLen, 1] -> [batch, seqLen, embeddingDim]
            outputShape = [input.Shape[0], input.Shape[1], embeddingDim];
        }
        else
        {
            // Generic case for any rank: input shape [...] -> [..., embeddingDim]
            // This matches PyTorch's nn.Embedding behavior which accepts any shape
            // and appends the embedding dimension to the output
            outputShape = new int[input.Rank + 1];
            for (int i = 0; i < input.Rank; i++)
            {
                outputShape[i] = input.Shape[i];
            }
            outputShape[^1] = embeddingDim;
        }

        var reshaped = Engine.Reshape(flatOutput, outputShape);

        // Vaswani §3.4 embedding scale (opt-in, token-ID mode only). TapeMultiplyScalar
        // records the op on the autodiff tape, so the embedding-table gradient is scaled
        // automatically during backprop (this layer is fully tape-based — no custom
        // Backward override to maintain). In eager inference (NoGradScope) it is a plain
        // value multiply.
        if (ScaleBySqrtDimension)
        {
            T sqrtDim = NumOps.Sqrt(NumOps.FromDouble(embeddingDim));
            reshaped = AiDotNet.Helpers.TensorTapeOps.TapeMultiplyScalar(Engine, reshaped, sqrtDim);
        }

        return reshaped;
    }

    /// <summary>
    /// Performs the forward pass of the embedding layer on GPU.
    /// </summary>
    /// <param name="inputs">The GPU-resident input tensor(s) containing token indices.</param>
    /// <returns>A GPU-resident tensor containing the embedding vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when no inputs are provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU engine is not available.</exception>
    /// <remarks>
    /// <para>
    /// This method performs embedding lookup entirely on GPU, keeping the output on GPU
    /// for subsequent GPU-accelerated operations. This eliminates CPU-GPU data transfers
    /// for intermediate results in deep networks.
    /// </para>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of embedding lookup.
    /// Instead of moving data between CPU and GPU, all computation stays on the GPU,
    /// making it much faster for large vocabularies and batch sizes.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        // Materialize lazy embedding tensor before GPU lookups.
        EnsureEmbeddingInitialized();

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];
        int embeddingDim = _embeddingTensor.Shape[1];
        int vocabularySize = _embeddingTensor.Shape[0];

        // Download input to CPU to convert to integer indices
        // (Indices are typically small, so this is acceptable)
        var inputTensor = input;

        // Store for potential backward pass (only in training mode)
        if (IsTrainingMode)
        {
            _lastInput = inputTensor;
            _originalInputShape = inputTensor._shape;
        }

        // Same invariant as the CPU path: this layer only accepts token indices.
        ValidateIndicesOrThrow(inputTensor, vocabularySize);

        // Standard embedding lookup for integer token indices
        int totalIndices = inputTensor.Length;
        var flatIndices = new Tensor<int>([totalIndices]);
        for (int i = 0; i < totalIndices; i++)
        {
            flatIndices[i] = Convert.ToInt32(NumOps.ToDouble(inputTensor.Data.Span[i]));
        }

        // Cache indices for backward pass
        if (IsTrainingMode)
        {
            _lastIndicesForGpu = flatIndices;
            _lastInputGpuShape = inputTensor._shape;
        }

        // Perform GPU embedding lookup - keeps result on GPU
        var gpuOutput = gpuEngine.EmbeddingLookupGpu(_embeddingTensor, flatIndices);

        // Vaswani §3.4 embedding scale — apply on the GPU path too so behavior is
        // backend-independent (the CPU Forward above applies the identical scale). Routed
        // through TapeMultiplyScalar, which dispatches to the active engine (GPU here) and
        // records the op on the autodiff tape, so the embedding-table gradient is scaled the
        // same way regardless of device. Without this the token-index GPU path returned
        // UNSCALED embeddings, making a model trained/served on GPU diverge from CPU.
        if (ScaleBySqrtDimension)
        {
            T sqrtDim = NumOps.Sqrt(NumOps.FromDouble(embeddingDim));
            gpuOutput = AiDotNet.Helpers.TensorTapeOps.TapeMultiplyScalar(Engine, gpuOutput, sqrtDim);
        }

        return gpuOutput;
    }

    /// <summary>
    /// Confirms the input really holds token indices, and explains the fix when it does not.
    /// </summary>
    /// <remarks>
    /// Validation runs on every call because token values can change while the shape remains constant.
    /// This replaces a silent mode switch or backend-specific gather failure with one actionable error.
    /// </remarks>
    private void ValidateIndicesOrThrow(Tensor<T> input, int vocabularySize)
    {
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input.Data.Span[i]);
            bool bad = double.IsNaN(val) || double.IsInfinity(val);
            int intVal = bad ? 0 : (int)val;
            if (bad || Math.Abs(val - intVal) > 1e-6 || intVal < 0 || intVal >= vocabularySize)
            {
                throw new ArgumentException(
                    $"EmbeddingLayer requires token indices, but element {i} is {val}, which is not in "
                    + $"[0, {vocabularySize}). Use DenseLayer for continuous feature projection, or a "
                    + "composite embedding layer when word, position, and token-type embeddings must be "
                    + "looked up in parallel and combined.",
                    nameof(input));
            }
        }
    }

    /// <summary>
    /// Updates the embedding matrix using the calculated gradients and the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the embedding matrix based on the gradients calculated during the backward pass.
    /// Only the embeddings for tokens that appeared in the input during the forward pass will be updated.
    /// The learning rate determines the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method actually changes the embeddings to improve future predictions.
    /// 
    /// After figuring out how each embedding should change:
    /// - The embedding matrix is updated by subtracting the gradients
    /// - Each value is adjusted proportionally to its gradient
    /// - The learning rate controls how big these adjustments are
    /// 
    /// For example:
    /// - If embedding for token #5 has a gradient of [0.1, -0.2, 0.3]
    /// - With learning rate of 0.01
    /// - The embedding will change by [-0.001, 0.002, -0.003]
    /// 
    /// Only embeddings for tokens that appeared in the recent input batch will be updated.
    /// Frequently used tokens will get more updates over time.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_embeddingGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        var scaledGradient = Engine.TensorMultiplyScalar(_embeddingGradient, learningRate);
        _embeddingTensor = Engine.TensorSubtract(_embeddingTensor, scaledGradient);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_embeddingTensor);
    }

    /// <summary>
    /// Computes the auxiliary loss for the EmbeddingLayer, which is embedding regularization.
    /// </summary>
    /// <returns>The embedding regularization loss value.</returns>
    /// <remarks>
    /// <para>
    /// Embedding regularization prevents embedding vectors from becoming too large or too similar,
    /// which can lead to overfitting. It applies L2 regularization on the embedding weights:
    /// Loss = (1/2) * Σ||embedding||²
    ///
    /// This regularization:
    /// - Prevents embeddings from growing unboundedly
    /// - Encourages smaller, more generalizable embedding values
    /// - Helps prevent overfitting to the training data
    /// - Promotes diverse embedding representations
    /// </para>
    /// <para><b>For Beginners:</b> This calculates a penalty for embeddings that become too large.
    ///
    /// Embedding regularization:
    /// - Measures how large the embedding vectors are
    /// - Penalizes very large embedding values
    /// - Encourages the model to use smaller, more manageable numbers
    /// - Prevents the model from memorizing training data too closely
    ///
    /// Why this is important:
    /// - Large embedding values can indicate overfitting
    /// - Regularization promotes better generalization to new data
    /// - Keeps embedding vectors at reasonable scales
    /// - Prevents embeddings from collapsing or diverging
    ///
    /// Think of it like a referee that prevents embeddings from becoming too extreme,
    /// keeping them in a reasonable range for better model performance.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            // Reset cached loss to avoid stale diagnostics
            _lastEmbeddingRegularizationLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];

        // Compute L2 regularization on embedding weights using Engine operation: (1/2) * Σ||embedding||²
        T sumSquaredNorms = Engine.TensorSumOfSquares(_embeddingTensor);

        // Average over all embedding values and scale by 0.5 (standard L2 regularization)
        int totalElements = vocabSize * embeddingDim;
        T regularizationLoss = NumericalStabilityHelper.SafeDiv(sumSquaredNorms, NumOps.FromDouble(totalElements * 2));

        // Store unweighted loss for diagnostics
        _lastEmbeddingRegularizationLoss = regularizationLoss;

        // Return weighted auxiliary loss
        return NumOps.Multiply(AuxiliaryLossWeight, regularizationLoss);
    }

    /// <summary>
    /// Gets diagnostic information about the embedding regularization.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about embedding health.</returns>
    /// <remarks>
    /// <para>
    /// This method provides insights into embedding behavior, including:
    /// - Embedding regularization loss
    /// - Average embedding magnitude
    /// - Regularization weight
    /// </para>
    /// <para><b>For Beginners:</b> This gives you information to monitor embedding quality.
    ///
    /// The diagnostics include:
    /// - Embedding Regularization Loss: Measure of embedding magnitude
    /// - Regularization Weight: How much the penalty influences training
    /// - Average Embedding Magnitude: Typical size of embedding vectors
    /// - Use Auxiliary Loss: Whether regularization is enabled
    ///
    /// These values help you:
    /// - Monitor if embeddings are growing too large
    /// - Detect potential overfitting in embedding layer
    /// - Tune the regularization weight
    /// - Ensure embeddings remain at reasonable scales
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        string regLossStr = Convert.ToString(_lastEmbeddingRegularizationLoss) ?? "0";
        string weightStr = Convert.ToString(AuxiliaryLossWeight) ?? "0.0001";

        var diagnostics = new Dictionary<string, string>
        {
            { "EmbeddingRegularizationLoss", regLossStr },
            { "RegularizationWeight", weightStr },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
        };

        int vocabSize = _embeddingTensor.Shape[0];
        int embeddingDim = _embeddingTensor.Shape[1];

        // Calculate average embedding magnitude using Engine operations
        // Sum of squares gives us Σ||embedding_i||² across all embeddings
        T totalSumOfSquares = Engine.TensorSumOfSquares(_embeddingTensor);

        // For average magnitude: sqrt(sum_of_squares / num_elements) * num_rows / num_rows
        // Simplified: average magnitude ≈ sqrt(total_sum_of_squares / total_elements) * sqrt(embedding_dim)
        // This is an approximation, but avoids per-row loops
        if (vocabSize > 0)
        {
            T avgSquaredMagnitude = NumericalStabilityHelper.SafeDiv(totalSumOfSquares, NumOps.FromDouble(vocabSize));
            T avgMagnitude = NumOps.Sqrt(avgSquaredMagnitude);
            string avgMagStr = Convert.ToString(avgMagnitude) ?? "0";
            diagnostics["AverageEmbeddingMagnitude"] = avgMagStr;
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
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input and embedding gradients
    /// from previous forward and backward passes. This is useful when starting to process a new batch of
    /// data or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    ///
    /// When resetting the state:
    /// - The saved input token IDs are cleared
    /// - The calculated gradients are cleared
    /// - The layer forgets previous calculations it performed
    ///
    /// This is typically called:
    /// - Between training batches to free up memory
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    ///
    /// It doesn't affect the learned embeddings themselves, just the temporary
    /// working data used during computation.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        // No backward has run yet: preserve the framework-wide zero-gradient query contract.
        if (_embeddingGradient == null)
            return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));

        // Bulk copy from contiguous tensor storage — avoids ToArray() double-copy
        return Vector<T>.FromMemory(_embeddingGradient.Data);
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _embeddingGradient = null;
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _embeddingGradient = null;

        // Clear GPU-related cached data
        _lastInputGpu?.Dispose();
        _lastInputGpu = null;
        _lastInputGpuShape = null;
        _lastIndicesForGpu = null;
    }
}
