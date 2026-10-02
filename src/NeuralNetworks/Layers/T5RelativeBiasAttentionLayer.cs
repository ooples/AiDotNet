using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// T5-style multi-head self-attention with learned relative position bias
/// (Raffel et al., "Exploring the Limits of Transfer Learning with a
/// Unified Text-to-Text Transformer", JMLR 2020).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Two paper-faithful deviations from the standard MultiHeadAttention layer:
/// </para>
/// <list type="number">
/// <item>
///   <b>No projection biases.</b> T5's Q/K/V/O projections are bias-free
///   (Raffel 2020 §2.1, "we use a simplified layer normalization where the
///   activations are only rescaled and no additive bias is applied"; the same
///   no-bias convention extends to attention projections in the reference t5x
///   implementation).
/// </item>
/// <item>
///   <b>Learned relative position bias added pre-softmax.</b> A bias tensor
///   of shape <c>[numHeads, seqQ, seqK]</c> is added to the raw attention
///   logits before softmax. Each entry comes from a small learnable table
///   <c>[numBuckets, numHeads]</c> indexed by the bucketed relative position
///   (i - j). The bucketing scheme is the canonical T5 logarithmic scheme:
///   half the buckets are reserved for exact small distances, the rest log-
///   spaced up to <c>maxDistance</c>. For bidirectional (encoder) attention,
///   half of the table covers negative offsets and half covers positive
///   offsets.
/// </item>
/// </list>
/// <para>
/// <b>Shared-bias convention:</b> The original T5 paper shares ONE relative-
/// position bias table across all encoder layers (Raffel 2020 §2.1 footnote 5,
/// "the relative position embedding is shared across all layers but each head
/// has its own embedding"). The constructor accepts an optional
/// <c>sharedRelativeBiasTable</c>; when supplied, this layer reuses it instead
/// of allocating its own. The <c>LayerHelper.CreateDefaultT5TextLayers</c>
/// factory wires one shared table through every T5 attention layer in the
/// stack — that is the paper-canonical configuration. Standalone construction
/// (e.g. unit tests) gives each layer its own bias table, which is the
/// "common-but-non-canonical" HuggingFace T5 default.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard attention layers learn position information
/// only through fixed sinusoidal patterns added to the input. T5 instead
/// learns directly how much to bias attention between any two positions —
/// a richer, fully-trained position signal that has been a major contributor
/// to T5's strong empirical performance.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 2")]
// Shape-preserving, so the generator derives Same on every axis and no OutputAxesFor is written here.
// From the tail of ForwardTraced: rank 3 returns output3D, shaped [batchSize, seqLen, _hiddenSize], and
// any other rank is reshaped to outShape, which copies input.Shape for every leading axis and then sets
// seqLen and _hiddenSize. Self-attention preserves the sequence length by construction, and the feature
// width comes back unchanged because _oWeights projects hidden -> hidden - the layer additionally
// REFUSES any input whose trailing axis is not _hiddenSize, so Same(Features) is not merely observed.
//
// Deliberately NOT [ElementWiseShape]: that shorthand claims preservation at any rank, and this layer
// throws for rank < 2 ("expects input of rank >= 2"). BatchOptional covers exactly the two forms the
// method's own comment names - "Accept [batch, seq, hidden] or [seq, hidden]". Higher ranks run too
// (leading axes are folded into batchSize and restored), but each extra leading axis would need a
// distinct role for a relation to refer to it, and there is no second batch-like role to give it.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class T5RelativeBiasAttentionLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _innerSize;
    private readonly int _numBuckets;
    private readonly int _maxDistance;
    private readonly bool _bidirectional;
    private readonly bool _ownsBiasTable;
    private readonly Random _rng;

    // T5 has NO biases on Q/K/V/O projections (Raffel 2020 §2.1).
    // Allocated on first use (EnsureWeightsAllocated): a conditioner now builds its whole stack at
    // construction, and eager [hidden, hidden] x 4 per layer made T5-XXL (24 x 4096^2 x 4) run out of
    // memory before any forward. The shapes are declared so ParameterCount needs no allocation.
    [TrainableParameter(Role = PersistentTensorRole.Weights, Shape = "_hiddenSize, _innerSize")]
    private Tensor<T> _qWeights;

    [TrainableParameter(Role = PersistentTensorRole.Weights, Shape = "_hiddenSize, _innerSize")]
    private Tensor<T> _kWeights;

    [TrainableParameter(Role = PersistentTensorRole.Weights, Shape = "_hiddenSize, _innerSize")]
    private Tensor<T> _vWeights;

    [TrainableParameter(Role = PersistentTensorRole.Weights, Shape = "_innerSize, _hiddenSize")]
    private Tensor<T> _oWeights;

    /// <summary>True once Q/K/V/O hold real, initialized storage.</summary>
    private bool _projectionsAllocated;

    // Relative position bias table. [numBuckets, numHeads].
    // Marked trainable only if this layer owns it; otherwise the owning
    // layer registers it instead so the optimizer sees exactly one copy.
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _relativeBiasTable;

    // Cached bucket-index lookup table for the current sequence length.
    // Recomputed only when seqLen changes; positions are fixed so this
    // is pure shape state, NOT a trainable parameter.
    private int _cachedSeqLen = -1;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<int>? _bucketIndices;

    [Scratch]
    private Tensor<T>? _qGradient;
    [Scratch]
    private Tensor<T>? _kGradient;
    [Scratch]
    private Tensor<T>? _vGradient;
    [Scratch]
    private Tensor<T>? _oGradient;
    [Scratch]
    private Tensor<T>? _biasTableGradient;

    private readonly bool _usesExternalPositionBias;

    /// <summary>The bias the enclosing stack computed for the current forward pass.</summary>
    [Scratch]
    private Tensor<T>? _externalPositionBias;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the relative bias table for testing and inspection.
    /// </summary>
    public Tensor<T> GetRelativeBiasTable() => _relativeBiasTable;

    /// <summary>
    /// Returns whether this layer owns its bias table (vs. sharing one
    /// supplied at construction time).
    /// </summary>
    public bool OwnsRelativeBiasTable => _ownsBiasTable;

    /// <summary>
    /// Returns layer-specific metadata required for cloning / serialisation.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HiddenSize"] = _hiddenSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumBuckets"] = _numBuckets.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MaxDistance"] = _maxDistance.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Bidirectional"] = _bidirectional.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["KeyValueDim"] = _headDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <summary>
    /// Initialises a new T5-style relative-bias self-attention layer.
    /// </summary>
    /// <param name="hiddenSize">
    /// Model hidden size (the input/output feature dimension). Must be
    /// divisible by <paramref name="numHeads"/>.
    /// </param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numBuckets">
    /// Number of relative-position buckets. Paper default for the encoder
    /// is 32. For unidirectional (decoder) attention this would typically
    /// be 16, but the constructor preserves the caller's value — the
    /// bucketing function is parameterised by both numBuckets and
    /// <paramref name="bidirectional"/>.
    /// </param>
    /// <param name="maxDistance">
    /// Maximum relative-position distance covered by the log-spaced bucket
    /// region. Beyond this distance, positions clip to the last bucket.
    /// Paper default: 128.
    /// </param>
    /// <param name="bidirectional">
    /// True for encoder self-attention (queries can attend to keys at any
    /// position); false for decoder causal self-attention (queries only
    /// attend to past keys). It sets the bucket layout and, when false, also masks future keys.
    /// </param>
    /// <param name="seed">Optional RNG seed for deterministic initialisation.</param>
    /// <param name="sharedRelativeBiasTable">
    /// When non-null, the layer adopts this caller-owned bias table instead
    /// of allocating its own. The factory <c>LayerHelper.CreateDefaultT5TextLayers</c>
    /// uses this to wire one paper-canonical shared table through every
    /// attention layer in the stack.
    /// </param>
    /// <param name="keyValueDim">Width of each head (T5 <c>d_kv</c>). When null, each head is <c>hiddenSize / numHeads</c>.</param>
    public T5RelativeBiasAttentionLayer(
        int hiddenSize,
        int numHeads,
        int numBuckets = 32,
        int maxDistance = 128,
        bool bidirectional = true,
        int? seed = null,
        Tensor<T>? sharedRelativeBiasTable = null,
        bool usesExternalPositionBias = false,
        int? keyValueDim = null)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), "hiddenSize must be positive.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");
        if (keyValueDim is { } kv && kv <= 0)
            throw new ArgumentOutOfRangeException(nameof(keyValueDim), "keyValueDim must be positive.");
        if (keyValueDim is null && hiddenSize % numHeads != 0)
            throw new ArgumentException(
                $"hiddenSize ({hiddenSize}) must be divisible by numHeads ({numHeads}).",
                nameof(numHeads));
        if (numBuckets <= 0)
            throw new ArgumentOutOfRangeException(nameof(numBuckets), "numBuckets must be positive.");
        if (maxDistance <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDistance), "maxDistance must be positive.");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        // T5 sizes heads independently of the model width (d_kv); without it each head is hidden / heads wide.
        _headDim = keyValueDim ?? hiddenSize / numHeads;
        _innerSize = numHeads * _headDim;
        _numBuckets = numBuckets;
        _maxDistance = maxDistance;
        _bidirectional = bidirectional;
        _rng = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : LayerInitializationSeedScope.NextRandom();

        // Q/K/V/O projections are [hiddenSize, hiddenSize] placeholders until first use; see
        // EnsureWeightsAllocated. The bias table is small and may be shared, so it is built here.
        _qWeights = new Tensor<T>([0, 0]);
        _kWeights = new Tensor<T>([0, 0]);
        _vWeights = new Tensor<T>([0, 0]);
        _oWeights = new Tensor<T>([0, 0]);

        _usesExternalPositionBias = usesExternalPositionBias;
        if (usesExternalPositionBias)
        {
            if (sharedRelativeBiasTable is not null)
                throw new ArgumentException(
                    "A layer that receives its position bias from its stack cannot also hold a shared table.",
                    nameof(sharedRelativeBiasTable));

            // The enclosing T5EncoderStack owns the one table and hands each block the bias it computed,
            // so this layer holds no table and no reference to one: nothing here can come apart on clone.
            _relativeBiasTable = new Tensor<T>([0, 0]);
            _ownsBiasTable = false;
        }
        else if (sharedRelativeBiasTable is not null)
        {
            // Validate the shared table matches this layer's geometry. A
            // mismatched shape would silently break attention scoring.
            if (sharedRelativeBiasTable.Shape.Length != 2 ||
                sharedRelativeBiasTable.Shape[0] != numBuckets ||
                sharedRelativeBiasTable.Shape[1] != numHeads)
            {
                throw new ArgumentException(
                    $"sharedRelativeBiasTable must have shape [{numBuckets}, {numHeads}], " +
                    $"got [{string.Join(", ", sharedRelativeBiasTable.Shape)}].",
                    nameof(sharedRelativeBiasTable));
            }
            _relativeBiasTable = sharedRelativeBiasTable;
            _ownsBiasTable = false;
            // Do NOT register as trainable — the owning layer does that, so
            // the optimizer doesn't see the same parameter twice (which would
            // produce 2× the intended gradient step).
        }
        else
        {
            // HuggingFace T5 initialises the bias table with mean=0,
            // std=hidden_size**-0.5. We use the same convention so that
            // single-layer-owned (non-shared) configurations remain on the
            // documented initialisation manifold.
            double biasStd = 1.0 / Math.Sqrt(hiddenSize);
            _relativeBiasTable = SampleNormalTensor(new[] { numBuckets, numHeads }, std: biasStd);
            _ownsBiasTable = true;
            // Registered with the projections in EnsureWeightsAllocated, so the registered order
            // stays Q, K, V, O, bias - the order the generated parameter surface declares.
        }
    }

    /// <inheritdoc />
    protected override bool ParametersAreConstructionSized => true;

    /// <inheritdoc />
    protected override void EnsureInitialized()
    {
        EnsureWeightsAllocated();
        base.EnsureInitialized();
    }

    /// <inheritdoc />
    internal override bool TryDeclareShape()
    {
        EnsureWeightsAllocated();
        return true;
    }

    /// <summary>
    /// Allocates, initializes and registers Q/K/V/O once. Idempotent, and it keeps weights a
    /// clone or checkpoint restore already installed rather than re-initializing over them.
    /// </summary>
    private void EnsureWeightsAllocated()
    {
        if (_projectionsAllocated) return;

        lock (InitializationLock)
        {
            if (_projectionsAllocated) return;

            if (WeightsAlreadyAllocated(_qWeights, _hiddenSize, _innerSize)
                && WeightsAlreadyAllocated(_kWeights, _hiddenSize, _innerSize)
                && WeightsAlreadyAllocated(_vWeights, _hiddenSize, _innerSize)
                && WeightsAlreadyAllocated(_oWeights, _innerSize, _hiddenSize))
            {
                // The restore path (SetTrainableParameters) already registered them.
                _projectionsAllocated = true;
                return;
            }

            // T5's initialisation (HF T5PreTrainedModel._init_weights, factor 1): q ~ N(0, (d * d_kv)^-1/2),
            // k and v ~ N(0, d^-1/2), o ~ N(0, (heads * d_kv)^-1/2). The query's extra d_kv^-1/2 is the score scaling
            // T5 leaves out of the forward. Drawn from the layer's RNG after the bias table.
            _qWeights = SampleNormalTensor(new[] { _hiddenSize, _innerSize }, Math.Pow((double)_hiddenSize * _headDim, -0.5));
            _kWeights = SampleNormalTensor(new[] { _hiddenSize, _innerSize }, Math.Pow(_hiddenSize, -0.5));
            _vWeights = SampleNormalTensor(new[] { _hiddenSize, _innerSize }, Math.Pow(_hiddenSize, -0.5));
            _oWeights = SampleNormalTensor(new[] { _innerSize, _hiddenSize }, Math.Pow(_innerSize, -0.5));

            RegisterTrainableParameter(_qWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_kWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_vWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_oWeights, PersistentTensorRole.Weights);
            // Do NOT register a shared table: the owning layer does, so the optimizer sees it once.
            if (_ownsBiasTable)
                RegisterTrainableParameter(_relativeBiasTable, PersistentTensorRole.Weights);

            _projectionsAllocated = true;
        }
    }


    /// <summary>
    /// Samples a normal-distributed [shape...] tensor with mean 0 and the
    /// requested standard deviation via Box-Muller.
    /// </summary>
    private Tensor<T> SampleNormalTensor(int[] shape, double std)
    {
        var t = new Tensor<T>(shape);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = 1.0 - _rng.NextDouble();
            double n = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            span[i] = NumOps.FromDouble(n * std);
        }
        return t;
    }

    /// <summary>
    /// Self-attention over <c>[batch, seq, hidden]</c> or <c>[seq, hidden]</c>. Q/K/V are projected, the T5
    /// relative bias is added to the unscaled scores, and the result goes through the output projection. A
    /// unidirectional (decoder) layer also masks future keys. Every op is an Engine op, so the tape records
    /// the graph.
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureWeightsAllocated();

        // Accept [batch, seq, hidden] or [seq, hidden]. Flatten leading dims to [batch, seq, hidden].
        int rank = input.Shape.Length;
        if (rank < 2)
            throw new ArgumentException(
                $"T5RelativeBiasAttentionLayer expects input of rank >= 2, got rank {rank}.",
                nameof(input));

        int seqLen = input.Shape[rank - 2];
        int featureDim = input.Shape[rank - 1];
        if (featureDim != _hiddenSize)
            throw new ArgumentException(
                $"Input feature dim ({featureDim}) does not match layer hiddenSize ({_hiddenSize}).",
                nameof(input));

        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++) batchSize *= input.Shape[i];

        var input3D = rank == 3
            ? input
            : Engine.Reshape(input, new[] { batchSize, seqLen, _hiddenSize });

        var bias = _usesExternalPositionBias
            ? ExternalPositionBias(seqLen)
            : BuildT5RelativeBias(seqLen);
        var output3D = Attend(input3D, input3D, bias, causal: !_bidirectional);

        if (rank == 3) return output3D;
        var outShape = new int[rank];
        for (int i = 0; i < rank - 2; i++) outShape[i] = input.Shape[i];
        outShape[rank - 2] = seqLen;
        outShape[rank - 1] = _hiddenSize;
        return Engine.Reshape(output3D, outShape);
    }

    /// <summary>
    /// Attends from <paramref name="x"/> <c>[Sq, hidden]</c> to <paramref name="memory"/> <c>[Sk, hidden]</c>,
    /// which is the same tensor for self-attention and the encoder output for cross-attention.
    /// <paramref name="positionBias"/> <c>[heads, Sq, Sk]</c> is a bias the caller computed (null for none).
    /// <paramref name="causal"/> masks keys after each query.
    /// </summary>
    /// <remarks>
    /// Encoder-decoders whose biases are richer than one 1-D table use this path; UDOP's summed 1-D and 2-D
    /// layout biases are an example. The layer's own table is not consulted here.
    /// </remarks>
    internal Tensor<T> Forward(Tensor<T> x, Tensor<T> memory, Tensor<T>? positionBias, bool causal)
    {
        EnsureWeightsAllocated();
        if (x.Rank != 2 || x.Shape[1] != _hiddenSize || memory.Rank != 2 || memory.Shape[1] != _hiddenSize)
            throw new ArgumentException(
                $"T5RelativeBiasAttentionLayer.Forward expects [S, {_hiddenSize}] queries and memory; got [{string.Join(", ", x.Shape.ToArray())}] and [{string.Join(", ", memory.Shape.ToArray())}].",
                nameof(x));
        int sq = x.Shape[0], sk = memory.Shape[0];
        if (positionBias is not null
            && (positionBias.Rank != 3 || positionBias.Shape[0] != _numHeads || positionBias.Shape[1] != sq || positionBias.Shape[2] != sk))
            throw new ArgumentException($"The position bias must be [{_numHeads}, {sq}, {sk}].", nameof(positionBias));
        var output = Attend(
            Engine.Reshape(x, new[] { 1, sq, _hiddenSize }),
            Engine.Reshape(memory, new[] { 1, sk, _hiddenSize }),
            positionBias, causal);
        return Engine.Reshape(output, new[] { sq, _hiddenSize });
    }

    /// <summary>
    /// T5 attention core over <c>[B, Sq, hidden]</c> queries and <c>[B, Sk, hidden]</c> memory. The scores are
    /// <c>q k^T + bias</c> with NO <c>1/sqrt(d_kv)</c>: T5 folds that factor into the query initialisation
    /// (Raffel et al. 2020; HF <c>T5Attention</c>, <c>scaling = 1.0</c>).
    /// </summary>
    private Tensor<T> Attend(Tensor<T> query3D, Tensor<T> memory3D, Tensor<T>? bias, bool causal)
    {
        int batchSize = query3D.Shape[0], sq = query3D.Shape[1], sk = memory3D.Shape[1];
        var queries = Engine.Reshape(query3D, new[] { batchSize * sq, _hiddenSize });
        var keysIn = Engine.Reshape(memory3D, new[] { batchSize * sk, _hiddenSize });

        // Project to [B, S, H, d_kv], then permute to [B, H, S, d_kv].
        var q = Engine.TensorPermute(Engine.Reshape(Engine.TensorMatMul(queries, _qWeights), new[] { batchSize, sq, _numHeads, _headDim }), new[] { 0, 2, 1, 3 });
        var k = Engine.TensorPermute(Engine.Reshape(Engine.TensorMatMul(keysIn, _kWeights), new[] { batchSize, sk, _numHeads, _headDim }), new[] { 0, 2, 1, 3 });
        var v = Engine.TensorPermute(Engine.Reshape(Engine.TensorMatMul(keysIn, _vWeights), new[] { batchSize, sk, _numHeads, _headDim }), new[] { 0, 2, 1, 3 });

        // Manual composition through tape-tracked Engine ops: FlashAttention<T>.Forward fills its output by
        // scalar indexing, which the tape cannot differentiate, and the fused SDPA op takes no additive bias.
        var scores = Engine.TensorMatMul(q, Engine.TensorPermute(k, new[] { 0, 1, 3, 2 }));      // [B, H, Sq, Sk]
        if (bias is not null) scores = Engine.TensorAdd(scores, bias);
        if (causal)
        {
            var mask = new Tensor<T>(new[] { _numHeads, sq, sk });
            T blocked = NumOps.FromDouble(-1e9);
            int shift = sk - sq; // queries are the last sq positions of the keys
            for (int h = 0; h < _numHeads; h++)
                for (int i = 0; i < sq; i++)
                    for (int j = i + shift + 1; j < sk; j++) mask[h, i, j] = blocked;
            scores = Engine.TensorAdd(scores, mask);
        }
        var context = Engine.TensorMatMul(Engine.TensorSoftmax(scores, axis: 3), v);              // [B, H, Sq, d_kv]

        // [B, H, Sq, d_kv] -> [B, Sq, H * d_kv] -> output projection to hidden.
        var merged = Engine.Reshape(Engine.TensorPermute(context, new[] { 0, 2, 1, 3 }), new[] { batchSize * sq, _innerSize });
        return Engine.Reshape(Engine.TensorMatMul(merged, _oWeights), new[] { batchSize, sq, _hiddenSize });
    }

    /// <summary>
    /// Builds the T5 relative position bias tensor of shape
    /// <c>[numHeads, seqLen, seqLen]</c> by looking up the trainable bias
    /// table with bucketed relative-position indices. The lookup goes
    /// through <see cref="IEngine.TensorEmbeddingLookup{T,T2}"/> so gradients
    /// flow back into <see cref="_relativeBiasTable"/> on backward.
    /// </summary>
    private Tensor<T> ExternalPositionBias(int seqLen)
    {
        var bias = _externalPositionBias
            ?? throw new InvalidOperationException(
                "This T5 attention layer takes its relative position bias from its T5EncoderStack; run it " +
                "through the stack, which computes the shared bias once per forward pass.");
        if (bias.Shape.Length != 3 || bias.Shape[0] != _numHeads || bias.Shape[1] != seqLen || bias.Shape[2] != seqLen)
            throw new ArgumentException(
                $"The supplied position bias must be [{_numHeads}, {seqLen}, {seqLen}], got [{string.Join(", ", bias.Shape.ToArray())}].");
        return bias;
    }

    /// <summary>Whether this layer receives its relative position bias from its stack.</summary>
    public bool UsesExternalPositionBias => _usesExternalPositionBias;

    /// <summary>Supplies the bias for the next forward pass; the stack clears it afterwards.</summary>
    internal void SetExternalPositionBias(Tensor<T>? bias) => _externalPositionBias = bias;

    private Tensor<T> BuildT5RelativeBias(int seqLen)
    {
        // Bucket-index matrix depends only on seqLen (and the layer's
        // bucketing config), so cache it. The matrix is integer-valued
        // and non-trainable.
        if (_cachedSeqLen != seqLen)
        {
            _bucketIndices = ComputeBucketIndices(seqLen);
            _cachedSeqLen = seqLen;
        }

        // [seqLen, seqLen] -> lookup against [numBuckets, numHeads] -> [seqLen, seqLen, numHeads]
        var looked = Engine.TensorEmbeddingLookup<T, int>(_relativeBiasTable, _bucketIndices!);
        // Permute to [numHeads, seqLen, seqLen] so it broadcasts against
        // the [B, numHeads, seqLen, seqLen] attention scores inside
        // FlashAttention.
        return Engine.TensorPermute(looked, new[] { 2, 0, 1 });
    }

    /// <summary>
    /// Computes the bucket index for every (queryPos, keyPos) pair in a
    /// sequence of length <paramref name="seqLen"/>, following the T5
    /// reference implementation (mesh-tensorflow's
    /// <c>_relative_position_bucket</c>).
    /// </summary>
    private Tensor<int> ComputeBucketIndices(int seqLen)
    {
        var idx = new Tensor<int>(new[] { seqLen, seqLen });
        for (int qPos = 0; qPos < seqLen; qPos++)
        {
            for (int kPos = 0; kPos < seqLen; kPos++)
            {
                int relativePosition = kPos - qPos;
                idx[qPos, kPos] = RelativePositionBucket(
                    relativePosition, _bidirectional, _numBuckets, _maxDistance);
            }
        }
        return idx;
    }

    /// <summary>
    /// Canonical T5 relative-position bucketing (Raffel 2020; mesh-tensorflow
    /// reference <c>_relative_position_bucket</c>).
    /// </summary>
    /// <remarks>
    /// For bidirectional attention, half the buckets cover negative
    /// offsets (keys earlier than the query) and half cover non-negative
    /// offsets. Within each half, the first <c>numBuckets/4</c> buckets
    /// hold exact small distances, and the remainder hold log-spaced
    /// distances up to <paramref name="maxDistance"/>. For unidirectional
    /// attention all buckets cover the non-negative range.
    /// </remarks>
    internal static int RelativePositionBucket(
        int relativePosition, bool bidirectional, int numBuckets, int maxDistance)
    {
        int ret = 0;
        int n = -relativePosition;

        if (bidirectional)
        {
            numBuckets /= 2;
            if (n < 0)
            {
                ret += numBuckets;
            }
            n = Math.Abs(n);
        }
        else
        {
            n = Math.Max(n, 0);
        }

        int maxExact = numBuckets / 2;
        if (n < maxExact)
        {
            ret += n;
        }
        else
        {
            double scale = Math.Log((double)n / maxExact) / Math.Log((double)maxDistance / maxExact);
            int valIfLarge = maxExact + (int)(scale * (numBuckets - maxExact));
            valIfLarge = Math.Min(valIfLarge, numBuckets - 1);
            ret += valIfLarge;
        }

        return ret;
    }

    private static void WriteInto(Tensor<T> dest, Vector<T> src, ref int offset)
    {
        var span = dest.Data.Span;
        for (int i = 0; i < dest.Length; i++) span[i] = src[offset + i];
        offset += (int)dest.Length;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        EnsureWeightsAllocated();
        if (_qGradient is null)
            return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));

        var qG = _qGradient.ToVector();
        var kG = (_kGradient ?? new Tensor<T>(_kWeights._shape)).ToVector();
        var vG = (_vGradient ?? new Tensor<T>(_vWeights._shape)).ToVector();
        var oG = (_oGradient ?? new Tensor<T>(_oWeights._shape)).ToVector();
        if (!_ownsBiasTable)
            return Vector<T>.Concatenate(Vector<T>.Concatenate(qG, kG), Vector<T>.Concatenate(vG, oG));
        var bG = (_biasTableGradient ?? new Tensor<T>(_relativeBiasTable._shape)).ToVector();
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(qG, kG),
            Vector<T>.Concatenate(Vector<T>.Concatenate(vG, oG), bG));
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _qGradient = null;
        _kGradient = null;
        _vGradient = null;
        _oGradient = null;
        _biasTableGradient = null;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        EnsureWeightsAllocated();
        if (_qGradient is null)
            throw new InvalidOperationException(
                "Backward pass must be called before updating parameters.");

        ApplySgd(_qWeights, _qGradient, learningRate);
        if (_kGradient is not null) ApplySgd(_kWeights, _kGradient, learningRate);
        if (_vGradient is not null) ApplySgd(_vWeights, _vGradient, learningRate);
        if (_oGradient is not null) ApplySgd(_oWeights, _oGradient, learningRate);
        if (_ownsBiasTable && _biasTableGradient is not null)
            ApplySgd(_relativeBiasTable, _biasTableGradient, learningRate);
    }

    private void ApplySgd(Tensor<T> weight, Tensor<T> gradient, T lr)
    {
        var updated = Engine.TensorSubtract(weight, Engine.TensorMultiplyScalar(gradient, lr));
        for (int i = 0; i < weight.Length; i++) weight[i] = updated[i];
        Engine.InvalidatePersistentTensor(weight);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _qGradient = null;
        _kGradient = null;
        _vGradient = null;
        _oGradient = null;
        _biasTableGradient = null;
        _cachedSeqLen = -1;
        _bucketIndices = null;
    }
}
