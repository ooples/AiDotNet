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
public partial class T5RelativeBiasAttentionLayer<T> : LayerBase<T>
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _numBuckets;
    private readonly int _maxDistance;
    private readonly bool _bidirectional;
    private readonly bool _ownsBiasTable;
    private readonly Random _rng;

    // T5 has NO biases on Q/K/V/O projections (Raffel 2020 §2.1).
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _qWeights;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _kWeights;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _vWeights;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _oWeights;

    // Relative position bias table. [numBuckets, numHeads].
    // Marked trainable only if this layer owns it; otherwise the owning
    // layer registers it instead so the optimizer sees exactly one copy.
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _relativeBiasTable;

    // Cached bucket-index lookup table for the current sequence length.
    // Recomputed only when seqLen changes; positions are fixed so this
    // is pure shape state, NOT a trainable parameter.
    private int _cachedSeqLen = -1;
    private Tensor<int>? _bucketIndices;

    private Tensor<T>? _qGradient;
    private Tensor<T>? _kGradient;
    private Tensor<T>? _vGradient;
    private Tensor<T>? _oGradient;
    private Tensor<T>? _biasTableGradient;

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
    /// attend to past keys). Affects bucket layout, not masking.
    /// </param>
    /// <param name="seed">Optional RNG seed for deterministic initialisation.</param>
    /// <param name="sharedRelativeBiasTable">
    /// When non-null, the layer adopts this caller-owned bias table instead
    /// of allocating its own. The factory <c>LayerHelper.CreateDefaultT5TextLayers</c>
    /// uses this to wire one paper-canonical shared table through every
    /// attention layer in the stack.
    /// </param>
    public T5RelativeBiasAttentionLayer(
        int hiddenSize,
        int numHeads,
        int numBuckets = 32,
        int maxDistance = 128,
        bool bidirectional = true,
        int? seed = null,
        Tensor<T>? sharedRelativeBiasTable = null)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), "hiddenSize must be positive.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException(
                $"hiddenSize ({hiddenSize}) must be divisible by numHeads ({numHeads}).",
                nameof(numHeads));
        if (numBuckets <= 0)
            throw new ArgumentOutOfRangeException(nameof(numBuckets), "numBuckets must be positive.");
        if (maxDistance <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDistance), "maxDistance must be positive.");

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _numBuckets = numBuckets;
        _maxDistance = maxDistance;
        _bidirectional = bidirectional;
        _rng = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();

        // Q/K/V/O projections: shape [hiddenSize, hiddenSize], Xavier-initialised.
        _qWeights = InitProjection(hiddenSize, hiddenSize);
        _kWeights = InitProjection(hiddenSize, hiddenSize);
        _vWeights = InitProjection(hiddenSize, hiddenSize);
        _oWeights = InitProjection(hiddenSize, hiddenSize);

        RegisterTrainableParameter(_qWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_kWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_vWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_oWeights, PersistentTensorRole.Weights);

        if (sharedRelativeBiasTable is not null)
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
            RegisterTrainableParameter(_relativeBiasTable, PersistentTensorRole.Weights);
        }
    }

    /// <summary>
    /// Glorot / Xavier-uniform initialisation for a [fanIn, fanOut] weight matrix.
    /// </summary>
    private Tensor<T> InitProjection(int fanIn, int fanOut)
    {
        // T5 reference (t5x) uses normal init scaled by 1/sqrt(fan_in). We use
        // Xavier-uniform with the same effective variance so behaviour is
        // numerically equivalent for downstream usage.
        double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
        var t = new Tensor<T>(new[] { fanIn, fanOut });
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            double u = _rng.NextDouble() * 2.0 - 1.0; // [-1, 1]
            span[i] = NumOps.FromDouble(u * limit);
        }
        return t;
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
    /// Performs the forward pass: project Q/K/V, scaled dot-product attention
    /// with the T5 relative bias added pre-softmax, then output projection.
    /// All shape operations route through Engine ops so the tape records the
    /// graph for autodiff.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Accept [batch, seq, hidden] or [seq, hidden]. Flatten leading
        // dims to [batch, seq, hidden] for processing.
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

        // ---- 1. Project to Q, K, V ----
        var x2D = Engine.Reshape(input3D, new[] { batchSize * seqLen, _hiddenSize });
        var qFlat = Engine.TensorMatMul(x2D, _qWeights);
        var kFlat = Engine.TensorMatMul(x2D, _kWeights);
        var vFlat = Engine.TensorMatMul(x2D, _vWeights);

        // Reshape to [B, S, H, D_h] then permute to [B, H, S, D_h].
        var qReshaped = Engine.Reshape(qFlat, new[] { batchSize, seqLen, _numHeads, _headDim });
        var kReshaped = Engine.Reshape(kFlat, new[] { batchSize, seqLen, _numHeads, _headDim });
        var vReshaped = Engine.Reshape(vFlat, new[] { batchSize, seqLen, _numHeads, _headDim });
        var q = Engine.TensorPermute(qReshaped, new[] { 0, 2, 1, 3 });
        var k = Engine.TensorPermute(kReshaped, new[] { 0, 2, 1, 3 });
        var v = Engine.TensorPermute(vReshaped, new[] { 0, 2, 1, 3 });

        // ---- 2. Build the T5 relative position bias [numHeads, seqLen, seqLen] ----
        var biasForAttn = BuildT5RelativeBias(seqLen);

        // ---- 3. Scaled dot-product attention with bias added pre-softmax ----
        // Manual SDPA composition via tape-tracked Engine ops only.
        // (FlashAttention<T>.Forward fills its output via scalar indexing,
        // so the autodiff tape can't propagate gradients back to Q/K/V —
        // using it here would silently freeze every parameter upstream of
        // this layer, including the conditioner's EmbeddingLayer. CLIP's
        // MultiHeadAttentionLayer hits the same trap and only uses
        // FlashAttention on its ALiBi path; everywhere else it routes
        // through Engine.ScaledDotProductAttention. We need bias support,
        // which the fused engine op doesn't expose, so we compose the
        // five steps manually — every op below is proven tape-tracked
        // by its use in other trainable layers.)

        // 3a. scores = Q · K^T over the head dimension.
        //     k shape [B, H, S, D_h] -> permute to [B, H, D_h, S] so
        //     MatMul produces [B, H, S, S].
        var kT = Engine.TensorPermute(k, new[] { 0, 1, 3, 2 });
        var scoresUnscaled = Engine.TensorMatMul(q, kT);

        // 3b. Scale by 1/√head_dim. BroadcastMultiply with a singleton
        //     scalar tensor is tape-tracked (proven via RMSNorm's γ
        //     application); a raw TensorMultiplyScalar is not.
        var scaleTensor = new Tensor<T>(new[] { 1 });
        scaleTensor[0] = NumOps.FromDouble(1.0 / Math.Sqrt(_headDim));
        var scoresScaled = Engine.TensorBroadcastMultiply(scoresUnscaled, scaleTensor);

        // 3c. Add T5 relative-position bias (Raffel 2020 §2.1) before softmax.
        //     biasForAttn shape [H, S, S] broadcasts against [B, H, S, S]
        //     via BroadcastAdd (TensorAdd requires exact shape match).
        var scoresWithBias = Engine.TensorBroadcastAdd(scoresScaled, biasForAttn);

        // 3d. Row-softmax over keys (last axis). Tape-tracked.
        var attnProbs = Engine.TensorSoftmax(scoresWithBias, axis: 3);

        // 3e. context = attnProbs · V → [B, H, S, D_h].
        var context4D = Engine.TensorMatMul(attnProbs, v);

        // ---- 4. Merge heads and apply output projection ----
        // [B, H, S, D_h] -> [B, S, H, D_h] -> [B, S, hidden]
        var contextPerm = Engine.TensorPermute(context4D, new[] { 0, 2, 1, 3 });
        var contextFlat = Engine.Reshape(contextPerm, new[] { batchSize * seqLen, _hiddenSize });
        var outFlat = Engine.TensorMatMul(contextFlat, _oWeights);
        var output3D = Engine.Reshape(outFlat, new[] { batchSize, seqLen, _hiddenSize });

        // Restore original leading-dim layout.
        if (rank == 3) return output3D;
        var outShape = new int[rank];
        for (int i = 0; i < rank - 2; i++) outShape[i] = input.Shape[i];
        outShape[rank - 2] = seqLen;
        outShape[rank - 1] = _hiddenSize;
        return Engine.Reshape(output3D, outShape);
    }

    /// <summary>
    /// Builds the T5 relative position bias tensor of shape
    /// <c>[numHeads, seqLen, seqLen]</c> by looking up the trainable bias
    /// table with bucketed relative-position indices. The lookup goes
    /// through <see cref="IEngine.TensorEmbeddingLookup{T,T2}"/> so gradients
    /// flow back into <see cref="_relativeBiasTable"/> on backward.
    /// </summary>
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

    /// <inheritdoc/>
    public override long ParameterCount
    {
        get
        {
            long ownedBias = _ownsBiasTable ? _relativeBiasTable.Length : 0;
            return _qWeights.Length + _kWeights.Length + _vWeights.Length + _oWeights.Length + ownedBias;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var q = _qWeights.ToVector();
        var k = _kWeights.ToVector();
        var v = _vWeights.ToVector();
        var o = _oWeights.ToVector();
        if (!_ownsBiasTable)
            return Vector<T>.Concatenate(Vector<T>.Concatenate(q, k), Vector<T>.Concatenate(v, o));
        var b = _relativeBiasTable.ToVector();
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(q, k),
            Vector<T>.Concatenate(Vector<T>.Concatenate(v, o), b));
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected {expected} parameters, got {parameters.Length}.");

        int offset = 0;
        WriteInto(_qWeights, parameters, ref offset);
        WriteInto(_kWeights, parameters, ref offset);
        WriteInto(_vWeights, parameters, ref offset);
        WriteInto(_oWeights, parameters, ref offset);
        if (_ownsBiasTable)
            WriteInto(_relativeBiasTable, parameters, ref offset);

        Engine.InvalidatePersistentTensor(_qWeights);
        Engine.InvalidatePersistentTensor(_kWeights);
        Engine.InvalidatePersistentTensor(_vWeights);
        Engine.InvalidatePersistentTensor(_oWeights);
        if (_ownsBiasTable) Engine.InvalidatePersistentTensor(_relativeBiasTable);
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
