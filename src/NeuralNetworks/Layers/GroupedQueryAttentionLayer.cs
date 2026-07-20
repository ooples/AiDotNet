using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements Grouped-Query Attention (GQA) from Ainslie et al., 2023.
/// </summary>
/// <remarks>
/// <para>
/// GQA generalizes standard Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
/// Instead of having one set of Key/Value projections per head, GQA uses fewer K/V heads
/// that are shared among groups of Query heads. This reduces the KV-cache memory by
/// a factor of numHeads/numKVHeads while preserving most of the model quality.
/// </para>
/// <para>
/// When numKVHeads == numHeads, GQA is equivalent to standard MHA.
/// When numKVHeads == 1, GQA is equivalent to MQA.
/// </para>
/// <para><b>For Beginners:</b> GQA is a memory-efficient attention mechanism used by modern LLMs.
///
/// In standard attention with 64 heads:
/// - 64 Query projections, 64 Key projections, 64 Value projections
/// - KV-cache stores 64 sets of keys and values per layer
///
/// With GQA (64 Q heads, 8 KV heads):
/// - 64 Query projections, but only 8 Key and 8 Value projections
/// - Each K/V head is shared by 8 Query heads (64/8 = 8)
/// - KV-cache stores only 8 sets → 8x less memory!
///
/// Used by Llama 2 70B, Llama 3, Mistral, Gemma 2, and most modern large LLMs.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High, TestInputShape = "4, 16", TestConstructorArgs = "4, 16, 4, 2")]
internal partial class GroupedQueryAttentionLayer<T> : LayerBase<T>
{
    private readonly int _numHeads;
    private readonly int _numKVHeads;
    private readonly int _headDimension;
    private readonly int _embeddingDimension;
    private readonly int _headsPerGroup;

    // Deferred (lazy) weight allocation (#1671). When true, the projection weights are left
    // zero-sized at construction and materialized (allocated + initialized) on first use. A
    // foundation-scale stack (e.g. Flag-DiT's 32 layers × 4096 hidden) otherwise eagerly
    // allocates ~1.3 B weights per model in the constructor — gigabytes and >10 s before a
    // single forward, which defeats the weight-streaming forward path and the cheap-construction
    // contract the sibling DenseLayer lazy path (NoisePredictorBase.LazyDense) already honors.
    // The weight shapes are fully derivable from the dimension fields above, so ParameterCount is
    // exact while deferred and GetParameters/SetParameters are correct once materialized. Eager
    // callers (the default) are completely unaffected.
    //
    // volatile + _materializeLock make the one-time materialization safe even if a future caller
    // invokes Forward on a shared instance from multiple threads (today CheckpointBlocks and the
    // diffusion sampling loop are sequential, so it never races): the lock-free fast path reads the
    // flag with acquire semantics, and EnsureWeightsMaterialized flips it to false LAST (release)
    // so any thread seeing false also sees the fully-allocated tensors — never a half-built state.
    private volatile bool _weightsDeferred;
    private readonly object _materializeLock = new();

    // Q projection: [embDim, numHeads * headDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _queryWeights;
    // K projection: [embDim, numKVHeads * headDim] (smaller!)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _keyWeights;
    // V projection: [embDim, numKVHeads * headDim] (smaller!)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _valueWeights;
    // Output projection: [numHeads * headDim, embDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _outputWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _outputBias;
    // Optional projection biases (StarCoder2-style attention). Zero-length when unused, so the parameter
    // layout is byte-identical to the bias-free default; Optional=true also omits them from the
    // source-generated trainable-parameter set when zero-sized.
    [TrainableParameter(Role = PersistentTensorRole.Biases, Optional = true)]
    private Tensor<T> _queryBias;
    [TrainableParameter(Role = PersistentTensorRole.Biases, Optional = true)]
    private Tensor<T> _keyBias;
    [TrainableParameter(Role = PersistentTensorRole.Biases, Optional = true)]
    private Tensor<T> _valueBias;
    private readonly bool _useProjectionBias;

    // Attention-logit soft-cap (Gemma-2 attn_logit_softcapping): when > 0, each scaled Q·Kᵀ score is
    // passed through softcap·tanh(score / softcap) before the softmax. 0 disables it (standard SDPA).
    private readonly double _attnLogitSoftcap;

    // Causal masking: when true, position i attends only to positions <= i (decoder / autoregressive LM).
    // Without it the attention is bidirectional, which silently corrupts every multi-token forward of a
    // causal decoder while leaving single-token forwards (one position, no future) correct.
    private readonly bool _useCausalMask;

    // Shape-keyed cache of the flat causal row-pattern (see GetCausalMask): the [seqQ*seqKV] boolean plane
    // is computed once per (seqQ, seqKV) and block-replicated across batch*heads on demand.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<(int, int), bool[]> _causalPatternCache = new();

    // Positional encoding
    private RotaryPositionalEncodingLayer<T>? _ropeLayer;
    private ALiBiPositionalBiasLayer<T>? _alibiLayer;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastProjectedQueries;
    private Tensor<T>? _lastProjectedKeys;
    private Tensor<T>? _lastProjectedValues;
    private Tensor<T>? _lastExpandedKeys;
    private Tensor<T>? _lastExpandedValues;
    private Tensor<T>? _lastAttentionWeights;
    private Tensor<T>? _lastAttentionContext;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _outputWeightsGradient;
    private Tensor<T>? _outputBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the total number of query heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the number of key/value heads (fewer than query heads in GQA).
    /// </summary>
    public int NumKVHeads => _numKVHeads;

    /// <summary>
    /// Gets whether this layer adds separate learned biases to the Q/K/V projections (e.g. Qwen2-style).
    /// Standard LLaMA-family models leave this off.
    /// </summary>
    public bool UsesProjectionBias => _useProjectionBias;

    /// <summary>
    /// Gets the dimension of each attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the attention-logit soft-cap magnitude (Gemma-2 <c>attn_logit_softcapping</c>);
    /// 0 when disabled. When positive, each scaled Q·Kᵀ score is passed through
    /// <c>softcap·tanh(score / softcap)</c> before the softmax.
    /// </summary>
    public double AttnLogitSoftcap => _attnLogitSoftcap;

    /// <summary>
    /// Gets the number of query heads per KV head group.
    /// </summary>
    public int HeadsPerGroup => _headsPerGroup;

    /// <summary>
    /// Gets the attention variant this layer implements.
    /// </summary>
    public AttentionVariant Variant
    {
        get
        {
            if (_numKVHeads == _numHeads) return AttentionVariant.MultiHead;
            if (_numKVHeads == 1) return AttentionVariant.MultiQuery;
            return AttentionVariant.GroupedQuery;
        }
    }

    /// <summary>
    /// Gets the positional encoding type used by this attention layer.
    /// </summary>
    public PositionalEncodingType PositionalEncoding { get; private set; } = PositionalEncodingType.None;

    /// <summary>
    /// Gets the RoPE theta parameter if RoPE is configured, or the default 10000.0.
    /// </summary>
    public double RoPETheta => _ropeLayer?.Theta ?? 10000.0;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override long ParameterCount =>
        _weightsDeferred
            // Weights not yet materialized: derive the exact count from the dimensions so
            // callers iterating ParameterCount before the first forward (e.g. a parent
            // predictor's CalculateParameterCount) get the real value without forcing allocation.
            ? (long)_embeddingDimension * (_numHeads * _headDimension)        // Q
              + 2L * _embeddingDimension * (_numKVHeads * _headDimension)     // K + V
              + (long)(_numHeads * _headDimension) * _embeddingDimension      // output
              + _embeddingDimension                                          // output bias
              + (_useProjectionBias                                          // optional q/k/v biases
                  ? (long)_numHeads * _headDimension + 2L * _numKVHeads * _headDimension
                  : 0L)
            : _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
              _outputWeights.Length + _queryBias.Length + _keyBias.Length + _valueBias.Length +
              _outputBias.Length;

    /// <summary>
    /// Creates a new Grouped-Query Attention layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="embeddingDimension">Embedding dimension (must be divisible by numHeads).</param>
    /// <param name="numHeads">Total number of query heads.</param>
    /// <param name="numKVHeads">Number of key/value heads (must divide numHeads evenly).</param>
    /// <param name="activationFunction">Optional activation function (defaults to identity).</param>
    public GroupedQueryAttentionLayer(
        int sequenceLength,
        int embeddingDimension,
        int numHeads,
        int numKVHeads,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null,
        bool deferAllocation = false,
        int? headDimension = null,
        bool useProjectionBias = false,
        double attnLogitSoftcap = 0.0,
        bool useCausalMask = false)
        : base(
            [sequenceLength, embeddingDimension],
            [sequenceLength, embeddingDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        // With an explicit head dimension the projection widths are numHeads*headDim (which may differ
        // from embeddingDimension, e.g. Gemma-style decoders), so embeddingDimension need not be divisible
        // by numHeads. Only the default (headDim = embeddingDimension/numHeads) requires that divisibility.
        if (headDimension is null && embeddingDimension % numHeads != 0)
        {
            throw new ArgumentException(
                $"Embedding dimension ({embeddingDimension}) must be divisible by numHeads ({numHeads}).");
        }
        if (headDimension is { } hd && hd <= 0)
        {
            throw new ArgumentException($"headDimension ({hd}) must be positive.", nameof(headDimension));
        }

        if (numHeads % numKVHeads != 0)
        {
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKVHeads ({numKVHeads}).");
        }

        _numHeads = numHeads;
        _numKVHeads = numKVHeads;
        _headDimension = headDimension ?? (embeddingDimension / numHeads);
        _embeddingDimension = embeddingDimension;
        _headsPerGroup = numHeads / numKVHeads;
        _useProjectionBias = useProjectionBias;
        if (attnLogitSoftcap < 0.0)
        {
            throw new ArgumentException(
                $"attnLogitSoftcap ({attnLogitSoftcap}) must be non-negative (0 disables the cap).",
                nameof(attnLogitSoftcap));
        }
        _attnLogitSoftcap = attnLogitSoftcap;
        _useCausalMask = useCausalMask;

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.Eager;
        _weightsDeferred = deferAllocation;

        if (deferAllocation)
        {
            // Defer the expensive projection-weight allocation to first use (see _weightsDeferred).
            // Zero-sized placeholders keep the fields non-null; EnsureWeightsMaterialized allocates
            // the real shapes and runs InitializeParameters before any forward / GetParameters /
            // SetParameters reads them.
            _queryWeights = new Tensor<T>([0]);
            _keyWeights = new Tensor<T>([0]);
            _valueWeights = new Tensor<T>([0]);
            _outputWeights = new Tensor<T>([0]);
            _outputBias = new Tensor<T>([0]);
            _queryBias = new Tensor<T>([0]);
            _keyBias = new Tensor<T>([0]);
            _valueBias = new Tensor<T>([0]);
        }
        else
        {
            // Q projection: full-sized [embDim, numHeads * headDim]
            _queryWeights = new Tensor<T>([embeddingDimension, numHeads * _headDimension]);
            // K/V projections: reduced [embDim, numKVHeads * headDim]
            _keyWeights = new Tensor<T>([embeddingDimension, numKVHeads * _headDimension]);
            _valueWeights = new Tensor<T>([embeddingDimension, numKVHeads * _headDimension]);
            // Output projection: [numHeads * headDim, embDim]
            _outputWeights = new Tensor<T>([numHeads * _headDimension, embeddingDimension]);
            _outputBias = new Tensor<T>([embeddingDimension]);
            // Projection biases: zero-length unless enabled (StarCoder2-style).
            _queryBias = new Tensor<T>([useProjectionBias ? numHeads * _headDimension : 0]);
            _keyBias = new Tensor<T>([useProjectionBias ? numKVHeads * _headDimension : 0]);
            _valueBias = new Tensor<T>([useProjectionBias ? numKVHeads * _headDimension : 0]);

            InitializeParameters();
        }
    }

    /// <summary>
    /// Materializes deferred projection weights on first use (allocate at the real shape, then
    /// initialize via <see cref="InitializeParameters"/>). No-op for eager-constructed layers and
    /// after the first materialization. See the <c>deferAllocation</c> constructor parameter.
    /// </summary>
    private void EnsureWeightsMaterialized()
    {
        // Lock-free fast path: once materialized the volatile read observes false and (by the
        // release write below) the fully-published weight tensors.
        if (!_weightsDeferred) return;
        lock (_materializeLock)
        {
            // Double-check under the lock: another thread may have materialized while we waited.
            if (!_weightsDeferred) return;
            _queryWeights = new Tensor<T>([_embeddingDimension, _numHeads * _headDimension]);
            _keyWeights = new Tensor<T>([_embeddingDimension, _numKVHeads * _headDimension]);
            _valueWeights = new Tensor<T>([_embeddingDimension, _numKVHeads * _headDimension]);
            _outputWeights = new Tensor<T>([_numHeads * _headDimension, _embeddingDimension]);
            _outputBias = new Tensor<T>([_embeddingDimension]);
            _queryBias = new Tensor<T>([_useProjectionBias ? _numHeads * _headDimension : 0]);
            _keyBias = new Tensor<T>([_useProjectionBias ? _numKVHeads * _headDimension : 0]);
            _valueBias = new Tensor<T>([_useProjectionBias ? _numKVHeads * _headDimension : 0]);
            InitializeParameters();
            // Flip the flag LAST (volatile release): a concurrent reader either sees true and
            // blocks on the lock above, or sees false with every tensor allocated + initialized —
            // never the in-between state the previous flag-first ordering allowed.
            _weightsDeferred = false;
        }
    }

    /// <summary>
    /// A deferred-allocation layer reports NOT initialized until its weights are materialized. Eager layers
    /// are always initialized. (The <c>[TrainableParameter]</c> source generator owns this layer's
    /// <c>EnsureInitialized</c> — it has sub-layer fields — so the deferred-weight allocation is driven
    /// through <see cref="EnsureParametersMaterialized"/> instead, which <c>MaterializeParameters()</c> calls.)
    /// </summary>
    public override bool IsInitialized => !_weightsDeferred;

    /// <inheritdoc/>
    /// <remarks>Forces the deferred projection weights to materialize — the hook
    /// <see cref="LayerBase{T}.MaterializeParameters"/> invokes, so the foundation-scale chunk-streaming
    /// path (#1624) reads real weights rather than zero-length placeholders.</remarks>
    protected override void EnsureParametersMaterialized() => EnsureWeightsMaterialized();

    /// <summary>
    /// Configures positional encoding for this GQA layer.
    /// </summary>
    public void ConfigurePositionalEncoding(
        PositionalEncodingType encodingType,
        double ropeTheta = 10000.0,
        int maxSequenceLength = 2048)
    {
        PositionalEncoding = encodingType;

        // Unregister previous sub-layers before replacing
        if (_ropeLayer is not null) UnregisterSubLayer(_ropeLayer);
        if (_alibiLayer is not null) UnregisterSubLayer(_alibiLayer);

        _ropeLayer = null;
        _alibiLayer = null;

        switch (encodingType)
        {
            case PositionalEncodingType.Rotary:
                _ropeLayer = new RotaryPositionalEncodingLayer<T>(
                    maxSequenceLength, _headDimension, ropeTheta);
                RegisterSubLayer(_ropeLayer);
                break;
            case PositionalEncodingType.ALiBi:
                _alibiLayer = new ALiBiPositionalBiasLayer<T>(_numHeads, maxSequenceLength);
                RegisterSubLayer(_alibiLayer);
                break;
            case PositionalEncodingType.None:
                break;
            default:
                throw new ArgumentException(
                    $"Unsupported positional encoding type for GroupedQueryAttentionLayer: {encodingType}.",
                    nameof(encodingType));
        }
    }

    private void InitializeParameters()
    {
        InitializeLayerWeights(_queryWeights, _queryWeights.Shape[0], _queryWeights.Shape[1]);
        InitializeLayerWeights(_keyWeights, _keyWeights.Shape[0], _keyWeights.Shape[1]);
        InitializeLayerWeights(_valueWeights, _valueWeights.Shape[0], _valueWeights.Shape[1]);
        InitializeLayerWeights(_outputWeights, _outputWeights.Shape[0], _outputWeights.Shape[1]);
        InitializeLayerBiases(_outputBias);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        EnsureWeightsMaterialized();
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int embDim = input.Shape[rank - 1];

        // Flatten to 3D [batch, seq, embDim]
        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        // Every shape op via Engine so the gradient tape records the transformation.
        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, embDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, embDim });

        bool cacheBwd = ShouldCacheForBackward; // #1668: gate all backward caches (arena safety)
        _lastInput = cacheBwd ? input3D : null;

        // Project Q, K, V
        var input2D = Engine.Reshape(input3D, new[] { batchSize * seqLen, embDim });
        var Q_flat = Engine.TensorMatMul(input2D, _queryWeights);
        var K_flat = Engine.TensorMatMul(input2D, _keyWeights);
        var V_flat = Engine.TensorMatMul(input2D, _valueWeights);

        // Optional projection biases (StarCoder2). Broadcast [outDim] over the flattened [N, outDim] projection.
        if (_queryBias.Length > 0)
        {
            Q_flat = Engine.TensorBroadcastAdd(Q_flat, Engine.Reshape(_queryBias, new[] { 1, _numHeads * _headDimension }));
            K_flat = Engine.TensorBroadcastAdd(K_flat, Engine.Reshape(_keyBias, new[] { 1, _numKVHeads * _headDimension }));
            V_flat = Engine.TensorBroadcastAdd(V_flat, Engine.Reshape(_valueBias, new[] { 1, _numKVHeads * _headDimension }));
        }

        // Reshape Q: [batch, seq, numHeads, headDim] -> [batch, numHeads, seq, headDim]
        var queries = Engine.TensorPermute(
            Engine.Reshape(Q_flat, new[] { batchSize, seqLen, _numHeads, _headDimension }),
            new[] { 0, 2, 1, 3 });

        // Reshape K/V: [batch, seq, numKVHeads, headDim] -> [batch, numKVHeads, seq, headDim]
        var keys = Engine.TensorPermute(
            Engine.Reshape(K_flat, new[] { batchSize, seqLen, _numKVHeads, _headDimension }),
            new[] { 0, 2, 1, 3 });
        var values = Engine.TensorPermute(
            Engine.Reshape(V_flat, new[] { batchSize, seqLen, _numKVHeads, _headDimension }),
            new[] { 0, 2, 1, 3 });

        Tensor<T> context;
        if (!cacheBwd && _alibiLayer == null)
        {
            // ── Inference fast path ──────────────────────────────────────────────
            // Fused interleaved RoPE + GQA-aware scaled-dot-product attention, both
            // dispatched to the device engine (float-specialized CPU / GPU kernels).
            // This eliminates the two dominant CPU self-time costs on the decoder
            // forward: the managed RoPE rotate loop (RotateTensor) and the scalar
            // ExpandKVHeads copy. The SDPA kernel broadcasts the shared KV heads
            // internally, so no expanded [batch, numHeads, seq, headDim] K/V is ever
            // materialized. Numerically identical to the training path below:
            //   - ApplyRoPEInterleaved matches RotateTensor exactly (GPT-J/GGML 2i,2i+1)
            //   - ScaledDotProductAttentionGqa uses scale = 1/sqrt(headDim), the same
            //     causal offset (key j visible to query i iff j <= i + (seqK - seqQ)),
            //     and the same attn-logit soft-cap.
            // Only taken outside training (no backward caches / no tape) and without
            // ALiBi (which needs the additive-bias FlashAttention path).
            if (_ropeLayer != null)
            {
                var (cosCache, sinCache) = _ropeLayer.GetInterleavedCaches(seqLen);
                queries = Engine.ApplyRoPEInterleaved(queries, cosCache, sinCache, startPosition: 0);
                keys = Engine.ApplyRoPEInterleaved(keys, cosCache, sinCache, startPosition: 0);
            }

            context = Engine.ScaledDotProductAttentionGqa(
                queries, keys, values,
                scale: 1.0 / Math.Sqrt(_headDimension),
                isCausal: _useCausalMask,
                softcap: _attnLogitSoftcap);
        }
        else
        {
            // ── Training / ALiBi path (tape-recorded, manual-backward caches) ─────
            // Apply RoPE to Q and K (before KV head expansion)
            if (_ropeLayer != null)
            {
                (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: 0);
            }

            _lastProjectedQueries = cacheBwd ? queries : null;
            _lastProjectedKeys = cacheBwd ? keys : null;
            _lastProjectedValues = cacheBwd ? values : null;

            // Expand K/V heads to match Q heads via repeat
            var expandedKeys = ExpandKVHeads(keys, batchSize, seqLen);
            var expandedValues = ExpandKVHeads(values, batchSize, seqLen);

            _lastExpandedKeys = cacheBwd ? expandedKeys : null;
            _lastExpandedValues = cacheBwd ? expandedValues : null;

            // Compute attention with weights caching: [batch, numHeads, seqQ, seqKV]
            // The attention-logit soft-cap flows only through the standard fused SDPA path; the ALiBi
            // FlashAttention path has no soft-cap parameter, so reject the (never-faithful) combination
            // rather than silently dropping the cap.
            if (_alibiLayer != null && _attnLogitSoftcap > 0.0)
            {
                throw new InvalidOperationException(
                    "attnLogitSoftcap is not supported together with ALiBi positional bias; " +
                    "the soft-cap applies only to the standard scaled dot-product attention path.");
            }
            var (ctx, attentionWeights) = _alibiLayer != null
                ? ComputeALiBiAttention(queries, expandedKeys, expandedValues, seqLen, batchSize)
                : ComputeStandardAttentionWithWeights(queries, expandedKeys, expandedValues);

            _lastAttentionWeights = cacheBwd ? attentionWeights : null;
            context = ctx;
        }

        // Reshape back: [batch, numHeads, seq, headDim] -> [batch, seq, embDim]
        var contextPermuted = Engine.TensorPermute(context, new[] { 0, 2, 1, 3 });
        var contextTransposed = Engine.Reshape(
            contextPermuted,
            new[] { batchSize * seqLen, _numHeads * _headDimension });

        // Cache pre-projection context for output weights gradient
        _lastAttentionContext = cacheBwd
            ? Engine.Reshape(contextTransposed, new[] { batchSize, seqLen, _numHeads * _headDimension })
            : null;

        // Output projection
        var output = Engine.TensorMatMul(contextTransposed, _outputWeights);
        var output3D = Engine.Reshape(output, new[] { batchSize, seqLen, _embeddingDimension });

        // Add bias — reshape bias fresh each call so the tape has a live GradFn
        // chain from _outputBias on every training step (a cached reshape primed
        // during inference would dead-end backward at the cached handle).
        var biasBroadcast = Engine.Reshape(_outputBias, new[] { 1, 1, _embeddingDimension });
        var outputWithBias = Engine.TensorBroadcastAdd(output3D, biasBroadcast);
        var result = ApplyActivation(outputWithBias);

        _lastOutput = cacheBwd ? result : null;

        // Reshape back to original rank — via Engine for tape recording.
        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _embeddingDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _embeddingDimension;
        return Engine.Reshape(result, outputShape);
    }

    /// <summary>
    /// Expands K/V from [batch, numKVHeads, seq, headDim] to [batch, numHeads, seq, headDim]
    /// by repeating each KV head headsPerGroup times.
    /// </summary>
    private Tensor<T> ExpandKVHeads(Tensor<T> kv, int batchSize, int seqLen)
    {
        if (_numKVHeads == _numHeads)
            return kv; // No expansion needed (standard MHA)

        var expanded = TensorAllocator.Rent<T>(new[] { batchSize, _numHeads, seqLen, _headDimension });

        for (int b = 0; b < batchSize; b++)
        {
            for (int kvh = 0; kvh < _numKVHeads; kvh++)
            {
                for (int g = 0; g < _headsPerGroup; g++)
                {
                    int qh = kvh * _headsPerGroup + g;
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int d = 0; d < _headDimension; d++)
                        {
                            expanded[new[] { b, qh, s, d }] = kv[new[] { b, kvh, s, d }];
                        }
                    }
                }
            }
        }

        return expanded;
    }

    private (Tensor<T> Context, Tensor<T> AttentionWeights) ComputeALiBiAttention(
        Tensor<T> queries, Tensor<T> keys, Tensor<T> values, int seqLen, int batchSize)
    {
        var aliBiBias = _alibiLayer?.ComputeBias(seqLen, seqLen);
        var flashConfig = new FlashAttentionConfig { ReturnAttentionWeights = true };
        var (flashOutput, flashWeights) = FlashAttention<T>.Forward(queries, keys, values, flashConfig, attentionBias: aliBiBias);
        if (flashWeights is null)
        {
            throw new InvalidOperationException(
                "FlashAttention returned null attention weights despite ReturnAttentionWeights=true. " +
                "This would corrupt the backward pass. Ensure FlashAttention.Forward returns weights when requested.");
        }
        return (flashOutput, flashWeights);
    }

    private (Tensor<T> Context, Tensor<T> AttentionWeights) ComputeStandardAttentionWithWeights(
        Tensor<T> queries, Tensor<T> keys, Tensor<T> values)
    {
        // Causal decoder without a soft-cap: use FlashAttention's causal mask (softcap-free path), the same
        // mechanism the stateless paged fallback uses. Soft-capped causal attention (Gemma-2) falls through
        // to the Engine SDPA path below, which applies an additive causal mask so the cap and mask compose.
        if (_useCausalMask && _attnLogitSoftcap <= 0.0)
        {
            var flashConfig = new FlashAttentionConfig { ReturnAttentionWeights = true, UseCausalMask = true };
            var (flashOutput, flashWeights) = FlashAttention<T>.Forward(
                queries.Contiguous(), keys.Contiguous(), values.Contiguous(), flashConfig);
            if (flashWeights is null)
            {
                throw new InvalidOperationException(
                    "FlashAttention returned null attention weights despite ReturnAttentionWeights=true.");
            }
            return (flashOutput, flashWeights);
        }

        var context = ComputeStandardAttention(queries, keys, values, out var attentionWeights);
        return (context, attentionWeights);
    }

    private Tensor<T> ComputeStandardAttention(Tensor<T> queries, Tensor<T> keys, Tensor<T> values, out Tensor<T> attentionWeightsOut)
    {
        // Standard scaled dot-product attention: softmax(Q·K^T / sqrt(d_k)) · V.
        // Manual implementation was 6 nested loops doing per-element NumOps
        // dispatches — O(batch · numHeads · seqLenQ · seqLenKV · headDim) virtual
        // calls per Q·K^T pass plus the same again for attn·V. Replaced with
        // Engine.ScaledDotProductAttention which fuses Q·K^T, scale, softmax,
        // and attn·V into one kernel call (and gives a SIMD/GPU dispatch when
        // available).
        int headDim = queries.Shape[3];
        // Causal decoders pass a boolean mask (true = a query may attend to that key, i.e. key <= query),
        // so future keys are excluded before the softmax. Non-causal (encoder) attention passes no mask.
        Tensor<bool>? mask = _useCausalMask
            ? GetCausalMask(queries.Shape[0], queries.Shape[1], queries.Shape[2], keys.Shape[2])
            : null;
        return Engine.ScaledDotProductAttention(
            queries, keys, values,
            mask,
            scale: 1.0 / Math.Sqrt(headDim),
            out attentionWeightsOut,
            softcap: _attnLogitSoftcap);
    }

    // Materializes the [batch, heads, seqQ, seqKV] boolean causal mask the fused SDPA kernel consumes (it
    // does not broadcast the batch/head dims). The per-(seqQ,seqKV) row plane is computed ONCE with a flat
    // fill and cached; each forward only block-copies that plane across batch*heads (no scalar indexer, no
    // O(seq^2) recompute). The masked attention itself stays in the fused Engine kernel (SIMD/GPU).
    private Tensor<bool> GetCausalMask(int batch, int heads, int seqQ, int seqKV)
    {
        var plane = _causalPatternCache.GetOrAdd((seqQ, seqKV), static key =>
        {
            var (sq, skv) = key;
            var p = new bool[sq * skv];
            // A query at position i (over the last sq keys) attends to keys [0 .. keyOffset + i].
            int keyOffset = skv - sq;
            for (int i = 0; i < sq; i++)
            {
                int lastAllowed = Math.Min(keyOffset + i, skv - 1);
                int rowBase = i * skv;
                for (int j = 0; j <= lastAllowed; j++)
                {
                    p[rowBase + j] = true;
                }
            }
            return p;
        });

        int planeLen = seqQ * seqKV;
        var data = new bool[batch * heads * planeLen];
        for (int k = 0; k < batch * heads; k++)
        {
            Array.Copy(plane, 0, data, k * planeLen, planeLen);
        }
        return new Tensor<bool>(data, new[] { batch, heads, seqQ, seqKV });
    }

    /// <summary>
    /// Aggregates expanded K/V gradients from [batch, numHeads, seq, headDim]
    /// back to [batch, numKVHeads, seq, headDim] by summing across head groups.
    /// </summary>
    private Tensor<T> AggregateKVGradients(Tensor<T> expandedGrad, int batchSize, int seqLen)
    {
        if (_numKVHeads == _numHeads)
            return expandedGrad; // No aggregation needed (standard MHA)

        var aggregated = TensorAllocator.Rent<T>(new[] { batchSize, _numKVHeads, seqLen, _headDimension });

        for (int b = 0; b < batchSize; b++)
        {
            for (int kvh = 0; kvh < _numKVHeads; kvh++)
            {
                for (int g = 0; g < _headsPerGroup; g++)
                {
                    int qh = kvh * _headsPerGroup + g;
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int d = 0; d < _headDimension; d++)
                        {
                            aggregated[new[] { b, kvh, s, d }] = NumOps.Add(
                                aggregated[new[] { b, kvh, s, d }],
                                expandedGrad[new[] { b, qh, s, d }]);
                        }
                    }
                }
            }
        }

        return aggregated;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        EnsureWeightsMaterialized();
        if (_queryWeightsGradient == null || _keyWeightsGradient == null ||
            _valueWeightsGradient == null || _outputWeightsGradient == null ||
            _outputBiasGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient, negLR));
        _outputWeights = Engine.TensorAdd(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGradient, negLR));
        _outputBias = Engine.TensorAdd(_outputBias, Engine.TensorMultiplyScalar(_outputBiasGradient, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        EnsureWeightsMaterialized();
        int totalParams = _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
                          _outputWeights.Length + _queryBias.Length + _keyBias.Length + _valueBias.Length +
                          _outputBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        // Order: Q/K/V/O weights, optional q/k/v biases (zero-length when unused → layout unchanged),
        // then the output bias LAST (the tensor-parallel partitioner reads it tail-wise).
        foreach (var tensor in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights, _queryBias, _keyBias, _valueBias, _outputBias })
        {
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        }

        return parameters;
    }

    public override Vector<T> GetParameterGradients()
    {
        EnsureWeightsMaterialized();
        var gradTensors = new[] { _queryWeightsGradient, _keyWeightsGradient, _valueWeightsGradient, _outputWeightsGradient, null, null, null, _outputBiasGradient };
        var weightTensors = new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights, _queryBias, _keyBias, _valueBias, _outputBias };
        int totalParams = weightTensors.Sum(w => w.Length);
        var result = new Vector<T>(totalParams);
        int index = 0;
        for (int g = 0; g < gradTensors.Length; g++)
        {
            var grad = gradTensors[g];
            int len = weightTensors[g].Length;
            if (grad != null)
            {
                for (int i = 0; i < len; i++)
                    result[index++] = grad[i];
            }
            else
            {
                index += len;
            }
        }
        return result;
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        EnsureWeightsMaterialized();
        int expectedParams = _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
                             _outputWeights.Length + _queryBias.Length + _keyBias.Length + _valueBias.Length +
                             _outputBias.Length;
        if (parameters.Length != expectedParams)
            throw new ArgumentException($"Expected {expectedParams} parameters, got {parameters.Length}");

        int index = 0;
        foreach (var tensor in new[] { _queryWeights, _keyWeights, _valueWeights, _outputWeights, _queryBias, _keyBias, _valueBias, _outputBias })
        {
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastProjectedQueries = null;
        _lastProjectedKeys = null;
        _lastProjectedValues = null;
        _lastExpandedKeys = null;
        _lastExpandedValues = null;
        _lastAttentionWeights = null;
        _lastAttentionContext = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputWeightsGradient = null;
        _outputBiasGradient = null;
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        // Persist the constructor's full parameter set so DeserializationHelper
        // can reconstruct the layer without fabricating any dimension. Without
        // SequenceLength + EmbeddingDimension here, the deser path would fall
        // back to inputShape[0]/[1] (correct for rank-2 [seq, dim] payloads)
        // or to hardcoded 16/64 if the shape is degenerate — issue #1239.
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["SequenceLength"] = InputShape[0].ToString();
        metadata["EmbeddingDimension"] = _embeddingDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["NumKVHeads"] = _numKVHeads.ToString();
        metadata["HeadsPerGroup"] = _headsPerGroup.ToString();
        metadata["Variant"] = Variant.ToString();
        metadata["PositionalEncoding"] = PositionalEncoding.ToString();
        // Persist the remaining shape/behaviour-affecting ctor arguments so a deserialized (cloned) layer is
        // functionally identical. Without these a clone silently lost its causal mask, custom head dimension
        // (Gemma), Q/K/V projection bias (Qwen2), attention logit soft-cap (Gemma-2), and RoPE — producing
        // wrong outputs on the cloned model (e.g. the paged incremental-serving clone of a GGUF decoder).
        metadata["HeadDimension"] = _headDimension.ToString(ci);
        metadata["UseCausalMask"] = _useCausalMask.ToString();
        metadata["UseProjectionBias"] = _useProjectionBias.ToString();
        metadata["AttnLogitSoftcap"] = _attnLogitSoftcap.ToString(ci);
        metadata["RoPETheta"] = RoPETheta.ToString(ci);
        return metadata;
    }

    /// <summary>
    /// Gets the query projection weights for external use (e.g., quantization).
    /// </summary>
    public Tensor<T> GetQueryWeights() { EnsureWeightsMaterialized(); return _queryWeights; }

    /// <summary>
    /// Gets the key projection weights for external use.
    /// </summary>
    public Tensor<T> GetKeyWeights() { EnsureWeightsMaterialized(); return _keyWeights; }

    /// <summary>
    /// Gets the value projection weights for external use.
    /// </summary>
    public Tensor<T> GetValueWeights() { EnsureWeightsMaterialized(); return _valueWeights; }

    /// <summary>
    /// Gets the output projection weights for external use.
    /// </summary>
    public Tensor<T> GetOutputWeights() { EnsureWeightsMaterialized(); return _outputWeights; }

}
