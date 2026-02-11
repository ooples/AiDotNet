using System.Threading;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Inference.Quantization;
using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference;

/// <summary>
/// Applies inference optimizations to neural network models based on configuration.
/// </summary>
/// <remarks>
/// <para>
/// InferenceOptimizer bridges the InferenceOptimizationConfig with actual inference
/// components (KV Cache, Speculative Decoding, etc.). It automatically detects which
/// optimizations are applicable to a given model and applies them.
/// </para>
/// <para><b>For Beginners:</b> This class makes your model faster during prediction.
///
/// When you call OptimizeForInference(), it:
/// 1. Detects what kind of model you have (transformer, neural network, etc.)
/// 2. Applies appropriate optimizations based on your config
/// 3. Returns an optimized inference context you can use for fast predictions
///
/// Example:
/// <code>
/// var optimizer = new InferenceOptimizer&lt;double&gt;(config);
/// var context = optimizer.CreateInferenceContext(model);
/// var result = context.Predict(input);  // Faster prediction!
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for computations.</typeparam>
internal class InferenceOptimizer<T>
{
    private readonly InferenceOptimizationConfig _config;
    private KVCache<T>? _kvCache;
    private PagedKVCache<T>? _pagedKVCache;
    private PagedAttentionKernel<T>? _pagedKernel;
    private long? _pagedSequenceId;
    private List<PagedCachedMultiHeadAttention<T>>? _pagedAttentionLayers;
    private static long s_nextPagedSequenceId = DateTime.UtcNow.Ticks;
    private IDraftModel<T>? _draftModel;
    private SpeculativeDecoder<T>? _speculativeDecoder;
    private bool _isInitialized;

    /// <summary>
    /// Gets the configuration being used.
    /// </summary>
    public InferenceOptimizationConfig Config => _config;

    /// <summary>
    /// Gets the KV cache if enabled and initialized.
    /// </summary>
    public KVCache<T>? KVCache => _kvCache;

    /// <summary>
    /// Gets whether the optimizer has been initialized with a model.
    /// </summary>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Creates a new InferenceOptimizer with the specified configuration.
    /// </summary>
    /// <param name="config">The inference optimization configuration.</param>
    public InferenceOptimizer(InferenceOptimizationConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    /// <summary>
    /// Creates a new InferenceOptimizer with default configuration.
    /// </summary>
    public InferenceOptimizer()
        : this(InferenceOptimizationConfig.Default)
    {
    }

    /// <summary>
    /// Creates an inference-optimized model instance based on the current configuration.
    /// </summary>
    /// <param name="model">The neural network to optimize.</param>
    /// <param name="cloneModel">Whether to clone the model before applying layer-level rewrites.</param>
    /// <returns>The optimized model and whether any optimizations were applied.</returns>
    /// <remarks>
    /// This method can apply stateless layer rewrites (e.g., MultiHeadAttention -> FlashAttentionLayer)
    /// and then initialize stateful inference features (e.g., KV-cache) on the resulting model.
    /// </remarks>
    public (NeuralNetworkBase<T> OptimizedModel, bool AnyOptimizationsApplied) OptimizeForInference(
        NeuralNetworkBase<T> model,
        bool cloneModel = true)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        _config.Validate();

        // Clone only when we might rewrite layers; otherwise keep original reference.
        bool mayRewriteAttention = _config.EnableFlashAttention || _config.EnableKVCache || _config.EnableWeightOnlyQuantization;
        var workingModel = model;
        if (cloneModel && mayRewriteAttention && HasOptimizableAttentionLayers(model))
        {
            try
            {
                // NeuralNetworkBase.Clone performs a deep copy via serialization.
                workingModel = (NeuralNetworkBase<T>)model.Clone();
            }
            catch (Exception ex)
            {
                // Some layer types may not yet support serialization-based cloning.
                // Do not mutate the user's original model; just skip optimizations.
                Console.WriteLine($"Warning: model cloning failed for inference optimizations: {ex.Message}. Skipping inference optimizations for this model instance.");
                InferenceDiagnostics.RecordException(
                    area: "InferenceOptimizer",
                    feature: "CloneForRewrite",
                    ex: ex,
                    reason: "Clone failed; skipping all inference optimizations to avoid mutating user model.");
                return (model, false);
            }
        }

        bool anyApplied = ApplyAttentionOptimizations(workingModel);
        InferenceDiagnostics.RecordDecision("InferenceOptimizer", "AttentionRewrites", enabled: anyApplied, reason: anyApplied ? "Applied" : "NoApplicableLayersOrDisabled");
        anyApplied |= ApplyWeightOnlyQuantization(workingModel);
        anyApplied |= Initialize(workingModel);

        return (workingModel, anyApplied);
    }

    /// <summary>
    /// Initializes inference optimizations for a neural network model.
    /// </summary>
    /// <param name="model">The neural network to optimize.</param>
    /// <returns>True if any optimizations were applied.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this once before making predictions.
    ///
    /// This method:
    /// - Analyzes your model to find attention layers
    /// - Sets up KV cache if applicable and enabled
    /// - Prepares speculative decoding if enabled
    /// - Puts attention layers in inference mode
    /// </para>
    /// </remarks>
    public bool Initialize(NeuralNetworkBase<T> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        _config.Validate();

        bool anyOptimizationsApplied = false;

        // Find and configure attention layers for KV caching
        if (_config.EnableKVCache)
        {
            anyOptimizationsApplied |= _config.EnablePagedKVCache
                ? InitializePagedKVCache(model)
                : InitializeKVCache(model);
        }
        else
        {
            InferenceDiagnostics.RecordDecision("InferenceOptimizer", "KVCache", enabled: false, reason: "DisabledByConfig");
        }

        // Initialize speculative decoding if enabled
        if (_config.EnableSpeculativeDecoding)
        {
            anyOptimizationsApplied |= InitializeSpeculativeDecoding(model);
        }
        else
        {
            InferenceDiagnostics.RecordDecision("InferenceOptimizer", "SpeculativeDecoding", enabled: false, reason: "DisabledByConfig");
        }

        _isInitialized = true;
        return anyOptimizationsApplied;
    }

    /// <summary>
    /// Sets up KV caching for transformer attention layers.
    /// </summary>
    private bool InitializeKVCache(NeuralNetworkBase<T> model)
    {
        // Find all cached attention layers or layers that support caching
        var attentionLayers = new List<CachedMultiHeadAttention<T>>();
        var gqaLayers = new List<CachedGroupedQueryAttention<T>>();
        int layerIndex = 0;

        foreach (var layer in model.Layers)
        {
            if (layer is CachedGroupedQueryAttention<T> cachedGqa)
            {
                cachedGqa.LayerIndex = layerIndex;
                gqaLayers.Add(cachedGqa);
                layerIndex++;
            }
            else if (layer is CachedMultiHeadAttention<T> cachedAttention)
            {
                cachedAttention.LayerIndex = layerIndex;
                attentionLayers.Add(cachedAttention);
                layerIndex++;
            }
        }

        if (attentionLayers.Count == 0 && gqaLayers.Count == 0)
        {
            // No attention layers found - KV cache not applicable
            return false;
        }

        bool anyInitialized = false;

        // Handle GQA layers
        if (gqaLayers.Count > 0)
        {
            anyInitialized = InitializeGQAKVCache(gqaLayers);
        }

        // Handle MHA layers (even if GQA layers also exist)
        if (attentionLayers.Count > 0 && _kvCache != null)
        {
            // Cache already created by GQA init; attach to MHA layers
            foreach (var layer in attentionLayers)
            {
                layer.Cache = _kvCache;
                layer.InferenceMode = true;
            }

            return true;
        }

        if (attentionLayers.Count == 0)
        {
            return anyInitialized;
        }

        // Determine cache parameters from the first attention layer
        var firstLayer = attentionLayers[0];
        int numHeads = firstLayer.HeadCount;
        int headDim = firstLayer.HeadDimension;
        int numLayers = attentionLayers.Count;
        int maxSeqLen = EstimateMaxSequenceLength(numLayers, numHeads, headDim);

        // Create KV cache configuration
        var cacheConfig = new KVCacheConfig
        {
            NumLayers = attentionLayers.Count,
            NumHeads = numHeads,
            HeadDimension = headDim,
            MaxSequenceLength = maxSeqLen,
            MaxBatchSize = _config.MaxBatchSize,
            PreAllocate = true,
            UseSlidingWindow = _config.UseSlidingWindowKVCache,
            WindowSize = _config.UseSlidingWindowKVCache
                ? Math.Min(_config.KVCacheWindowSize, maxSeqLen)
                : 1024,
            DataType = ResolveKVCacheDataType()
        };

        // Create and attach KV cache
        _kvCache = new KVCache<T>(cacheConfig);

        // Attach cache to all attention layers and enable inference mode
        foreach (var layer in attentionLayers)
        {
            layer.Cache = _kvCache;
            layer.InferenceMode = true;
        }

        return true;
    }

    /// <summary>
    /// Initializes KV cache for GQA layers using numKVHeads instead of numHeads.
    /// </summary>
    private bool InitializeGQAKVCache(List<CachedGroupedQueryAttention<T>> gqaLayers)
    {
        var firstLayer = gqaLayers[0];
        // Use KV head count for cache (the key memory saving of GQA)
        int numKVHeads = firstLayer.KVHeadCount;
        int headDim = firstLayer.HeadDimension;
        int numLayers = gqaLayers.Count;
        int maxSeqLen = EstimateMaxSequenceLength(numLayers, numKVHeads, headDim);

        var cacheConfig = new KVCacheConfig
        {
            NumLayers = numLayers,
            NumHeads = numKVHeads, // Key: cache uses KV heads, not full heads
            HeadDimension = headDim,
            MaxSequenceLength = maxSeqLen,
            MaxBatchSize = _config.MaxBatchSize,
            PreAllocate = true,
            UseSlidingWindow = _config.UseSlidingWindowKVCache,
            WindowSize = _config.UseSlidingWindowKVCache
                ? Math.Min(_config.KVCacheWindowSize, maxSeqLen)
                : 1024,
            DataType = ResolveKVCacheDataType()
        };

        _kvCache = new KVCache<T>(cacheConfig);

        foreach (var layer in gqaLayers)
        {
            layer.Cache = _kvCache;
            layer.InferenceMode = true;
        }

        return true;
    }

    private CacheDataType ResolveKVCacheDataType()
    {
        bool fp16Capable = typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(Half);
        bool int8Capable = fp16Capable;

        CacheDataType resolved;
        if (_config.KVCacheQuantization == KVCacheQuantizationMode.Int8 && int8Capable)
        {
            resolved = CacheDataType.Int8;
        }
        else
        {
            resolved = _config.KVCachePrecision switch
            {
                KVCachePrecisionMode.Float32 => CacheDataType.Float32,
                KVCachePrecisionMode.Float16 => fp16Capable ? CacheDataType.Float16 : CacheDataType.Float32,
                _ => fp16Capable ? CacheDataType.Float16 : CacheDataType.Float32
            };
        }

        InferenceDiagnostics.RecordDecision(
            area: "InferenceOptimizer",
            feature: "KVCachePrecision",
            enabled: resolved == CacheDataType.Float16 || resolved == CacheDataType.Int8,
            reason: $"Precision={_config.KVCachePrecision};Quant={_config.KVCacheQuantization};Resolved={resolved};Type={typeof(T).Name}");

        return resolved;
    }

    private bool InitializePagedKVCache(NeuralNetworkBase<T> model)
    {
        var attentionLayers = new List<PagedCachedMultiHeadAttention<T>>();
        int layerIndex = 0;

        foreach (var layer in model.Layers)
        {
            if (layer is PagedCachedMultiHeadAttention<T> pagedAttention)
            {
                pagedAttention.LayerIndex = layerIndex;
                attentionLayers.Add(pagedAttention);
                layerIndex++;
            }
        }

        if (attentionLayers.Count == 0)
        {
            // No paged attention layers present; fall back to contiguous cache if applicable.
            return InitializeKVCache(model);
        }

        var firstLayer = attentionLayers[0];
        int numHeads = firstLayer.HeadCount;
        int headDim = firstLayer.HeadDimension;
        int numLayers = attentionLayers.Count;

        long availableBytes = (long)_config.KVCacheMaxSizeMB * 1024 * 1024;
        int blockSize = _config.PagedKVCacheBlockSize;

        _pagedKVCache = PagedKVCache<T>.FromMemorySize(availableBytes, numLayers, numHeads, headDim, blockSize);
        _pagedKernel = new PagedAttentionKernel<T>(_pagedKVCache, new PagedAttentionConfig
        {
            NumHeads = numHeads,
            HeadDimension = headDim,
            BlockSize = blockSize,
            MaxBatchSize = _config.MaxBatchSize
        });

        // Allocate a fresh sequence ID for this optimized model instance (one model == one sequence).
        if (!TryAllocatePagedSequenceId(_pagedKVCache, initialTokens: 0, out long sequenceId))
        {
            InferenceDiagnostics.RecordDecision(
                area: "InferenceOptimizer",
                feature: "PagedKVCache",
                enabled: false,
                reason: "AllocateSequenceFailed(OutOfMemoryOrExhausted)");
            _pagedKVCache = null;
            _pagedKernel = null;
            _pagedAttentionLayers = null;
            _pagedSequenceId = null;
            return false;
        }

        _pagedSequenceId = sequenceId;
        _pagedAttentionLayers = attentionLayers;

        foreach (var layer in attentionLayers)
        {
            layer.Kernel = _pagedKernel;
            layer.SequenceId = sequenceId;
            layer.InferenceMode = true;
        }

        return true;
    }

    private static bool TryAllocatePagedSequenceId(PagedKVCache<T> cache, int initialTokens, out long sequenceId)
    {
        const int maxAttempts = 1024;
        var spin = new SpinWait();

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            sequenceId = Interlocked.Increment(ref s_nextPagedSequenceId);
            if (cache.AllocateSequence(sequenceId, initialTokens))
            {
                return true;
            }

            spin.SpinOnce();
        }

        sequenceId = 0;
        return false;
    }

    private static bool TryAllocatePagedSequenceId(PagedKVCache<T> cache, long preferredId, int initialTokens, out long sequenceId)
    {
        if (cache.AllocateSequence(preferredId, initialTokens))
        {
            sequenceId = preferredId;
            return true;
        }

        return TryAllocatePagedSequenceId(cache, initialTokens, out sequenceId);
    }

    private bool HasOptimizableAttentionLayers(NeuralNetworkBase<T> model)
    {
        foreach (var layer in model.Layers)
        {
            if (layer is MultiHeadAttentionLayer<T> || layer is FlashAttentionLayer<T> ||
                layer is SelfAttentionLayer<T> || layer is GroupedQueryAttentionLayer<T>)
                return true;

            if (_config.EnableWeightOnlyQuantization &&
                typeof(T) == typeof(float) &&
                layer is DenseLayer<float>)
            {
                return true;
            }
        }

        return false;
    }

    private bool ApplyAttentionOptimizations(NeuralNetworkBase<T> model)
    {
        bool useCausalMask = ResolveCausalMask(model);
        InferenceDiagnostics.RecordDecision(
            area: "InferenceOptimizer",
            feature: "CausalMask",
            enabled: useCausalMask,
            reason: _config.AttentionMasking == AttentionMaskingMode.Auto ? "Auto" : _config.AttentionMasking.ToString());

        // KV-cache is only beneficial for incremental decoding patterns; default to enabling it only when causal masking applies.
        bool enableKVCache = _config.EnableKVCache && useCausalMask;
        bool enablePagedKVCache = enableKVCache && _config.EnablePagedKVCache;
        bool enableFlashAttention = _config.EnableFlashAttention;

        bool anyRewritten = false;

        for (int i = 0; i < model.Layers.Count; i++)
        {
            var layer = model.Layers[i];

            if (layer is SelfAttentionLayer<T> selfAttention && (enableKVCache || enableFlashAttention))
            {
                var converted = TryConvertSelfAttentionToMultiHead(selfAttention);
                if (converted != null)
                {
                    model.Layers[i] = converted;
                    anyRewritten = true;

                    // Re-process this index under MultiHeadAttention rules.
                    i--;
                    continue;
                }

                InferenceDiagnostics.RecordDecision(
                    area: "InferenceOptimizer",
                    feature: "SelfAttentionRewrite",
                    enabled: false,
                    reason: "UnsupportedSelfAttentionLayer(HeadCountOrShape)");
                continue;
            }

            if (layer is MultiHeadAttentionLayer<T> mha)
            {
                var inputShape = mha.GetInputShape();
                if (inputShape.Length < 2)
                {
                    continue;
                }

                int seqLen = inputShape[0];
                int embDim = inputShape[1];
                int headCount = mha.HeadCount;
                var activation = mha.ScalarActivation;

                if (enableKVCache)
                {
                    if (enablePagedKVCache)
                    {
                        var paged = new PagedCachedMultiHeadAttention<T>(
                            sequenceLength: seqLen,
                            embeddingDimension: embDim,
                            headCount: headCount,
                            useCausalMask: useCausalMask,
                            activationFunction: activation);
                        paged.EnableWeightOnlyQuantization = _config.EnableWeightOnlyQuantization;
                        paged.SetParameters(mha.GetParameters());

                        // Preserve positional encoding configuration from source MHA layer
                        if (mha.PositionalEncoding != PositionalEncodingType.None)
                        {
                            paged.ConfigurePositionalEncoding(
                                mha.PositionalEncoding,
                                ropeTheta: mha.RoPETheta,
                                maxSequenceLength: seqLen);
                        }
                        else if (_config.PositionalEncoding == PositionalEncodingType.Rotary ||
                                 _config.PositionalEncoding == PositionalEncodingType.ALiBi)
                        {
                            paged.ConfigurePositionalEncoding(
                                _config.PositionalEncoding,
                                ropeTheta: _config.RoPETheta,
                                maxSequenceLength: seqLen);
                        }

                        model.Layers[i] = paged;
                    }
                    else
                    {
                        var cached = new CachedMultiHeadAttention<T>(
                            sequenceLength: seqLen,
                            embeddingDimension: embDim,
                            headCount: headCount,
                            useFlashAttention: enableFlashAttention,
                            layerIndex: 0,
                            useCausalMask: useCausalMask,
                            activationFunction: activation);
                        cached.SetParameters(mha.GetParameters());

                        // Preserve positional encoding configuration from source MHA layer
                        if (mha.PositionalEncoding != PositionalEncodingType.None)
                        {
                            cached.ConfigurePositionalEncoding(
                                mha.PositionalEncoding,
                                ropeTheta: mha.RoPETheta,
                                maxSequenceLength: seqLen);
                        }
                        else if (_config.PositionalEncoding == PositionalEncodingType.Rotary ||
                                 _config.PositionalEncoding == PositionalEncodingType.ALiBi)
                        {
                            cached.ConfigurePositionalEncoding(
                                _config.PositionalEncoding,
                                ropeTheta: _config.RoPETheta,
                                maxSequenceLength: seqLen);
                        }

                        model.Layers[i] = cached;
                    }
                    anyRewritten = true;
                    continue;
                }

                if (enableFlashAttention)
                {
                    var flashConfig = FlashAttentionConfig.Default;
                    flashConfig.UseCausalMask = useCausalMask;

                    var flashLayer = new FlashAttentionLayer<T>(
                        sequenceLength: seqLen,
                        embeddingDimension: embDim,
                        headCount: headCount,
                        config: flashConfig,
                        activationFunction: activation);
                    flashLayer.SetParameters(mha.GetParameters());
                    model.Layers[i] = flashLayer;
                    anyRewritten = true;
                }

                continue;
            }

            if (layer is FlashAttentionLayer<T> flash && enableKVCache)
            {
                var inputShape = flash.GetInputShape();
                if (inputShape.Length < 2)
                {
                    continue;
                }

                int seqLen = inputShape[0];
                int embDim = inputShape[1];
                int headCount = flash.HeadCount;
                var activation = flash.ScalarActivation;

                if (enablePagedKVCache)
                {
                    var paged = new PagedCachedMultiHeadAttention<T>(
                        sequenceLength: seqLen,
                        embeddingDimension: embDim,
                        headCount: headCount,
                        useCausalMask: useCausalMask,
                        activationFunction: activation);
                    paged.EnableWeightOnlyQuantization = _config.EnableWeightOnlyQuantization;
                    paged.SetParameters(flash.GetParameters());

                    // Preserve positional encoding configuration from source FlashAttention layer
                    if (flash.PositionalEncoding != PositionalEncodingType.None)
                    {
                        paged.ConfigurePositionalEncoding(
                            flash.PositionalEncoding,
                            ropeTheta: flash.RoPETheta,
                            maxSequenceLength: seqLen);
                    }
                    else if (_config.PositionalEncoding == PositionalEncodingType.Rotary ||
                             _config.PositionalEncoding == PositionalEncodingType.ALiBi)
                    {
                        paged.ConfigurePositionalEncoding(
                            _config.PositionalEncoding,
                            ropeTheta: _config.RoPETheta,
                            maxSequenceLength: seqLen);
                    }

                    model.Layers[i] = paged;
                }
                else
                {
                    var cached = new CachedMultiHeadAttention<T>(
                        sequenceLength: seqLen,
                        embeddingDimension: embDim,
                        headCount: headCount,
                        useFlashAttention: enableFlashAttention,
                        layerIndex: 0,
                        useCausalMask: useCausalMask,
                        activationFunction: activation);
                    cached.SetParameters(flash.GetParameters());
                    model.Layers[i] = cached;
                }
                anyRewritten = true;
            }

            // Handle Grouped-Query Attention -> CachedGroupedQueryAttention
            // GQA rewrite: use CachedGroupedQueryAttention for regular KV cache.
            // Paged KV cache does not yet support GQA, so fall back to regular cache.
            if (layer is GroupedQueryAttentionLayer<T> gqa && enableKVCache)
            {
                var inputShape = gqa.GetInputShape();
                if (inputShape.Length < 2)
                    continue;

                int seqLen = inputShape[0];
                int embDim = inputShape[1];
                var activation = gqa.ScalarActivation;

                var cachedGqa = new CachedGroupedQueryAttention<T>(
                    sequenceLength: seqLen,
                    embeddingDimension: embDim,
                    numHeads: gqa.NumHeads,
                    numKVHeads: gqa.NumKVHeads,
                    useFlashAttention: enableFlashAttention,
                    layerIndex: 0,
                    useCausalMask: useCausalMask,
                    activationFunction: activation);
                cachedGqa.SetParameters(gqa.GetParameters());

                // Preserve positional encoding
                if (gqa.PositionalEncoding != PositionalEncodingType.None)
                {
                    cachedGqa.ConfigurePositionalEncoding(
                        gqa.PositionalEncoding,
                        ropeTheta: gqa.RoPETheta,
                        maxSequenceLength: seqLen);
                }
                else if (_config.PositionalEncoding == PositionalEncodingType.Rotary ||
                         _config.PositionalEncoding == PositionalEncodingType.ALiBi)
                {
                    cachedGqa.ConfigurePositionalEncoding(
                        _config.PositionalEncoding,
                        ropeTheta: _config.RoPETheta,
                        maxSequenceLength: seqLen);
                }

                model.Layers[i] = cachedGqa;
                anyRewritten = true;
            }
        }

        return anyRewritten;
    }

    private bool ApplyWeightOnlyQuantization(NeuralNetworkBase<T> model)
    {
        if (!_config.EnableWeightOnlyQuantization)
        {
            InferenceDiagnostics.RecordDecision("InferenceOptimizer", "WeightOnlyQuantization", enabled: false, reason: "DisabledByConfig");
            return false;
        }

        if (typeof(T) != typeof(float))
        {
            InferenceDiagnostics.RecordDecision("InferenceOptimizer", "WeightOnlyQuantization", enabled: false, reason: $"UnsupportedType({typeof(T).Name})");
            return false;
        }

        var mode = _config.InferenceQuantization;
        bool any = false;
        for (int i = 0; i < model.Layers.Count; i++)
        {
            // Quantize DenseLayer (existing)
            if (model.Layers[i] is DenseLayer<float> dense)
            {
                try
                {
                    var replacement = dense.VectorActivation != null
                        ? new QuantizedDenseLayer(dense, dense.VectorActivation)
                        : new QuantizedDenseLayer(dense);

                    if (replacement is ILayer<T> typedReplacement)
                    {
                        model.Layers[i] = typedReplacement;
                        any = true;
                    }
                }
                catch (InvalidOperationException ex)
                {
                    InferenceDiagnostics.RecordException("InferenceOptimizer", "WeightOnlyQuantization", ex, "DenseLayerQuantizationFailed;FallbackToFP");
                }
            }
            // Quantize MultiHeadAttentionLayer (supports INT8, FP8, NF4)
            else if (model.Layers[i] is MultiHeadAttentionLayer<float> mha)
            {
                try
                {
                    var replacement = new QuantizedAttentionLayer(mha, mode);
                    if (replacement is ILayer<T> typedReplacement)
                    {
                        model.Layers[i] = typedReplacement;
                        any = true;
                    }
                }
                catch (InvalidOperationException ex)
                {
                    InferenceDiagnostics.RecordException("InferenceOptimizer", "WeightOnlyQuantization", ex, "MHAQuantizationFailed;FallbackToFP");
                }
            }
            // Quantize GroupedQueryAttentionLayer (supports INT8, FP8, NF4)
            else if (model.Layers[i] is GroupedQueryAttentionLayer<float> gqa)
            {
                try
                {
                    var replacement = new QuantizedAttentionLayer(gqa, mode);
                    if (replacement is ILayer<T> typedReplacement)
                    {
                        model.Layers[i] = typedReplacement;
                        any = true;
                    }
                }
                catch (InvalidOperationException ex)
                {
                    InferenceDiagnostics.RecordException("InferenceOptimizer", "WeightOnlyQuantization", ex, "GQAQuantizationFailed;FallbackToFP");
                }
            }
        }

        string appliedTypes = any ? $"Applied({mode})" : "NoApplicableLayers";
        InferenceDiagnostics.RecordDecision("InferenceOptimizer", "WeightOnlyQuantization", enabled: any, reason: appliedTypes);
        return any;
    }

    private MultiHeadAttentionLayer<T>? TryConvertSelfAttentionToMultiHead(SelfAttentionLayer<T> layer)
    {
        var inputShape = layer.GetInputShape();
        if (inputShape.Length < 2)
            return null;

        int seqLen = inputShape[0];
        int embDim = inputShape[1];
        if (seqLen <= 0 || embDim <= 0)
            return null;

        int headCount = TryGetHeadCountFromMetadata(layer) ?? 0;
        if (headCount <= 0)
            return null;

        if (embDim % headCount != 0)
            return null;

        // SelfAttentionLayer has Q/K/V projections plus bias, but no output projection.
        // We convert it into a MultiHeadAttentionLayer with an identity output projection so that
        // downstream inference rewrites (FlashAttention / KV-cache) can be applied consistently.
        var activation = layer.ScalarActivation;
        var mha = new MultiHeadAttentionLayer<T>(seqLen, embDim, headCount, activationFunction: activation);

        var selfParams = layer.GetParameters();
        int projSize = embDim * embDim;
        int expectedSelf = (3 * projSize) + embDim;
        if (selfParams.Length != expectedSelf)
            return null;

        var numOps = MathHelper.GetNumericOperations<T>();

        // MultiHead params: Q, K, V, O, bias
        var combined = new Vector<T>((4 * projSize) + embDim);
        int idx = 0;

        // Copy Q/K/V (3 * projSize)
        for (int i = 0; i < 3 * projSize; i++)
            combined[idx++] = selfParams[i];

        // Output weights: identity matrix (embDim x embDim) flattened row-major
        for (int r = 0; r < embDim; r++)
        {
            for (int c = 0; c < embDim; c++)
            {
                combined[idx++] = r == c ? numOps.One : numOps.Zero;
            }
        }

        // Output bias (embDim)
        for (int i = 0; i < embDim; i++)
            combined[idx++] = selfParams[(3 * projSize) + i];

        mha.SetParameters(combined);
        return mha;
    }

    private static int? TryGetHeadCountFromMetadata(ILayer<T> layer)
    {
        if (layer is not LayerBase<T> layerBase)
            return null;

        if (!layerBase.GetMetadata().TryGetValue("HeadCount", out var raw) || string.IsNullOrWhiteSpace(raw))
            return null;

        return int.TryParse(raw, out var parsed) ? parsed : null;
    }

    private bool ResolveCausalMask(NeuralNetworkBase<T> model)
    {
        return _config.AttentionMasking switch
        {
            AttentionMaskingMode.Causal => true,
            AttentionMaskingMode.Disabled => false,
            _ => InferCausalFromModel(model)
        };
    }

    private bool InferCausalFromModel(NeuralNetworkBase<T> model)
    {
        // Default to causal when the user enables generation-oriented inference features.
        // This matches industry-standard expectations for autoregressive decoding and avoids
        // relying on users to set TaskType explicitly.
        if (_config.EnableKVCache || _config.EnableSpeculativeDecoding)
            return true;

        // Otherwise, keep heuristics conservative to avoid changing semantics for non-generative models.
        return model.Architecture.TaskType == NeuralNetworkTaskType.TextGeneration;
    }

    /// <summary>
    /// Estimates the maximum sequence length based on config and memory constraints.
    /// </summary>
    /// <param name="numLayers">Number of attention layers in the model.</param>
    /// <param name="numHeads">Number of attention heads per layer.</param>
    /// <param name="headDim">Dimension of each attention head.</param>
    /// <returns>Maximum sequence length that fits within the configured memory budget.</returns>
    private int EstimateMaxSequenceLength(int numLayers, int numHeads, int headDim)
    {
        // KV cache memory per token = numLayers * numHeads * headDim * 2 (K and V) * bytesPerElement
        // For batch size, multiply by maxBatchSize
        // Total: maxSeqLen * numLayers * numHeads * headDim * 2 * bytesPerElement * batchSize <= maxMemoryBytes

        long maxMemoryBytes = (long)_config.KVCacheMaxSizeMB * 1024 * 1024;

        // Estimate bytes per element based on type T
        int bytesPerElement = EstimateBytesPerElement();

        // Memory per token per batch item = numLayers * numHeads * headDim * 2 * bytesPerElement
        long memoryPerToken = (long)numLayers * numHeads * headDim * 2 * bytesPerElement;

        // Account for batch size
        long memoryPerTokenWithBatch = memoryPerToken * _config.MaxBatchSize;

        // Prevent division by zero
        if (memoryPerTokenWithBatch <= 0)
        {
            return 2048; // Reasonable default
        }

        // Calculate maximum sequence length
        long calculatedMaxSeqLen = maxMemoryBytes / memoryPerTokenWithBatch;

        // Apply reasonable bounds using MathHelper.Clamp for net471 compatibility
        const long minSeqLen = 128;
        const long maxSeqLen = 32768; // Reasonable upper bound

        return (int)MathHelper.Clamp(calculatedMaxSeqLen, minSeqLen, maxSeqLen);
    }

    /// <summary>
    /// Estimates bytes per element based on the generic type T.
    /// </summary>
    private static int EstimateBytesPerElement()
    {
        // Common numeric types used in neural networks
        var type = typeof(T);
        if (type == typeof(float)) return 4;
        if (type == typeof(double)) return 8;
        if (type == typeof(Half)) return 2;
        if (type == typeof(decimal)) return 16;

        // Default to float size if unknown
        return 4;
    }

    /// <summary>
    /// Sets up speculative decoding for autoregressive models.
    /// </summary>
    private bool InitializeSpeculativeDecoding(NeuralNetworkBase<T> model)
    {
        // Facade-friendly behavior: speculative decoding configuration must never crash inference.
        // If a requested draft model is unavailable, fall back to an N-gram draft model and record diagnostics.
        try
        {
            // For Custom draft models, an internal caller can provide one via SetCustomDraftModel().
            if (_config.DraftModelType == DraftModelType.Custom)
            {
                if (_draftModel != null)
                {
                    InferenceDiagnostics.RecordDecision("InferenceOptimizer", "SpeculativeDraftModel", enabled: true, reason: "CustomProvided");
                    return true;
                }

                _draftModel = CreateNGramDraftModel();
                InferenceDiagnostics.RecordDecision("InferenceOptimizer", "SpeculativeDraftModel", enabled: _draftModel != null, reason: "CustomNotProvided_FallbackToNGram");
                return _draftModel != null;
            }

            IDraftModel<T>? draftModel = _config.DraftModelType switch
            {
                DraftModelType.NGram => CreateNGramDraftModel(),
                DraftModelType.SmallNeural => CreateNeuralDraftModel(model),
                _ => CreateNGramDraftModel()
            };

            _draftModel = draftModel ?? CreateNGramDraftModel();
            InferenceDiagnostics.RecordDecision(
                "InferenceOptimizer",
                "SpeculativeDraftModel",
                enabled: _draftModel != null,
                reason: draftModel != null ? _config.DraftModelType.ToString() : $"Unavailable({_config.DraftModelType})_FallbackToNGram");

            return _draftModel != null;
        }
        catch (Exception ex)
        {
            InferenceDiagnostics.RecordException("InferenceOptimizer", "SpeculativeDecoding", ex, "Draft model init failed; falling back to NGram.");
            try
            {
                _draftModel = CreateNGramDraftModel();
                InferenceDiagnostics.RecordDecision("InferenceOptimizer", "SpeculativeDraftModel", enabled: _draftModel != null, reason: "ExceptionFallbackToNGram");
                return _draftModel != null;
            }
            catch
            {
                InferenceDiagnostics.RecordDecision("InferenceOptimizer", "SpeculativeDraftModel", enabled: false, reason: "FallbackFailed");
                _draftModel = null;
                return false;
            }
        }
    }

    /// <summary>
    /// Creates an N-gram based draft model.
    /// </summary>
    private IDraftModel<T>? CreateNGramDraftModel()
    {
        // NGram draft model with default settings
        return new NGramDraftModel<T>(ngramSize: 3);
    }

    /// <summary>
    /// Creates a small neural network draft model.
    /// </summary>
    /// <remarks>
    /// SmallNeural draft models require a pre-trained companion model that is smaller
    /// and faster than the target model but trained on similar data. This cannot be
    /// automatically generated from the target model.
    /// </remarks>
    /// <exception cref="NotSupportedException">
    /// Always thrown because SmallNeural draft models require external pre-trained models.
    /// </exception>
    private IDraftModel<T>? CreateNeuralDraftModel(NeuralNetworkBase<T> model)
    {
        // SmallNeural draft models require a separate pre-trained smaller model. We do not expose
        // draft model wiring via the public facade in the MVP, so treat this as unavailable.
        InferenceDiagnostics.RecordDecision("InferenceOptimizer", "SpeculativeDraftModel", enabled: false, reason: "SmallNeuralUnavailable_FallbackToNGram");
        return null;
    }

    /// <summary>
    /// Enables inference mode on the model for optimized prediction.
    /// </summary>
    /// <param name="model">The model to put in inference mode.</param>
    public void EnableInferenceMode(NeuralNetworkBase<T> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        // Enable inference mode on all applicable layers
        foreach (var layer in model.Layers)
        {
            if (layer is CachedMultiHeadAttention<T> cachedAttention)
            {
                cachedAttention.InferenceMode = true;
            }
            else if (layer is CachedGroupedQueryAttention<T> cachedGqa)
            {
                cachedGqa.InferenceMode = true;
            }
        }
    }

    /// <summary>
    /// Disables inference mode for training.
    /// </summary>
    /// <param name="model">The model to put in training mode.</param>
    public void DisableInferenceMode(NeuralNetworkBase<T> model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        foreach (var layer in model.Layers)
        {
            if (layer is CachedMultiHeadAttention<T> cachedAttention)
            {
                cachedAttention.InferenceMode = false;
            }
            else if (layer is PagedCachedMultiHeadAttention<T> pagedAttention)
            {
                pagedAttention.InferenceMode = false;
            }
            else if (layer is CachedGroupedQueryAttention<T> cachedGqa)
            {
                cachedGqa.InferenceMode = false;
            }
        }
    }

    /// <summary>
    /// Clears the KV cache. Call this when starting a new sequence.
    /// </summary>
    public void ClearCache()
    {
        _kvCache?.Clear();
        if (_pagedKVCache != null && _pagedSequenceId.HasValue)
        {
            try
            {
                _pagedKVCache.FreeSequence(_pagedSequenceId.Value);
            }
            catch
            {
                // Best-effort cleanup.
            }

            // Re-allocate with the same ID if possible; otherwise allocate a new one.
            if (!TryAllocatePagedSequenceId(_pagedKVCache, _pagedSequenceId.Value, initialTokens: 0, out long allocated))
            {
                InferenceDiagnostics.RecordDecision(
                    area: "InferenceOptimizer",
                    feature: "PagedKVCache",
                    enabled: false,
                    reason: "ClearCacheAllocateSequenceFailed(OutOfMemoryOrExhausted)");

                // Safe fallback: disable paged inference mode on layers and keep session alive.
                if (_pagedAttentionLayers != null)
                {
                    foreach (var layer in _pagedAttentionLayers)
                    {
                        layer.InferenceMode = false;
                        layer.Kernel = null;
                        layer.ResetState();
                    }
                }

                _pagedSequenceId = null;
                return;
            }

            _pagedSequenceId = allocated;

            if (_pagedAttentionLayers != null && _pagedSequenceId.HasValue)
            {
                foreach (var layer in _pagedAttentionLayers)
                {
                    layer.SequenceId = _pagedSequenceId.Value;
                    layer.ResetState();
                    layer.InferenceMode = true;
                    layer.Kernel ??= _pagedKernel;
                }
            }
        }
    }

    /// <summary>
    /// Gets statistics about the inference optimizer's state.
    /// </summary>
    /// <returns>Dictionary of statistics.</returns>
    public Dictionary<string, object> GetStatistics()
    {
        var stats = new Dictionary<string, object>
        {
            ["IsInitialized"] = _isInitialized,
            ["KVCacheEnabled"] = _config.EnableKVCache,
            ["SpeculativeDecodingEnabled"] = _config.EnableSpeculativeDecoding,
            ["BatchingEnabled"] = _config.EnableBatching,
            ["PagedKVCacheInitialized"] = _pagedKVCache != null,
            ["PagedAttentionLayerCount"] = _pagedAttentionLayers?.Count ?? 0,
            ["PagedAttentionWeightOnlyQuantizationEnabled"] = _pagedAttentionLayers?.Any(l => l.EnableWeightOnlyQuantization) ?? false,
            ["InferenceQuantizationMode"] = _config.InferenceQuantization.ToString(),
            ["PositionalEncoding"] = _config.PositionalEncoding.ToString()
        };

        if (_kvCache != null)
        {
            foreach (var kvp in _kvCache.GetStatistics())
            {
                stats[$"KVCache_{kvp.Key}"] = kvp.Value;
            }
        }

        if (_speculativeDecoder != null)
        {
            stats["SpeculationDepth"] = _config.SpeculationDepth;
            stats["DraftModelType"] = _config.DraftModelType.ToString();
        }

        return stats;
    }

    /// <summary>
    /// Gets the speculative decoder if enabled and created.
    /// </summary>
    public SpeculativeDecoder<T>? SpeculativeDecoder => _speculativeDecoder;

    /// <summary>
    /// Gets the draft model if speculative decoding is enabled.
    /// </summary>
    public IDraftModel<T>? DraftModel => _draftModel;

    /// <summary>
    /// Sets a custom draft model for speculative decoding.
    /// </summary>
    /// <param name="draftModel">The custom draft model implementation.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this method when you have your own draft model implementation.
    ///
    /// This is required when using DraftModelType.Custom or when you want to replace the
    /// default NGram draft model with a more sophisticated model.
    ///
    /// Your custom draft model must implement IDraftModel&lt;T&gt; and provide:
    /// - Draft token generation
    /// - Probability estimation for speculative decoding verification
    ///
    /// Example:
    /// <code>
    /// var optimizer = new InferenceOptimizer&lt;float&gt;(config);
    /// optimizer.SetCustomDraftModel(myCustomDraftModel);
    /// optimizer.Initialize(mainModel);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when draftModel is null.</exception>
    public void SetCustomDraftModel(IDraftModel<T> draftModel)
    {
        _draftModel = draftModel ?? throw new ArgumentNullException(nameof(draftModel));
    }

    /// <summary>
    /// Creates a speculative decoder with the given target forward function.
    /// </summary>
    /// <param name="targetForward">Function that runs the target model and returns probabilities.</param>
    /// <returns>The created SpeculativeDecoder, or null if speculative decoding is not enabled.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this to create the speculative decoder for text generation.
    ///
    /// The targetForward function should:
    /// - Take a Vector of token IDs (the full sequence so far)
    /// - Return a Matrix where targetForward[position, tokenId] = probability
    ///
    /// Example:
    /// <code>
    /// var decoder = optimizer.CreateSpeculativeDecoder(tokens => myModel.GetProbabilities(tokens));
    /// var result = decoder.Generate(inputTokens, maxNewTokens: 100, temperature);
    /// </code>
    /// </para>
    /// </remarks>
    public SpeculativeDecoder<T>? CreateSpeculativeDecoder(Func<Vector<int>, Matrix<T>> targetForward)
    {
        if (_draftModel == null || !_config.EnableSpeculativeDecoding)
        {
            return null;
        }

        var speculativeConfig = new SpeculativeDecodingConfig<T>
        {
            NumDraftTokens = _config.SpeculationDepth,
            UseTreeSpeculation = _config.UseTreeSpeculation ||
                                _config.SpeculativeMethod == SpeculativeMethod.Medusa ||
                                _config.SpeculativeMethod == SpeculativeMethod.Eagle,
            AdaptiveDraftLength = _config.SpeculationPolicy == SpeculationPolicy.Auto,
            TreeBranchFactor = _config.SpeculativeMethod == SpeculativeMethod.Medusa ? 4 : 2,
            MaxTreeDepth = Math.Max(1, _config.SpeculationDepth),
            MinAcceptanceRate = MathHelper.GetNumericOperations<T>().FromDouble(0.5)
        };

        _speculativeDecoder = new SpeculativeDecoder<T>(_draftModel, targetForward, speculativeConfig);
        return _speculativeDecoder;
    }
}
