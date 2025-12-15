using AiDotNet.Configuration;
using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.NeuralNetworks;
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
public class InferenceOptimizer<T>
{
    private readonly InferenceOptimizationConfig _config;
    private KVCache<T>? _kvCache;
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

        bool anyOptimizationsApplied = false;

        // Find and configure attention layers for KV caching
        if (_config.EnableKVCache)
        {
            anyOptimizationsApplied |= InitializeKVCache(model);
        }

        // Initialize speculative decoding if enabled
        if (_config.EnableSpeculativeDecoding)
        {
            anyOptimizationsApplied |= InitializeSpeculativeDecoding(model);
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
        int layerIndex = 0;

        foreach (var layer in model.Layers)
        {
            if (layer is CachedMultiHeadAttention<T> cachedAttention)
            {
                cachedAttention.LayerIndex = layerIndex;
                attentionLayers.Add(cachedAttention);
                layerIndex++;
            }
        }

        if (attentionLayers.Count == 0)
        {
            // No attention layers found - KV cache not applicable
            return false;
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
            PreAllocate = true
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
        // For Custom draft models, the user must call SetCustomDraftModel() before Initialize()
        if (_config.DraftModelType == DraftModelType.Custom)
        {
            if (_draftModel == null)
            {
                throw new InvalidOperationException(
                    "DraftModelType.Custom requires calling SetCustomDraftModel() before Initialize(). " +
                    "Provide your IDraftModel<T> implementation via SetCustomDraftModel(), then call Initialize().");
            }
            // Custom draft model already set via SetCustomDraftModel()
            return true;
        }

        // Create draft model based on configuration
        IDraftModel<T>? draftModel = _config.DraftModelType switch
        {
            DraftModelType.NGram => CreateNGramDraftModel(),
            DraftModelType.SmallNeural => CreateNeuralDraftModel(model),
            _ => throw new NotSupportedException($"Unknown DraftModelType: {_config.DraftModelType}")
        };

        // Note: SpeculativeDecoder requires a target forward function
        // This will be set when actually doing inference via CreateSpeculativeDecoder
        _draftModel = draftModel;
        return true;
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
        // SmallNeural draft models cannot be automatically created from the target model.
        // They require a separate pre-trained smaller model that approximates the target's behavior.
        // Use DraftModelType.NGram for automatic draft model creation, or
        // use DraftModelType.Custom and provide your own IDraftModel<T> implementation.
        throw new NotSupportedException(
            "DraftModelType.SmallNeural requires a pre-trained companion model that cannot be " +
            "automatically generated. Use DraftModelType.NGram for automatic draft model creation, " +
            "or implement IDraftModel<T> and use DraftModelType.Custom with SetCustomDraftModel().");
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
        }
    }

    /// <summary>
    /// Clears the KV cache. Call this when starting a new sequence.
    /// </summary>
    public void ClearCache()
    {
        _kvCache?.Clear();
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
            ["BatchingEnabled"] = _config.EnableBatching
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
            UseTreeSpeculation = _config.UseTreeSpeculation
        };

        _speculativeDecoder = new SpeculativeDecoder<T>(_draftModel, targetForward, speculativeConfig);
        return _speculativeDecoder;
    }
}
