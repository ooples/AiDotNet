namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for inference-time optimizations to maximize prediction throughput and efficiency.
/// </summary>
/// <remarks>
/// <para>
/// This configuration controls advanced inference optimizations including KV caching for transformers,
/// request batching for throughput, and speculative decoding for faster autoregressive generation.
/// These optimizations are automatically applied during prediction based on your configuration.
/// </para>
/// <para><b>For Beginners:</b> Inference optimization makes your model's predictions faster and more efficient.
///
/// Key features:
/// - <b>KV Cache:</b> Remembers previous computations in attention layers (2-10x faster for long sequences)
/// - <b>Batching:</b> Groups multiple predictions together (higher throughput)
/// - <b>Speculative Decoding:</b> Uses a small model to draft tokens, then verifies (1.5-3x faster generation)
///
/// Default settings are optimized for most use cases. Simply enable and let the library handle the rest.
///
/// Example:
/// <code>
/// var config = InferenceOptimizationConfig.Default;
///
/// var result = await new PredictionModelBuilder&lt;double, ...&gt;()
///     .ConfigureModel(myModel)
///     .ConfigureInferenceOptimizations(config)
///     .BuildAsync(x, y);
/// </code>
/// </para>
/// </remarks>
public class InferenceOptimizationConfig
{
    /// <summary>
    /// Gets a default configuration with sensible settings for most use cases.
    /// </summary>
    /// <remarks>
    /// Default settings:
    /// - KV Cache: Enabled for transformer models, 1GB max size
    /// - Batching: Enabled with adaptive batch sizing
    /// - Speculative Decoding: Disabled (requires explicit configuration)
    /// </remarks>
    public static InferenceOptimizationConfig Default => new()
    {
        EnableKVCache = true,
        EnableBatching = true,
        EnableSpeculativeDecoding = false
    };

    /// <summary>
    /// Gets a high-performance configuration optimized for maximum throughput.
    /// </summary>
    /// <remarks>
    /// All optimizations enabled with aggressive settings:
    /// - KV Cache: Enabled with 2GB max size
    /// - Batching: Enabled with larger batch sizes
    /// - Speculative Decoding: Enabled with NGram draft model
    /// </remarks>
    public static InferenceOptimizationConfig HighPerformance => new()
    {
        EnableKVCache = true,
        KVCacheMaxSizeMB = 2048,
        EnableBatching = true,
        MaxBatchSize = 64,
        EnableSpeculativeDecoding = true,
        SpeculationDepth = 5
    };

    #region KV Cache Settings

    /// <summary>
    /// Gets or sets whether KV (Key-Value) caching is enabled for attention layers.
    /// </summary>
    /// <value>True to enable KV caching (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> KV caching speeds up transformer models by remembering previous computations.
    ///
    /// How it works:
    /// - Attention layers compute keys and values for each token
    /// - Without caching: Recomputes all keys/values for every new token
    /// - With caching: Stores previous keys/values, only computes for new tokens
    ///
    /// Benefits:
    /// - 2-10x faster for long sequences
    /// - Essential for autoregressive generation (GPT-style)
    /// - Minimal memory overhead for huge speedup
    ///
    /// When to disable:
    /// - Memory-constrained environments
    /// - Very short sequences (overhead exceeds benefit)
    /// - Non-transformer models (no effect)
    /// </para>
    /// </remarks>
    public bool EnableKVCache { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum KV cache size in megabytes.
    /// </summary>
    /// <value>Maximum cache size in MB (default: 1024 = 1GB).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how much memory the KV cache can use.
    ///
    /// Guidelines:
    /// - 512MB: Good for small models or memory-constrained systems
    /// - 1024MB (default): Balanced for most use cases
    /// - 2048MB+: For large models or long sequences
    ///
    /// When cache fills up, oldest entries are evicted (LRU policy).
    /// </para>
    /// </remarks>
    public int KVCacheMaxSizeMB { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the KV cache eviction policy.
    /// </summary>
    /// <value>Cache eviction policy (default: LRU).</value>
    public CacheEvictionPolicy KVCacheEvictionPolicy { get; set; } = CacheEvictionPolicy.LRU;

    #endregion

    #region Batching Settings

    /// <summary>
    /// Gets or sets whether request batching is enabled.
    /// </summary>
    /// <value>True to enable batching (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Batching groups multiple predictions together for efficiency.
    ///
    /// Benefits:
    /// - Higher throughput (more predictions per second)
    /// - Better GPU utilization
    /// - Lower per-request latency under load
    ///
    /// How it works:
    /// - Incoming prediction requests are queued
    /// - When batch is full OR timeout reached, batch is processed together
    /// - Results are returned to each caller
    ///
    /// Trade-offs:
    /// - Slight latency increase for single requests (waiting for batch)
    /// - Significant throughput increase under load
    /// </para>
    /// </remarks>
    public bool EnableBatching { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum batch size for grouped predictions.
    /// </summary>
    /// <value>Maximum batch size (default: 32).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many predictions to group together.
    ///
    /// Guidelines:
    /// - 8-16: Good for memory-constrained systems
    /// - 32 (default): Balanced for most cases
    /// - 64+: For high-throughput GPU inference
    ///
    /// Larger batches = better throughput but more memory.
    /// </para>
    /// </remarks>
    public int MaxBatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the minimum batch size before processing.
    /// </summary>
    /// <value>Minimum batch size (default: 1).</value>
    public int MinBatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the maximum time to wait for batch to fill in milliseconds.
    /// </summary>
    /// <value>Batch timeout in milliseconds (default: 10ms).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How long to wait before processing a partial batch.
    ///
    /// Lower values = lower latency but smaller batches.
    /// Higher values = larger batches but more waiting.
    /// </para>
    /// </remarks>
    public int BatchTimeoutMs { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether adaptive batch sizing is enabled.
    /// </summary>
    /// <value>True to enable adaptive sizing (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Automatically adjusts batch size based on system load.
    ///
    /// When enabled:
    /// - Low load: Smaller batches for lower latency
    /// - High load: Larger batches for higher throughput
    /// - Automatically balances latency vs throughput
    /// </para>
    /// </remarks>
    public bool AdaptiveBatchSize { get; set; } = true;

    #endregion

    #region Validation

    /// <summary>
    /// Validates the configuration and throws if any values are invalid.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when configuration values are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this method to ensure your configuration is valid before use.
    ///
    /// Validation rules:
    /// - KVCacheMaxSizeMB must be positive
    /// - MaxBatchSize must be positive
    /// - MinBatchSize must be positive and not exceed MaxBatchSize
    /// - BatchTimeoutMs must be non-negative
    /// - SpeculationDepth must be non-negative
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (KVCacheMaxSizeMB <= 0)
        {
            throw new InvalidOperationException(
                $"KVCacheMaxSizeMB must be positive. Got: {KVCacheMaxSizeMB}");
        }

        if (MaxBatchSize <= 0)
        {
            throw new InvalidOperationException(
                $"MaxBatchSize must be positive. Got: {MaxBatchSize}");
        }

        if (MinBatchSize <= 0)
        {
            throw new InvalidOperationException(
                $"MinBatchSize must be positive. Got: {MinBatchSize}");
        }

        if (MinBatchSize > MaxBatchSize)
        {
            throw new InvalidOperationException(
                $"MinBatchSize ({MinBatchSize}) cannot exceed MaxBatchSize ({MaxBatchSize}).");
        }

        if (BatchTimeoutMs < 0)
        {
            throw new InvalidOperationException(
                $"BatchTimeoutMs must be non-negative. Got: {BatchTimeoutMs}");
        }

        if (SpeculationDepth < 0)
        {
            throw new InvalidOperationException(
                $"SpeculationDepth must be non-negative. Got: {SpeculationDepth}");
        }
    }

    #endregion

    #region Speculative Decoding Settings

    /// <summary>
    /// Gets or sets whether speculative decoding is enabled.
    /// </summary>
    /// <value>True to enable speculative decoding (default: false).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Speculative decoding speeds up autoregressive generation (GPT-style).
    ///
    /// How it works:
    /// 1. A small "draft" model quickly generates candidate tokens
    /// 2. The main model verifies all candidates in one pass
    /// 3. Accepted tokens are kept, rejected ones are regenerated
    ///
    /// Benefits:
    /// - 1.5-3x faster generation for LLMs
    /// - No quality loss (verification ensures correctness)
    ///
    /// Requirements:
    /// - Autoregressive model (generates tokens sequentially)
    /// - Draft model must be available (NGram or smaller neural network)
    ///
    /// When to disable:
    /// - Non-autoregressive models
    /// - Single-pass predictions
    /// - When draft model overhead exceeds benefit
    /// </para>
    /// </remarks>
    public bool EnableSpeculativeDecoding { get; set; } = false;

    /// <summary>
    /// Gets or sets the type of draft model to use for speculative decoding.
    /// </summary>
    /// <value>Draft model type (default: NGram).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The draft model generates candidate tokens quickly.
    ///
    /// Options:
    /// - <b>NGram:</b> Simple statistical model (fast, no GPU needed)
    /// - <b>SmallNeural:</b> Smaller version of the main model (more accurate drafts)
    ///
    /// NGram is usually sufficient and has near-zero overhead.
    /// </para>
    /// </remarks>
    public DraftModelType DraftModelType { get; set; } = DraftModelType.NGram;

    /// <summary>
    /// Gets or sets the speculation depth (number of tokens to draft ahead).
    /// </summary>
    /// <value>Speculation depth (default: 4).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many tokens the draft model predicts at once.
    ///
    /// Guidelines:
    /// - 3-4: Conservative, high acceptance rate
    /// - 5-6: Balanced (default: 4)
    /// - 7+: Aggressive, may have more rejections
    ///
    /// Higher depth = more speedup potential but more wasted work on rejections.
    /// </para>
    /// </remarks>
    public int SpeculationDepth { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to use tree-structured speculation.
    /// </summary>
    /// <value>True to enable tree speculation (default: false).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree speculation generates multiple candidate sequences in parallel.
    ///
    /// Instead of one sequence of draft tokens, generates a tree of possibilities.
    /// Can improve acceptance rate but uses more memory.
    /// </para>
    /// </remarks>
    public bool UseTreeSpeculation { get; set; } = false;

    #endregion
}

/// <summary>
/// Cache eviction policies for KV cache management.
/// </summary>
public enum CacheEvictionPolicy
{
    /// <summary>Least Recently Used - evicts entries that haven't been accessed recently.</summary>
    LRU,
    /// <summary>First In First Out - evicts oldest entries first.</summary>
    FIFO,
    /// <summary>Least Frequently Used - evicts entries with lowest access count.</summary>
    LFU
}

/// <summary>
/// Types of draft models for speculative decoding.
/// </summary>
public enum DraftModelType
{
    /// <summary>N-gram based statistical model (fast, no GPU).</summary>
    NGram,
    /// <summary>Small neural network model (more accurate, uses GPU).</summary>
    SmallNeural,
    /// <summary>Custom user-provided draft model.</summary>
    Custom
}
