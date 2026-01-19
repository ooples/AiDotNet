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
/// var result = await new AiModelBuilder&lt;double, ...&gt;()
///     .ConfigureModel(myModel)
///     .ConfigureInferenceOptimizations(config)
///     .BuildAsync();
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

    /// <summary>
    /// Gets or sets whether to use a sliding window KV-cache for long contexts.
    /// </summary>
    /// <remarks>
    /// When enabled, only the most recent <see cref="KVCacheWindowSize"/> tokens are kept.
    /// This is a common industry approach for long-context serving to cap memory usage.
    /// </remarks>
    public bool UseSlidingWindowKVCache { get; set; } = false;

    /// <summary>
    /// Gets or sets the sliding window size in tokens when <see cref="UseSlidingWindowKVCache"/> is enabled.
    /// </summary>
    /// <value>Window size in tokens (default: 1024).</value>
    public int KVCacheWindowSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the precision used for KV-cache storage.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Industry-standard serving stores KV-cache in FP16 to halve memory usage and increase cache capacity.
    /// The default <see cref="KVCachePrecisionMode.Auto"/> selects FP16 when KV-cache is enabled and the numeric
    /// type supports it.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This setting controls how much memory your model uses during autoregressive inference.
    ///
    /// - FP16: Uses about half the memory (recommended default)
    /// - FP32: Uses more memory but can be slightly more numerically accurate
    ///
    /// Most production systems prefer FP16 KV-cache for capacity and throughput.
    /// </para>
    /// </remarks>
    public KVCachePrecisionMode KVCachePrecision { get; set; } = KVCachePrecisionMode.Auto;

    /// <summary>
    /// Gets or sets the quantization mode used for KV-cache storage.
    /// </summary>
    /// <remarks>
    /// <para>
    /// KV-cache quantization can further reduce memory beyond FP16 by storing keys/values in int8 with scaling.
    /// This is an opt-in advanced feature because it can introduce small numerical error.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// - None (default): Store KV-cache in FP16/FP32 depending on <see cref="KVCachePrecision"/>.
    /// - Int8: Store KV-cache in 8-bit integers to save memory (advanced).
    /// </para>
    /// </remarks>
    public KVCacheQuantizationMode KVCacheQuantization { get; set; } = KVCacheQuantizationMode.None;

    /// <summary>
    /// Gets or sets whether to use a paged KV-cache backend (vLLM-style) for long-context / multi-sequence serving.
    /// </summary>
    /// <remarks>
    /// When enabled, the system may choose a paged cache implementation that allocates KV memory in fixed-size blocks.
    /// This is the industry-standard approach for high-throughput serving where many sequences are active concurrently.
    /// Users can disable this to force the traditional contiguous KV-cache.
    /// </remarks>
    public bool EnablePagedKVCache { get; set; } = true;

    /// <summary>
    /// Gets or sets the block size (in tokens) for the paged KV-cache when enabled.
    /// </summary>
    /// <remarks>
    /// Common values are 16 or 32. Smaller blocks reduce internal fragmentation; larger blocks reduce table overhead.
    /// </remarks>
    public int PagedKVCacheBlockSize { get; set; } = 16;

    #endregion

    #region Attention Settings

    /// <summary>
    /// Gets or sets whether Flash Attention is enabled (when applicable).
    /// </summary>
    /// <remarks>
    /// Flash Attention computes exact attention without materializing the full N×N attention matrix,
    /// reducing memory bandwidth pressure and improving throughput for long sequences.
    /// </remarks>
    public bool EnableFlashAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets how attention masking should be applied for optimized attention implementations.
    /// </summary>
    /// <remarks>
    /// - Auto: Applies causal masking for known autoregressive models (e.g., text generation), otherwise no mask.
    /// - Disabled: Never applies causal masking.
    /// - Causal: Always applies causal masking (GPT-style).
    /// </remarks>
    public AttentionMaskingMode AttentionMasking { get; set; } = AttentionMaskingMode.Auto;

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

        if (UseSlidingWindowKVCache && KVCacheWindowSize <= 0)
        {
            throw new InvalidOperationException(
                $"KVCacheWindowSize must be positive when UseSlidingWindowKVCache is enabled. Got: {KVCacheWindowSize}");
        }

        if (EnablePagedKVCache && PagedKVCacheBlockSize <= 0)
        {
            throw new InvalidOperationException(
                $"PagedKVCacheBlockSize must be positive when EnablePagedKVCache is enabled. Got: {PagedKVCacheBlockSize}");
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
    /// - <b>SmallNeural:</b> Smaller companion model (more accurate drafts)
    ///
    /// NGram is usually sufficient and has near-zero overhead.
    ///
    /// <para>
    /// <b>Note:</b> Small neural draft models require an external companion model. In the MVP, the library
    /// falls back to <see cref="DraftModelType.NGram"/> when a companion draft model is not available.
    /// </para>
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

    /// <summary>
    /// Gets or sets the policy for when speculative decoding should run.
    /// </summary>
    /// <remarks>
    /// Auto is recommended: it can back off speculative decoding under high load (e.g., large batches)
    /// to avoid throughput regressions, while still enabling it for latency-sensitive scenarios.
    /// </remarks>
    public SpeculationPolicy SpeculationPolicy { get; set; } = SpeculationPolicy.Auto;

    /// <summary>
    /// Gets or sets the speculative decoding method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default <see cref="SpeculativeMethod.Auto"/> currently selects <see cref="SpeculativeMethod.ClassicDraftModel"/>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This chooses the "style" of speculative decoding.
    /// </para>
    /// </remarks>
    public SpeculativeMethod SpeculativeMethod { get; set; } = SpeculativeMethod.Auto;

    #endregion

    #region Inference Quantization (Advanced)

    /// <summary>
    /// Gets or sets whether weight-only INT8 quantization is enabled for inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Weight-only quantization reduces memory bandwidth and improves cache locality by storing weights in int8
    /// with per-output scaling. Activations remain in FP32/FP16, and accumulation is performed in float.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This makes your model weights smaller so the CPU/GPU can read them faster.
    /// </para>
    /// <para>
    /// This is disabled by default until validated across more layer types and kernels. When enabled, the optimizer
    /// will apply it opportunistically and fall back safely when unsupported.
    /// </para>
    /// </remarks>
    public bool EnableWeightOnlyQuantization { get; set; } = false;

    #endregion
}

/// <summary>
/// Policies for enabling/disabling speculative decoding at runtime.
/// </summary>
public enum SpeculationPolicy
{
    /// <summary>
    /// Automatically decide based on runtime conditions (recommended).
    /// </summary>
    Auto,

    /// <summary>
    /// Always enable speculative decoding when configured.
    /// </summary>
    ForceOn,

    /// <summary>
    /// Always disable speculative decoding even if enabled in config.
    /// </summary>
    ForceOff,

    /// <summary>
    /// Prefer speculative decoding to reduce latency, even under moderate load.
    /// </summary>
    LatencyFirst,

    /// <summary>
    /// Prefer throughput and stability: use speculative decoding only when conditions are ideal.
    /// </summary>
    ThroughputFirst
}

/// <summary>
/// Selects the speculative decoding method.
/// </summary>
public enum SpeculativeMethod
{
    /// <summary>
    /// Automatically select the best available method (defaults to ClassicDraftModel today).
    /// </summary>
    Auto,

    /// <summary>
    /// Classic draft-model speculative decoding (standard).
    /// </summary>
    ClassicDraftModel,

    /// <summary>
    /// Medusa-style multi-head proposals (hook for future internal implementation).
    /// </summary>
    Medusa,

    /// <summary>
    /// EAGLE-style enhanced draft proposals (hook for future internal implementation).
    /// </summary>
    Eagle
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
    /// <summary>Custom draft model (internal/serving integration).</summary>
    Custom
}

/// <summary>
/// Controls how attention masking is applied for optimized attention implementations.
/// </summary>
public enum AttentionMaskingMode
{
    /// <summary>
    /// Automatically select masking based on model/task heuristics.
    /// </summary>
    Auto,

    /// <summary>
    /// Do not apply causal masking.
    /// </summary>
    Disabled,

    /// <summary>
    /// Apply causal masking (autoregressive decoding).
    /// </summary>
    Causal
}

/// <summary>
/// Controls the numeric precision of KV-cache storage.
/// </summary>
public enum KVCachePrecisionMode
{
    /// <summary>
    /// Select an industry-standard default.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses FP16 when KV-cache is enabled and the numeric type supports conversion; otherwise falls back to FP32.
    /// </para>
    /// </remarks>
    Auto,

    /// <summary>
    /// Store KV-cache in FP16 (half precision) to reduce memory use.
    /// </summary>
    Float16,

    /// <summary>
    /// Store KV-cache in FP32 (single precision) for maximal numerical fidelity.
    /// </summary>
    Float32
}

/// <summary>
/// Controls optional KV-cache quantization for inference.
/// </summary>
public enum KVCacheQuantizationMode
{
    /// <summary>No quantization (default).</summary>
    None,

    /// <summary>Signed int8 quantization with scaling (advanced, opt-in).</summary>
    Int8
}
