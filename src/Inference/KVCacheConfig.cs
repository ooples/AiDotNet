namespace AiDotNet.Inference;

/// <summary>
/// Configuration for Key-Value cache used in autoregressive inference.
/// </summary>
/// <remarks>
/// <para>
/// KV-Cache is essential for efficient autoregressive generation (like in GPT models).
/// Without caching, each new token requires recomputing attention for all previous tokens.
/// With caching, we only compute attention for the new token and look up cached keys/values.
/// </para>
/// <para><b>For Beginners:</b> KV-Cache makes text generation much faster.
///
/// When generating text token by token:
/// - Without cache: Generate token 100 by processing tokens 1-99 again (slow!)
/// - With cache: Generate token 100 using cached computations from tokens 1-99 (fast!)
///
/// This can speed up generation by 10-100x for long sequences.
///
/// The cache stores the Key and Value projections from attention layers,
/// which don't change once computed for a given position.
/// </para>
/// </remarks>
internal class KVCacheConfig
{
    /// <summary>
    /// Maximum sequence length the cache can hold.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Pre-allocates memory for this many tokens. Choose based on your use case:
    /// - Chatbots: 2048-4096
    /// - Long documents: 8192-32768
    /// - Code generation: 4096-8192
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>
    /// Number of transformer layers to cache.
    /// </summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Number of attention heads per layer.
    /// </summary>
    public int NumHeads { get; set; } = 12;

    /// <summary>
    /// Dimension of each attention head.
    /// </summary>
    public int HeadDimension { get; set; } = 64;

    /// <summary>
    /// Maximum batch size for the cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For serving multiple requests, set this to your maximum concurrent batch size.
    /// Memory usage scales linearly with batch size.
    /// </para>
    /// </remarks>
    public int MaxBatchSize { get; set; } = 1;

    /// <summary>
    /// Whether to use sliding window attention (for long sequences).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, only the most recent WindowSize tokens are kept in cache.
    /// Older tokens are evicted. This limits memory usage for very long sequences.
    /// </para>
    /// </remarks>
    public bool UseSlidingWindow { get; set; } = false;

    /// <summary>
    /// Size of sliding window (if enabled).
    /// </summary>
    public int WindowSize { get; set; } = 1024;

    /// <summary>
    /// Data type for cache storage.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Using FP16 halves memory usage with minimal accuracy loss.
    /// Recommended for inference, especially on GPUs.
    /// </para>
    /// </remarks>
    public CacheDataType DataType { get; set; } = CacheDataType.Float32;

    /// <summary>
    /// Whether to pre-allocate all memory at initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Pre-allocation is faster during inference but uses more memory upfront.
    /// Dynamic allocation saves memory but may cause fragmentation.
    /// </para>
    /// </remarks>
    public bool PreAllocate { get; set; } = true;

    /// <summary>
    /// Device placement for the cache (CPU or GPU).
    /// </summary>
    public CacheDevice Device { get; set; } = CacheDevice.Auto;

    /// <summary>
    /// Computes the total memory required for the cache in bytes.
    /// </summary>
    public long EstimateMemoryBytes()
    {
        long elementsPerLayer = (long)MaxBatchSize * NumHeads * MaxSequenceLength * HeadDimension;
        long totalElements = elementsPerLayer * NumLayers * 2; // K and V

        int bytesPerElement = DataType switch
        {
            CacheDataType.Int8 => 1,
            CacheDataType.Float16 => 2,
            CacheDataType.Float32 => 4,
            CacheDataType.Float64 => 8,
            CacheDataType.BFloat16 => 2,
            _ => 4
        };

        return totalElements * bytesPerElement;
    }

    /// <summary>
    /// Creates a default configuration for common model sizes.
    /// </summary>
    public static KVCacheConfig ForModel(string modelSize)
    {
        return modelSize.ToLowerInvariant() switch
        {
            "gpt2" or "small" => new KVCacheConfig
            {
                NumLayers = 12,
                NumHeads = 12,
                HeadDimension = 64,
                MaxSequenceLength = 1024
            },
            "gpt2-medium" or "medium" => new KVCacheConfig
            {
                NumLayers = 24,
                NumHeads = 16,
                HeadDimension = 64,
                MaxSequenceLength = 1024
            },
            "gpt2-large" or "large" => new KVCacheConfig
            {
                NumLayers = 36,
                NumHeads = 20,
                HeadDimension = 64,
                MaxSequenceLength = 1024
            },
            "llama-7b" => new KVCacheConfig
            {
                NumLayers = 32,
                NumHeads = 32,
                HeadDimension = 128,
                MaxSequenceLength = 4096,
                DataType = CacheDataType.Float16
            },
            "llama-13b" => new KVCacheConfig
            {
                NumLayers = 40,
                NumHeads = 40,
                HeadDimension = 128,
                MaxSequenceLength = 4096,
                DataType = CacheDataType.Float16
            },
            "llama-70b" => new KVCacheConfig
            {
                NumLayers = 80,
                NumHeads = 64,
                HeadDimension = 128,
                MaxSequenceLength = 4096,
                DataType = CacheDataType.Float16,
                UseSlidingWindow = true,
                WindowSize = 2048
            },
            _ => new KVCacheConfig()
        };
    }
}

/// <summary>
/// Data types supported for KV-Cache storage.
/// </summary>
internal enum CacheDataType
{
    /// <summary>Signed 8-bit integer quantization (int8) with scaling.</summary>
    Int8,

    /// <summary>Half precision (16-bit float).</summary>
    Float16,

    /// <summary>Single precision (32-bit float).</summary>
    Float32,

    /// <summary>Double precision (64-bit float).</summary>
    Float64,

    /// <summary>Brain float 16 (used by TPUs).</summary>
    BFloat16
}

/// <summary>
/// Device placement options for KV-Cache.
/// </summary>
internal enum CacheDevice
{
    /// <summary>Automatically select based on available hardware.</summary>
    Auto,

    /// <summary>Store cache in CPU memory.</summary>
    CPU,

    /// <summary>Store cache in GPU memory.</summary>
    GPU
}
