using System;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// Tuning knobs for the <see cref="ContinuousBatchingEngine{T}"/>. Every value has an industry-standard default,
/// so the engine runs well with <c>new EngineOptions()</c>; the facade builder derives these automatically from
/// the model and <see cref="AiDotNet.Configuration.InferenceOptimizationConfig"/> so a beginner never sets them.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> these control how many requests run at once and how much KV-cache memory the
/// engine may use. You almost never need to touch them — the library picks good values for your model and
/// hardware. They exist so an expert can cap memory or concurrency for a specific deployment.</para>
/// </remarks>
public sealed class EngineOptions
{
    /// <summary>Maximum number of sequences running (being decoded) concurrently. Default 256.</summary>
    public int MaxNumSequences { get; init; } = 256;

    /// <summary>Maximum tokens processed across a single batch (prefill + decode). Bounds per-step compute. Default 8192.</summary>
    public int MaxBatchedTokens { get; init; } = 8192;

    /// <summary>Token slots per KV-cache block. Default 16 (vLLM's default).</summary>
    public int BlockSize { get; init; } = 16;

    /// <summary>Number of KV-cache blocks in the pool (sizes total KV memory). Default 4096.</summary>
    public int NumKvBlocks { get; init; } = 4096;

    /// <summary>Model EOS token id; generating it ends a sequence (unless <see cref="SamplingParameters.IgnoreEos"/>). Null = none.</summary>
    public int? EosTokenId { get; init; }

    /// <summary>
    /// KV-cache storage precision. Int8 lets a paged runner store roughly twice as much KV in the same memory,
    /// so more sequences / longer contexts fit at once. Informational for the recompute path (no physical KV).
    /// </summary>
    public AiDotNet.Configuration.KVCacheQuantizationMode KvCacheQuantization { get; init; }
        = AiDotNet.Configuration.KVCacheQuantizationMode.None;

    /// <summary>
    /// Enables automatic cross-request prefix caching: a new request that begins with the same tokens as a
    /// previous one reuses that prompt's cached KV instead of recomputing it (block-aligned prefixes only).
    /// Off by default. On the paged fast path this is a real compute saving.
    /// </summary>
    public bool EnablePrefixCache { get; init; }

    /// <summary>Maximum number of cached prefixes (LRU eviction beyond it). Default 128.</summary>
    public int PrefixCacheCapacity { get; init; } = 128;

    /// <summary>Validates the options, throwing <see cref="ArgumentException"/> on an invalid value.</summary>
    public void Validate()
    {
        if (MaxNumSequences < 1) throw new ArgumentException("MaxNumSequences must be >= 1.", nameof(MaxNumSequences));
        if (MaxBatchedTokens < 1) throw new ArgumentException("MaxBatchedTokens must be >= 1.", nameof(MaxBatchedTokens));
        if (BlockSize < 1) throw new ArgumentException("BlockSize must be >= 1.", nameof(BlockSize));
        if (NumKvBlocks < 1) throw new ArgumentException("NumKvBlocks must be >= 1.", nameof(NumKvBlocks));
    }
}
