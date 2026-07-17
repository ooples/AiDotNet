using System;
using AiDotNet.Configuration;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// Translates the library-wide <see cref="InferenceOptimizationConfig"/> (the facade builder's
/// <c>ConfigureInferenceOptimizations</c> surface) into concrete <see cref="EngineOptions"/> for the serving
/// engine. This is the bridge that makes "vLLM speed by default" real: a beginner sets nothing and the engine
/// is configured from the same defaults the rest of the library already uses.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> you configure inference once, in one place (<c>ConfigureInferenceOptimizations</c>),
/// and every part of the library — including this serving engine — reads those settings. This class does the
/// translation so you never hand-tune the engine's KV pool or batch limits.</para>
/// </remarks>
public static class ServingConfigMapper
{
    /// <summary>
    /// Builds engine options from an inference-optimization config. When no config is supplied,
    /// <see cref="InferenceOptimizationConfig.Default"/> is used.
    /// </summary>
    /// <param name="config">The inference config (null ⇒ defaults).</param>
    /// <param name="eosTokenId">The model/tokenizer EOS token id, if known.</param>
    /// <param name="maxContextTokens">Assumed maximum context length used to size the KV block pool so the
    /// full batch fits at that length. Defaults to 4096.</param>
    public static EngineOptions ToEngineOptions(
        InferenceOptimizationConfig? config,
        int? eosTokenId = null,
        int maxContextTokens = 4096)
    {
        var c = config ?? InferenceOptimizationConfig.Default;
        if (maxContextTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxContextTokens));

        int blockSize = c.PagedKVCacheBlockSize >= 1 ? c.PagedKVCacheBlockSize : 16;
        int maxSequences = c.EnableBatching ? Math.Max(1, c.MaxBatchSize) : 1;

        // Size the block pool so the whole batch can reach the assumed context length, plus a generation slot.
        int blocksPerSequence = (maxContextTokens + blockSize - 1) / blockSize + 1;
        int numKvBlocks = Math.Max(blocksPerSequence, checked(maxSequences * blocksPerSequence));

        // Quantized KV takes less memory per token, so the same budget holds more blocks (higher concurrency /
        // longer contexts). Int8 ≈ half the bytes of fp16 ⇒ ~2× the block capacity.
        int quantFactor = c.KVCacheQuantization == KVCacheQuantizationMode.Int8 ? 2 : 1;
        numKvBlocks = checked(numKvBlocks * quantFactor);

        // Per-step compute budget: enough for one full-context prefill, or one decode token per running seq.
        int maxBatchedTokens = Math.Max(maxContextTokens, maxSequences);

        return new EngineOptions
        {
            BlockSize = blockSize,
            MaxNumSequences = maxSequences,
            NumKvBlocks = numKvBlocks,
            MaxBatchedTokens = maxBatchedTokens,
            EosTokenId = eosTokenId,
            KvCacheQuantization = c.KVCacheQuantization,
        };
    }

    /// <summary>True if the config asks for speculative decoding.</summary>
    public static bool IsSpeculativeEnabled(InferenceOptimizationConfig? config)
        => (config ?? InferenceOptimizationConfig.Default).EnableSpeculativeDecoding;

    /// <summary>The configured speculation depth (draft tokens per round), floored at 1.</summary>
    public static int SpeculationDepth(InferenceOptimizationConfig? config)
        => Math.Max(1, (config ?? InferenceOptimizationConfig.Default).SpeculationDepth);
}
