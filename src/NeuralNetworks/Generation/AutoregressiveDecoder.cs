// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Generation;

/// <summary>
/// Generic autoregressive decode loop (#1632 / #95): the reusable "generate" driver the codebase
/// lacked — GPT4Vision / Blip / Flamingo each hand-rolled this loop. It owns the loop + EOS + token
/// feedback + RNG lifetime; the model supplies the per-step "embed the token, run the (KV-cached)
/// forward, return next-token logits" via the <c>stepLogits</c> delegate. Because the cached
/// attention layers append to the KV cache inside that forward, each step only pays for the new
/// token instead of recomputing the prefix (proven equivalent by KVCacheDecodeEquivalenceTests).
/// </summary>
/// <typeparam name="T">Numeric type of the logits.</typeparam>
public static class AutoregressiveDecoder<T>
{
    /// <summary>
    /// Greedily/stochastically decodes up to <paramref name="maxNewTokens"/> tokens.
    /// </summary>
    /// <param name="stepLogits">Produces the next-token logits. Called with <c>null</c> on the first
    /// (prefill) step, then with the previously-sampled token id on each subsequent step — the model
    /// embeds it, runs its forward (which advances the KV cache), and returns the logits.</param>
    /// <param name="maxNewTokens">Maximum tokens to generate (≥ 0).</param>
    /// <param name="options">Sampling options; null ⇒ <see cref="SamplingOptions.Default"/>.</param>
    /// <param name="isEndToken">Optional EOS predicate; generation stops (without emitting it) when true.</param>
    /// <param name="suppressToken">Optional predicate for tokens to suppress (e.g. PAD): when a sampled
    /// token matches, it is neither emitted nor fed back, but the step is still consumed (the model
    /// re-samples on the next step from the unchanged prefix). Mirrors a <c>continue</c> in a hand-rolled
    /// fixed-length loop — the same semantics as HuggingFace's <c>suppress_tokens</c>.</param>
    /// <returns>The generated token ids, in order (length ≤ <paramref name="maxNewTokens"/>).</returns>
    public static IReadOnlyList<int> Decode(
        Func<int?, Vector<T>> stepLogits,
        int maxNewTokens,
        SamplingOptions? options = null,
        Func<int, bool>? isEndToken = null,
        Func<int, bool>? suppressToken = null)
    {
        if (stepLogits is null) throw new ArgumentNullException(nameof(stepLogits));
        if (maxNewTokens < 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
        options ??= SamplingOptions.Default;

        // Resolve the RNG ONCE for the whole sequence — a Seed must give a reproducible *sequence*,
        // not the same token every step (which is what re-seeding per call would produce).
        Random? rng = options.IsGreedy
            ? null
            : options.Seed.HasValue
                ? Tensors.Helpers.RandomHelper.CreateSeededRandom(options.Seed.Value)
                : Tensors.Helpers.RandomHelper.ThreadSafeRandom;

        var tokens = new List<int>(maxNewTokens);
        int? prev = null;
        for (int step = 0; step < maxNewTokens; step++)
        {
            var logits = stepLogits(prev)
                ?? throw new InvalidOperationException("stepLogits returned null logits.");
            int next = options.IsGreedy
                ? TokenSampler<T>.ArgMax(logits)
                : TokenSampler<T>.Sample(logits, options, rng);

            if (isEndToken is not null && isEndToken(next)) break;
            // Suppressed tokens consume the step but are neither emitted nor fed back; the prefix is
            // unchanged so the next step re-samples from it.
            if (suppressToken is not null && suppressToken(next)) continue;
            tokens.Add(next);
            prev = next;
        }
        return tokens;
    }
}
