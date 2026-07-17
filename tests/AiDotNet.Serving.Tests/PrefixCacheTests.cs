using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Serving.Engine;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for automatic cross-request prefix caching: a new request sharing a block-aligned prompt prefix with an
/// earlier one reuses the cached KV. Correctness must be unchanged (caching only affects speed), and on the
/// paged path the shared prefix must actually be skipped (fewer positions recomputed).
/// </summary>
public class PrefixCacheTests
{
    private const int Vocab = 32;

    private static ReferencePagedAttentionRunner<double> NewRunner()
        => new(vocabularySize: Vocab, dModel: 32, numLayers: 2, numHeads: 4, ffnDim: 64,
               blockSize: 8, maxBlocks: 128, seed: 99);

    private static int[] Generate(ContinuousBatchingEngine<double> engine, string id, int[] prompt, int maxTokens)
    {
        engine.AddRequest(new GenerationRequest(id, prompt,
            new SamplingParameters { Temperature = 0.0, MaxTokens = maxTokens }));
        RequestOutput? final = null;
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 4000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.RequestId == id && o.IsFinished) final = o;
        }
        return final!.Outputs[0].TokenIds.ToArray();
    }

    // ---- PrefixCache unit behavior ----

    [Fact]
    public void Cache_LookupAfterRegister_ReturnsLongestBlockAlignedPrefix()
    {
        var bm = new BlockManager(totalBlocks: 64, blockSize: 8);
        var cache = new PrefixCache(bm, blockSize: 8, capacity: 16);

        var prompt = Enumerable.Range(1, 24).ToArray(); // 24 tokens = 3 blocks
        bm.Allocate("owner", 24);
        cache.Register(prompt, "owner");

        // A new prompt sharing the first 16 tokens (2 blocks) -> hit at length 16.
        var other = prompt.Take(16).Concat(new[] { 99, 98 }).ToArray();
        var hit = cache.Lookup(other);
        Assert.NotNull(hit);
        Assert.Equal(16, hit!.Value.PrefixLength);

        // No shared prefix -> miss.
        Assert.Null(cache.Lookup(new[] { 500, 501, 502, 503, 504, 505, 506, 507 }));
    }

    [Fact]
    public void Cache_Evicts_Lru_BeyondCapacity()
    {
        var bm = new BlockManager(64, 8);
        var cache = new PrefixCache(bm, 8, capacity: 1);

        bm.Allocate("a", 8); cache.Register(Enumerable.Range(0, 8).ToArray(), "a");
        bm.Allocate("b", 8); cache.Register(Enumerable.Range(100, 8).ToArray(), "b");

        Assert.Equal(1, cache.Count); // capacity 1 -> only the newest remains
        Assert.Null(cache.Lookup(Enumerable.Range(0, 8).ToArray()));       // evicted
        Assert.NotNull(cache.Lookup(Enumerable.Range(100, 8).ToArray()));  // retained
    }

    // ---- Engine correctness: caching must not change output ----

    [Fact]
    public void SharedPrefix_SecondRequest_SameOutputWithOrWithoutCache()
    {
        var prompt1 = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };  // 16-token shared prefix below
        var prompt2 = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21 };

        int[] noCache;
        using (var e = new ContinuousBatchingEngine<double>(new PagedRunnerAdapter<double>(NewRunner()),
            new EngineOptions { BlockSize = 8, NumKvBlocks = 128 }))
        {
            Generate(e, "warm", prompt1, 3);
            noCache = Generate(e, "q", prompt2, 6);
        }

        int[] cached;
        using (var e = new ContinuousBatchingEngine<double>(new PagedRunnerAdapter<double>(NewRunner()),
            new EngineOptions { BlockSize = 8, NumKvBlocks = 128, EnablePrefixCache = true }))
        {
            Generate(e, "warm", prompt2, 3); // registers the 16-token prefix of prompt2
            cached = Generate(e, "q", prompt2, 6);
        }

        Assert.Equal(noCache, cached); // caching never changes the result
    }

    // ---- Engine efficiency: cached prefix is not recomputed on the paged path ----

    [Fact]
    public void SharedPrefix_PagedPath_SkipsCachedPrefixCompute()
    {
        var runner = NewRunner();
        using var engine = new ContinuousBatchingEngine<double>(new PagedRunnerAdapter<double>(runner),
            new EngineOptions { BlockSize = 8, NumKvBlocks = 128, EnablePrefixCache = true });

        // First request: prompt of 16 tokens (2 blocks) -> registers the 16-token prefix.
        var first = Enumerable.Range(1, 16).ToArray();
        Generate(engine, "a", first, 2);
        long afterFirst = runner.PositionsComputed;

        // Second request shares the full 16-token prefix, then 2 new prompt tokens (within vocab).
        var second = first.Concat(new[] { 17, 18 }).ToArray();
        long before = runner.PositionsComputed;
        Generate(engine, "b", second, 2);
        long secondCost = runner.PositionsComputed - before;

        // Without caching the second prefill would recompute all 18 prompt positions; with the cached 16-token
        // prefix it computes only the 2 new prompt positions (+2 decode steps) — far fewer than 18.
        Assert.True(secondCost < 18, $"expected cached prefix to be skipped; second request computed {secondCost} positions");
    }
}
