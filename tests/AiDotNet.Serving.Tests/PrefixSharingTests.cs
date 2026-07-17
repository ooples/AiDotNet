using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for prefix sharing: the <see cref="BlockManager.ForkPrefix"/> primitive and its use by the engine to
/// let N&gt;1 parallel samples share one copy of the prompt's KV (prefill-once). Correctness must be unchanged;
/// memory usage must drop when the prompt is shared.
/// </summary>
public class PrefixSharingTests
{
    private const int Vocab = 100;

    private sealed class CounterRunner : IServingModelRunner<double>
    {
        public int VocabularySize => Vocab;
        public int ExecuteCalls { get; private set; }
        public IReadOnlyList<Vector<double>> Execute(IReadOnlyList<SequenceExecution<double>> batch)
        {
            ExecuteCalls++;
            var result = new List<Vector<double>>(batch.Count);
            foreach (var exec in batch)
            {
                int last = exec.AllTokenIds[exec.AllTokenIds.Count - 1];
                var row = new double[Vocab];
                row[(last + 1) % Vocab] = 1.0;
                result.Add(new Vector<double>(row));
            }
            return result;
        }
    }

    // ---- BlockManager.ForkPrefix -----------------------------------------------------

    [Fact]
    public void ForkPrefix_SharesOnlyPrefixBlocks()
    {
        var bm = new BlockManager(totalBlocks: 16, blockSize: 4);
        bm.Allocate("p", 8);            // 2 blocks
        bm.Append("p", 4);             // grows to 12 tokens -> 3 blocks
        int usedBefore = bm.NumUsedBlocks;

        var child = bm.ForkPrefix("p", "c", prefixTokens: 8).ToArray(); // share first 2 blocks only
        var parent = bm.GetBlockTable("p");

        Assert.Equal(2, child.Length);
        Assert.Equal(parent[0], child[0]);
        Assert.Equal(parent[1], child[1]);
        Assert.Equal(usedBefore, bm.NumUsedBlocks); // sharing allocates nothing
        Assert.Equal(8, bm.GetLength("c"));
    }

    [Fact]
    public void ForkPrefix_SharedBlocks_ReclaimedOnlyWhenAllRelease()
    {
        var bm = new BlockManager(16, 4);
        bm.Allocate("p", 8); // 2 blocks
        bm.ForkPrefix("p", "c", 8);
        Assert.Equal(2, bm.NumUsedBlocks);

        bm.Free("c");
        Assert.Equal(2, bm.NumUsedBlocks); // p still holds them
        bm.Free("p");
        Assert.Equal(0, bm.NumUsedBlocks);
        Assert.Equal(16, bm.NumFreeBlocks);
    }

    [Fact]
    public void ForkPrefix_PrefixBeyondParentLength_Throws()
    {
        var bm = new BlockManager(16, 4);
        bm.Allocate("p", 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => bm.ForkPrefix("p", "c", 8));
    }

    // ---- Engine prefix sharing (N > 1) -----------------------------------------------

    private static Dictionary<string, RequestOutput> RunToCompletion(ContinuousBatchingEngine<double> engine)
    {
        var final = new Dictionary<string, RequestOutput>();
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 10000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.IsFinished) final[o.RequestId] = o;
        }
        return final;
    }

    [Fact]
    public void ParallelSampling_AlignedPrompt_SharesPromptBlocks_AndStaysCorrect()
    {
        var runner = new CounterRunner();
        // BlockSize 4, prompt length 4 (block-aligned) -> siblings fork the single prompt block.
        var options = new EngineOptions { BlockSize = 4, NumKvBlocks = 32 };
        using var engine = new ContinuousBatchingEngine<double>(runner, options);

        engine.AddRequest(new GenerationRequest("r1", new[] { 1, 2, 3, 4 },
            new SamplingParameters { Temperature = 0.0, MaxTokens = 3, N = 3 }));

        // After the first step the owner has prefilled and the siblings are forked; peak block usage should be
        // far below 3 independent prompt copies. Track the max usage across the run.
        int maxUsed = 0;
        var final = new Dictionary<string, RequestOutput>();
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 10000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.IsFinished) final[o.RequestId] = o;
            maxUsed = Math.Max(maxUsed, engine.GetStatistics().TotalKvBlocks - engine.GetStatistics().FreeKvBlocks);
        }

        var output = final["r1"];
        Assert.Equal(3, output.Outputs.Count);
        foreach (var completion in output.Outputs)
            Assert.Equal(new[] { 5, 6, 7 }, completion.TokenIds); // greedy from prompt ...4 -> 5,6,7

        // 3 independent sequences would each need 1 prompt block + up to 1 growth block = up to 6 blocks;
        // with a shared prompt block the prompt costs 1 (not 3), so peak usage stays well under 6.
        Assert.True(maxUsed <= 4, $"expected shared-prompt usage <= 4 blocks, saw {maxUsed}");
    }

    [Fact]
    public void ParallelSampling_NonAlignedPrompt_FallsBack_ButStaysCorrect()
    {
        var runner = new CounterRunner();
        var options = new EngineOptions { BlockSize = 4, NumKvBlocks = 64 };
        using var engine = new ContinuousBatchingEngine<double>(runner, options);

        // prompt length 3 (not a multiple of 4) -> fallback to independent prefill for each sibling.
        engine.AddRequest(new GenerationRequest("r1", new[] { 1, 2, 3 },
            new SamplingParameters { Temperature = 0.0, MaxTokens = 3, N = 3 }));

        var output = RunToCompletion(engine)["r1"];
        Assert.Equal(3, output.Outputs.Count);
        foreach (var completion in output.Outputs)
            Assert.Equal(new[] { 4, 5, 6 }, completion.TokenIds);
    }

    [Fact]
    public void ParallelSampling_NonAlignedPrompt_PagedPath_SharesAlignedPrefix_AndAgrees()
    {
        // Prompt length 10 with block size 8 -> aligned prefix of 8 tokens is shared; only the 2-token tail is
        // recomputed per sibling. Uses the paged runner so we can observe the compute savings.
        var runner = new ReferencePagedAttentionRunner<double>(
            vocabularySize: 32, dModel: 32, numLayers: 2, numHeads: 4, ffnDim: 64, blockSize: 8, maxBlocks: 128, seed: 7);
        using var engine = new ContinuousBatchingEngine<double>(new PagedRunnerAdapter<double>(runner),
            new EngineOptions { BlockSize = 8, NumKvBlocks = 128, MaxNumSequences = 8 });

        var prompt = Enumerable.Range(1, 10).ToArray(); // 10 tokens (not a multiple of 8)
        engine.AddRequest(new GenerationRequest("r1", prompt,
            new SamplingParameters { Temperature = 0.0, MaxTokens = 4, N = 3 }));

        RequestOutput? final = null;
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 4000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.IsFinished) final = o;
        }

        Assert.Equal(3, final!.Outputs.Count);
        var reference = final.Outputs[0].TokenIds.ToArray();
        foreach (var completion in final.Outputs)
            Assert.Equal(reference, completion.TokenIds.ToArray());

        // Owner prefills all 10 prompt positions; each of the 2 siblings recomputes only the 2-token tail (not
        // the full 10). So total prefill positions is far below 3 * 10 = 30 independent prefills.
        // owner(10) + 2 siblings * 2 tail + decode steps (3 seqs * up to 4) << 30 + 12.
        Assert.True(runner.PositionsComputed < 30, $"expected shared aligned prefix; computed {runner.PositionsComputed} positions");
    }

    [Fact]
    public void ParallelSampling_AbortBeforePrefill_CleansUp()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner(),
            new EngineOptions { BlockSize = 4, NumKvBlocks = 32 });
        engine.AddRequest(new GenerationRequest("r1", new[] { 1, 2, 3, 4 },
            new SamplingParameters { Temperature = 0.0, MaxTokens = 100, N = 3 }));

        Assert.True(engine.AbortRequest("r1")); // abort before any step
        RunToCompletion(engine);
        Assert.False(engine.HasUnfinishedRequests);
        Assert.Equal(0, engine.GetStatistics().RunningSequences);
        Assert.Equal(0.0, engine.GetStatistics().KvCacheUsage);
    }
}
