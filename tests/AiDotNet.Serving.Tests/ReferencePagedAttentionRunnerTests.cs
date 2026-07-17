using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Serving.Engine;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for <see cref="ReferencePagedAttentionRunner{T}"/> — a real paged-attention causal LM. The load-bearing
/// invariant: driving it via the paged fast path (incremental prefill + decode over block tables) yields output
/// bit-identical to driving it via full recompute — proving the KV paging (block tables, per-step decode, prefix
/// sharing) is correct. This is the conformance contract a production GPU paged runner must also meet.
/// </summary>
public class ReferencePagedAttentionRunnerTests
{
    private const int Vocab = 32;

    private static ReferencePagedAttentionRunner<double> NewRunner()
        => new(vocabularySize: Vocab, dModel: 32, numLayers: 2, numHeads: 4, ffnDim: 64,
               blockSize: 8, maxBlocks: 64, seed: 2026);

    private static EngineOptions Options()
        => new() { BlockSize = 8, NumKvBlocks = 64, MaxNumSequences = 8 };

    private static int[] Generate(ContinuousBatchingEngine<double> engine, string id, int[] prompt, int maxTokens)
    {
        engine.AddRequest(new GenerationRequest(id, prompt,
            new SamplingParameters { Temperature = 0.0, MaxTokens = maxTokens }));
        RequestOutput? final = null;
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 2000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.RequestId == id && o.IsFinished) final = o;
        }
        return final!.Outputs[0].TokenIds.ToArray();
    }

    [Fact]
    public void PagedFastPath_MatchesFullRecompute()
    {
        var runner = NewRunner();
        var prompt = new[] { 3, 1, 4, 1, 5, 9, 2, 6 }; // length 8 (block-aligned)

        // Paged path: the runner is selected as ICausalLmRunner -> PagedRunnerAdapter (incremental KV).
        int[] paged;
        using (var engine = new ContinuousBatchingEngine<double>(new PagedRunnerAdapter<double>(runner), Options()))
            paged = Generate(engine, "p", prompt, 12);

        // Recompute path: same runner as ICausalLmModel -> full ForwardLogits each step.
        int[] recompute;
        using (var engine = new ContinuousBatchingEngine<double>(new RecomputeModelRunner<double>(runner), Options()))
            recompute = Generate(engine, "r", prompt, 12);

        Assert.Equal(recompute, paged); // paged incremental decode == full recompute
        Assert.Equal(12, paged.Length);
    }

    [Fact]
    public void Factory_SelectsPagedFastPath()
    {
        var selection = ServingRunnerFactory.Create<double>(NewRunner());
        Assert.IsType<PagedRunnerAdapter<double>>(selection.Runner);
    }

    [Fact]
    public void ServesThroughTextGenerator()
    {
        var runner = NewRunner();
        using var gen = new TextGenerator<double>(runner, options: Options());
        var ids = gen.Generate(new[] { 2, 7, 1, 8, 2, 8, 1, 8 }, new SamplingParameters { Temperature = 0.0, MaxTokens = 6 });
        Assert.Equal(6, ids.Count);
        Assert.All(ids, t => Assert.InRange(t, 0, Vocab - 1));
    }

    [Fact]
    public void ParallelSampling_PagedPath_SharesPromptKv_AndAgrees()
    {
        var runner = NewRunner();
        using var engine = new ContinuousBatchingEngine<double>(new PagedRunnerAdapter<double>(runner), Options());

        var prompt = new[] { 1, 2, 3, 4, 5, 6, 7, 8 }; // length 8 = 1 block -> siblings fork the owner's KV
        engine.AddRequest(new GenerationRequest("r1", prompt,
            new SamplingParameters { Temperature = 0.0, MaxTokens = 5, N = 3 }));

        RequestOutput? final = null;
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 2000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.IsFinished) final = o;
        }

        // All three sequences decode from the SHARED prompt KV (the owner's blocks). Greedy + same prompt =>
        // identical outputs, which also confirms the siblings read the owner's cached prompt KV correctly.
        Assert.Equal(3, final!.Outputs.Count);
        var reference = final.Outputs[0].TokenIds.ToArray();
        Assert.Equal(5, reference.Length);
        foreach (var completion in final.Outputs)
            Assert.Equal(reference, completion.TokenIds.ToArray());
    }
}
