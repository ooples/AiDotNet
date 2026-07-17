using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// End-to-end scheduling tests for <see cref="ContinuousBatchingEngine{T}"/> driven by a deterministic fake
/// model. Because the fake is a stateless counter (next token = last + 1, mod vocab), recompute-preemption
/// produces byte-identical output — so these assert exact generated sequences even under KV-memory pressure,
/// proving the scheduler, admission, preemption, stop conditions, batching, and abort paths are correct.
/// </summary>
public class ContinuousBatchingEngineTests
{
    private const int Vocab = 100;

    /// <summary>
    /// Deterministic stateless "model": the next token is always (last token + 1) mod vocab. Returned as a
    /// one-hot logits row so greedy decoding selects it. Stateless ⇒ recompute yields identical results.
    /// </summary>
    private sealed class CounterRunner : IServingModelRunner<double>
    {
        public int VocabularySize => Vocab;
        public int ExecuteCalls { get; private set; }
        public int MaxBatchSeen { get; private set; }

        public IReadOnlyList<Vector<double>> Execute(IReadOnlyList<SequenceExecution<double>> batch)
        {
            ExecuteCalls++;
            MaxBatchSeen = Math.Max(MaxBatchSeen, batch.Count);
            var result = new List<Vector<double>>(batch.Count);
            foreach (var exec in batch)
            {
                int last = exec.AllTokenIds[exec.AllTokenIds.Count - 1];
                int next = (last + 1) % Vocab;
                var row = new double[Vocab];
                row[next] = 1.0;
                result.Add(new Vector<double>(row));
            }
            return result;
        }
    }

    private static SamplingParameters Greedy(int maxTokens, int minTokens = 0, IReadOnlyList<int>? stops = null)
        => new() { Temperature = 0.0, MaxTokens = maxTokens, MinTokens = minTokens, StopTokenIds = stops };

    private static GenerationRequest Req(string id, int[] prompt, SamplingParameters p)
        => new(id, prompt, p);

    // Expected counter output: maxTokens tokens starting at (lastPrompt+1).
    private static int[] ExpectedCounter(int lastPrompt, int count)
        => Enumerable.Range(1, count).Select(k => (lastPrompt + k) % Vocab).ToArray();

    private static Dictionary<string, RequestOutput> RunToCompletion(
        ContinuousBatchingEngine<double> engine, int maxSteps = 10000)
    {
        var final = new Dictionary<string, RequestOutput>();
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > maxSteps) throw new InvalidOperationException("Engine did not converge.");
            foreach (var output in engine.Step())
                if (output.IsFinished) final[output.RequestId] = output;
        }
        return final;
    }

    // ---- Single request ------------------------------------------------------------------

    [Fact]
    public void SingleRequest_Greedy_GeneratesDeterministicSequence_LengthCapped()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner());
        engine.AddRequest(Req("r1", new[] { 5 }, Greedy(maxTokens: 4)));

        var final = RunToCompletion(engine);

        var output = final["r1"];
        var completion = Assert.Single(output.Outputs);
        Assert.Equal(ExpectedCounter(5, 4), completion.TokenIds); // 6,7,8,9
        Assert.True(completion.IsFinished);
        Assert.Equal("length", completion.FinishReason);
    }

    [Fact]
    public void EosToken_StopsGeneration_BeforeMaxTokens()
    {
        var runner = new CounterRunner();
        using var engine = new ContinuousBatchingEngine<double>(runner, new EngineOptions { EosTokenId = 0 });
        // prompt 98 -> 99, then 0 (== EOS) stops.
        engine.AddRequest(Req("r1", new[] { 98 }, Greedy(maxTokens: 20)));

        var output = RunToCompletion(engine)["r1"];
        var completion = Assert.Single(output.Outputs);
        Assert.Equal(new[] { 99, 0 }, completion.TokenIds);
        Assert.Equal("stop", completion.FinishReason);
    }

    [Fact]
    public void MinTokens_SuppressesEarlyEos()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner(), new EngineOptions { EosTokenId = 0 });
        // EOS (0) would appear at position 2 (99,0,...) but MinTokens=5 suppresses it; hits length cap at 10.
        engine.AddRequest(Req("r1", new[] { 98 }, Greedy(maxTokens: 10, minTokens: 5)));

        var completion = Assert.Single(RunToCompletion(engine)["r1"].Outputs);
        Assert.Equal(10, completion.TokenIds.Count);
        Assert.Equal("length", completion.FinishReason);
        Assert.Contains(0, completion.TokenIds); // the EOS token was emitted, not treated as a stop
    }

    [Fact]
    public void CustomStopToken_EndsSequence()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner());
        // prompt 40 -> 41,42,43 ; stop token 43.
        engine.AddRequest(Req("r1", new[] { 40 }, Greedy(maxTokens: 50, stops: new[] { 43 })));

        var completion = Assert.Single(RunToCompletion(engine)["r1"].Outputs);
        Assert.Equal(new[] { 41, 42, 43 }, completion.TokenIds);
        Assert.Equal("stop", completion.FinishReason);
    }

    // ---- Continuous batching -------------------------------------------------------------

    [Fact]
    public void ManyRequests_AllFinish_WithCorrectOutputs_AndBatchTogether()
    {
        var runner = new CounterRunner();
        using var engine = new ContinuousBatchingEngine<double>(runner);

        for (int i = 0; i < 16; i++)
            engine.AddRequest(Req($"r{i}", new[] { i }, Greedy(maxTokens: 6)));

        var final = RunToCompletion(engine);

        Assert.Equal(16, final.Count);
        for (int i = 0; i < 16; i++)
        {
            var completion = Assert.Single(final[$"r{i}"].Outputs);
            Assert.Equal(ExpectedCounter(i, 6), completion.TokenIds);
        }
        Assert.True(runner.MaxBatchSeen > 1, "requests should have been batched together in at least one step");
    }

    // ---- Preemption under memory pressure ------------------------------------------------

    [Fact]
    public void TinyKvPool_ForcesPreemption_ButOutputsStayCorrect()
    {
        var runner = new CounterRunner();
        // 3 blocks x 4 slots = 12 KV slots total; several concurrent sequences each want up to 2+8 slots.
        var options = new EngineOptions { BlockSize = 4, NumKvBlocks = 3, MaxNumSequences = 8 };
        using var engine = new ContinuousBatchingEngine<double>(runner, options);

        for (int i = 0; i < 6; i++)
            engine.AddRequest(Req($"r{i}", new[] { 10 + i, 20 + i }, Greedy(maxTokens: 8)));

        var final = RunToCompletion(engine);

        Assert.Equal(6, final.Count);
        for (int i = 0; i < 6; i++)
        {
            var completion = Assert.Single(final[$"r{i}"].Outputs);
            Assert.Equal(ExpectedCounter(20 + i, 8), completion.TokenIds); // last prompt token is 20+i
            Assert.True(completion.IsFinished);
        }
        Assert.True(engine.GetStatistics().TotalPreemptions > 0, "tiny pool should have forced at least one preemption");
    }

    [Fact]
    public void PromptLargerThanPool_IsAbortedGracefully()
    {
        using var engine = new ContinuousBatchingEngine<double>(
            new CounterRunner(), new EngineOptions { BlockSize = 4, NumKvBlocks = 2 }); // 8 slots max
        engine.AddRequest(Req("big", Enumerable.Range(1, 20).ToArray(), Greedy(maxTokens: 4))); // 20 > 8

        var completion = Assert.Single(RunToCompletion(engine)["big"].Outputs);
        Assert.True(completion.IsFinished);
        Assert.Equal("abort", NormalizeAbort(completion.FinishReason));
    }

    private static string NormalizeAbort(string? reason)
        => reason is "prompt_too_long" or "abort" ? "abort" : reason ?? "";

    // ---- Parallel sampling (N > 1) -------------------------------------------------------

    [Fact]
    public void ParallelSampling_ProducesNCompletions()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner());
        engine.AddRequest(Req("r1", new[] { 7 }, new SamplingParameters { Temperature = 0.0, MaxTokens = 3, N = 4 }));

        var output = RunToCompletion(engine)["r1"];
        Assert.Equal(4, output.Outputs.Count);
        Assert.Equal(new[] { 0, 1, 2, 3 }, output.Outputs.Select(o => o.SequenceIndex).OrderBy(x => x).ToArray());
        foreach (var completion in output.Outputs)
            Assert.Equal(ExpectedCounter(7, 3), completion.TokenIds); // greedy ⇒ all identical
    }

    // ---- Abort ---------------------------------------------------------------------------

    [Fact]
    public void AbortRequest_FinishesAborted_AndFreesMemory()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner());
        engine.AddRequest(Req("r1", new[] { 3 }, Greedy(maxTokens: 1000)));

        engine.Step(); // admit + first token
        Assert.True(engine.AbortRequest("r1"));

        var final = RunToCompletion(engine);
        Assert.False(engine.HasUnfinishedRequests);
        Assert.Equal(0, engine.GetStatistics().RunningSequences);
        Assert.Equal(0.0, engine.GetStatistics().KvCacheUsage); // blocks reclaimed
        if (final.TryGetValue("r1", out var output))
            Assert.Equal("abort", output.Outputs[0].FinishReason);
    }

    [Fact]
    public void AbortUnknownRequest_ReturnsFalse()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner());
        Assert.False(engine.AbortRequest("nope"));
    }

    // ---- Statistics ----------------------------------------------------------------------

    [Fact]
    public void Statistics_ReflectFinishedCount()
    {
        using var engine = new ContinuousBatchingEngine<double>(new CounterRunner());
        for (int i = 0; i < 5; i++)
            engine.AddRequest(Req($"r{i}", new[] { i }, Greedy(maxTokens: 2)));

        RunToCompletion(engine);

        var stats = engine.GetStatistics();
        Assert.Equal(5, stats.TotalFinishedRequests);
        Assert.Equal(0, stats.RunningSequences);
        Assert.Equal(0, stats.WaitingSequences);
    }
}
