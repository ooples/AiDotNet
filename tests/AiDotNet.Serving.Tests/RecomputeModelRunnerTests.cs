using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for <see cref="RecomputeModelRunner{T}"/> — the adapter that drives any <see cref="ICausalLmModel{T}"/>
/// in the engine by recomputing logits from the full token sequence. Covers last-position extraction across the
/// output ranks a model may return ([vocab], [seq,vocab], [1,seq,vocab]) and an end-to-end run through the
/// continuous-batching engine.
/// </summary>
public class RecomputeModelRunnerTests
{
    private const int Vocab = 100;

    /// <summary>Deterministic counter LM: next token = (last token + 1) mod vocab, emitted as one-hot logits
    /// at the requested output rank so the adapter's shape handling is exercised.</summary>
    private sealed class CounterLm : ICausalLmModel<double>
    {
        private readonly int _rank;
        public CounterLm(int rank) => _rank = rank;

        public int VocabularySize => Vocab;
        public int? EosTokenId => null;

        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, n - 1]));

            switch (_rank)
            {
                case 1:
                {
                    var t = new Tensor<double>(new[] { Vocab });
                    t[(last + 1) % Vocab] = 1.0;
                    return t;
                }
                case 2:
                {
                    var t = new Tensor<double>(new[] { n, Vocab });
                    t[n - 1, (last + 1) % Vocab] = 1.0; // last position one-hot
                    return t;
                }
                default:
                {
                    var t = new Tensor<double>(new[] { 1, n, Vocab });
                    t[0, n - 1, (last + 1) % Vocab] = 1.0;
                    return t;
                }
            }
        }
    }

    private static SequenceExecution<double> Exec(params int[] tokens)
        => new("s", tokens, tokens.Length, Array.Empty<int>(), Array.Empty<BlockCopy>(), true);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Execute_ExtractsLastPositionLogits_ForEachRank(int rank)
    {
        var runner = new RecomputeModelRunner<double>(new CounterLm(rank));
        Assert.Equal(Vocab, runner.VocabularySize);

        var logits = Assert.Single(runner.Execute(new[] { Exec(5, 6, 41) }));
        // Argmax should be (41 + 1) = 42.
        int argmax = Enumerable.Range(0, Vocab).Aggregate((a, b) => logits[b] > logits[a] ? b : a);
        Assert.Equal(42, argmax);
    }

    [Fact]
    public void NullModel_Throws()
        => Assert.Throws<ArgumentNullException>(() => new RecomputeModelRunner<double>(null!));

    [Fact]
    public void EndToEnd_ThroughEngine_ProducesDeterministicCounterOutput()
    {
        using var engine = new ContinuousBatchingEngine<double>(new RecomputeModelRunner<double>(new CounterLm(3)));
        engine.AddRequest(new GenerationRequest("r1", new[] { 5 },
            new SamplingParameters { Temperature = 0.0, MaxTokens = 4 }));

        RequestOutput? final = null;
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 1000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.IsFinished) final = o;
        }

        var completion = Assert.Single(final!.Outputs);
        Assert.Equal(new[] { 6, 7, 8, 9 }, completion.TokenIds);
        Assert.Equal("length", completion.FinishReason);
    }
}
