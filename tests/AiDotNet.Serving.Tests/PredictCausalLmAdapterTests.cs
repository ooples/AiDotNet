using System;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for <see cref="PredictCausalLmAdapter{T}"/> — the widest-coverage fallback that presents any
/// token-in / vocab-width-logits-out forward function as an ICausalLmModel. Verified end-to-end through the
/// engine so the adapter's forward + the recompute runner + the scheduler all agree.
/// </summary>
public class PredictCausalLmAdapterTests
{
    private const int Vocab = 64;

    // A forward function that returns [1, seq, vocab] one-hot at (last+1) mod vocab for each position.
    private static Tensor<double> CounterForward(Tensor<double> tokenIds)
    {
        int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
        var t = new Tensor<double>(new[] { 1, n, Vocab });
        for (int p = 0; p < n; p++)
        {
            int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, p]));
            t[0, p, (last + 1) % Vocab] = 1.0;
        }
        return t;
    }

    [Fact]
    public void Properties_AreExposed()
    {
        var adapter = new PredictCausalLmAdapter<double>(CounterForward, Vocab, eosTokenId: 7);
        Assert.Equal(Vocab, adapter.VocabularySize);
        Assert.Equal(7, adapter.EosTokenId);
    }

    [Fact]
    public void NullForward_Throws()
        => Assert.Throws<ArgumentNullException>(() => new PredictCausalLmAdapter<double>(null!, Vocab));

    [Fact]
    public void ZeroVocab_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() => new PredictCausalLmAdapter<double>(CounterForward, 0));

    [Fact]
    public void EndToEnd_ThroughEngine_ProducesDeterministicOutput()
    {
        var adapter = new PredictCausalLmAdapter<double>(CounterForward, Vocab);
        using var engine = new ContinuousBatchingEngine<double>(new RecomputeModelRunner<double>(adapter));
        engine.AddRequest(new GenerationRequest("r1", new[] { 3 },
            new SamplingParameters { Temperature = 0.0, MaxTokens = 4 }));

        RequestOutput? final = null;
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 1000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.IsFinished) final = o;
        }

        var completion = Assert.Single(final!.Outputs);
        Assert.Equal(new[] { 4, 5, 6, 7 }, completion.TokenIds);
    }

    [Fact]
    public void Factory_WithoutCapabilityOrVocab_ThrowsHelpfulError()
    {
        var ex = Assert.Throws<NotSupportedException>(() => ServingRunnerFactory.Create<double>("plain string"));
        Assert.Contains("Predict-based adapter", ex.Message);
    }
}
