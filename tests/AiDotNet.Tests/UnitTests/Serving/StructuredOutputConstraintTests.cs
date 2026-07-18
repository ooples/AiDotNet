using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Serving.ContinuousBatching;
using AiDotNet.Serving.StructuredOutput;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serving;

/// <summary>
/// Tests for structured/guided decoding: the <see cref="ITokenConstraint"/> masking contract and its
/// integration into the continuous-batching sampler (constrained requests emit only permitted tokens).
/// </summary>
public class StructuredOutputConstraintTests
{
    private const int Eos = 2;
    private const int Vocab = 10;

    private static float[] FlatLogits(params int[] peaks)
    {
        // Uniform 0, with the given token ids boosted so an UNCONSTRAINED argmax would pick peaks[0].
        var l = new float[Vocab];
        foreach (int p in peaks) l[p] = 10f;
        return l;
    }

    [Fact]
    public void FromSequence_MasksAllButNextToken_AndAdvances()
    {
        var c = TokenFsmConstraint.FromSequence(new[] { 3, 7 }, Eos);

        // Step 0: only token 3 permitted (EOS not yet, not accepting).
        var logits = FlatLogits(5, 3, 7);
        c.ApplyMask(logits);
        Assert.Equal(float.NegativeInfinity, logits[5]);
        Assert.Equal(float.NegativeInfinity, logits[7]);
        Assert.Equal(float.NegativeInfinity, logits[Eos]);
        Assert.False(float.IsNegativeInfinity(logits[3]));
        Assert.False(c.IsComplete);

        c.Accept(3);

        // Step 1: only token 7 permitted.
        logits = FlatLogits(5, 3, 7);
        c.ApplyMask(logits);
        Assert.Equal(float.NegativeInfinity, logits[3]);
        Assert.False(float.IsNegativeInfinity(logits[7]));
        Assert.False(c.IsComplete);

        c.Accept(7);

        // Terminal accepting state: only EOS permitted, and the constraint reports complete.
        Assert.True(c.IsComplete);
        logits = FlatLogits(5, 3, 7);
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[Eos]));
        Assert.Equal(float.NegativeInfinity, logits[3]);
        Assert.Equal(float.NegativeInfinity, logits[7]);
    }

    [Fact]
    public void FromChoices_SharesPrefixStates_AndAcceptsEither()
    {
        // "yes" = [3,7], "yak" = [3,8]: after 3 both 7 and 8 remain viable.
        var c = TokenFsmConstraint.FromChoices(new IReadOnlyList<int>[] { new[] { 3, 7 }, new[] { 3, 8 } }, Eos);

        var logits = FlatLogits(5);
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[3]));
        Assert.Equal(float.NegativeInfinity, logits[7]);
        Assert.Equal(float.NegativeInfinity, logits[8]);

        c.Accept(3);
        logits = FlatLogits(5);
        c.ApplyMask(logits);
        Assert.False(float.IsNegativeInfinity(logits[7]));
        Assert.False(float.IsNegativeInfinity(logits[8]));
        Assert.Equal(float.NegativeInfinity, logits[3]);
        Assert.False(c.IsComplete);

        c.Accept(8);
        Assert.True(c.IsComplete); // [3,8] is a complete choice
    }

    [Fact(Timeout = 60000)]
    public async Task Batcher_StructuredConstraint_ForcesAllowedTokens_OverModelPreference()
    {
        await Task.Yield();

        // Model ALWAYS prefers token 5 (unconstrained argmax would emit 5,5,5,...).
        Tensor<float> model(Tensor<float> input)
        {
            int seq = input.Shape[^1];
            var logits = new Tensor<float>(new[] { 1, seq, Vocab });
            for (int p = 0; p < seq; p++)
                for (int i = 0; i < Vocab; i++)
                    logits[new[] { 0, p, i }] = i == 5 ? 10f : 0f;
            return logits;
        }

        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = true,
            EosTokenId = Eos,
            // Speculation ON in config: a constrained request must still disable it (else the greedy
            // draft/verify path would bypass the mask and emit the forbidden token 5).
            EnableSpeculativeDecoding = true,
            SpeculationDepth = 3
        }, model);

        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1, 4, 6 },
            MaxNewTokens = 5,
            Temperature = 0f, // greedy: the mask alone decides among permitted tokens
            EosTokenId = Eos,
            Constraint = TokenFsmConstraint.FromSequence(new[] { 3, 7 }, Eos)
        };

        var result = await batcher.GenerateAsync(request);

        Assert.DoesNotContain(5, result.GeneratedTokens); // the model's preferred token is never emitted
        Assert.Equal(3, result.GeneratedTokens[0]);
        Assert.Equal(7, result.GeneratedTokens[1]);
    }

    [Fact(Timeout = 60000)]
    public async Task Batcher_LogitBias_BansAndForcesTokens()
    {
        await Task.Yield();

        // Model prefers token 5. A large negative bias bans 5; a large positive bias on 8 makes it win.
        Tensor<float> model(Tensor<float> input)
        {
            int seq = input.Shape[^1];
            var logits = new Tensor<float>(new[] { 1, seq, Vocab });
            for (int p = 0; p < seq; p++)
                for (int i = 0; i < Vocab; i++)
                    logits[new[] { 0, p, i }] = i == 5 ? 10f : 0f;
            return logits;
        }

        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = true,
            EosTokenId = Eos
        }, model);

        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1, 4, 6 },
            MaxNewTokens = 3,
            Temperature = 0f,
            EosTokenId = Eos,
            LogitBias = new Dictionary<int, float> { [5] = -100f, [8] = 50f }
        };

        var result = await batcher.GenerateAsync(request);

        Assert.DoesNotContain(5, result.GeneratedTokens); // banned
        Assert.All(result.GeneratedTokens, t => Assert.Equal(8, t)); // forced winner
    }
}
