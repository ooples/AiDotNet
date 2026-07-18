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

    [Fact(Timeout = 60000)]
    public async Task Batcher_FrequencyPenalty_BreaksRepetitionLoops()
    {
        await Task.Yield();

        // Model narrowly prefers token 5 over 6 (10 vs 9). Greedy would emit 5 forever. A frequency penalty
        // pushes 5's logit down as it recurs, so after emitting it once the model switches to other tokens
        // instead of looping — the whole point of frequency_penalty.
        Tensor<float> model(Tensor<float> input)
        {
            int seq = input.Shape[^1];
            var logits = new Tensor<float>(new[] { 1, seq, Vocab });
            for (int p = 0; p < seq; p++)
                for (int i = 0; i < Vocab; i++)
                    logits[new[] { 0, p, i }] = i == 5 ? 10f : (i == 6 ? 9f : 0f);
            return logits;
        }

        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = true,
            EosTokenId = Eos
        }, model);

        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 6,
            Temperature = 0f,
            EosTokenId = Eos,
            FrequencyPenalty = 5f // each prior occurrence subtracts 5 from the token's logit
        };

        var result = await batcher.GenerateAsync(request);

        // Without the penalty this would be [5,5,5,5,5,5]; the penalty forces variety.
        Assert.True(result.GeneratedTokens.Count > 1);
        Assert.True(result.GeneratedTokens.Distinct().Count() > 1,
            $"frequency penalty should break the repetition loop; got [{string.Join(",", result.GeneratedTokens)}]");
    }

    [Fact(Timeout = 60000)]
    public async Task Batcher_LogProbs_RecordsChosenTokenAndTopK()
    {
        await Task.Yield();

        // Ranked logits: token 5 > 6 > 7 > rest. The softmax log-probs must reflect that ordering.
        Tensor<float> model(Tensor<float> input)
        {
            int seq = input.Shape[^1];
            var logits = new Tensor<float>(new[] { 1, seq, Vocab });
            for (int p = 0; p < seq; p++)
            {
                logits[new[] { 0, p, 5 }] = 10f;
                logits[new[] { 0, p, 6 }] = 9f;
                logits[new[] { 0, p, 7 }] = 8f;
            }
            return logits;
        }

        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = true,
            EosTokenId = Eos
        }, model);

        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 2,
            Temperature = 0f,        // greedy: chosen = argmax = 5
            EosTokenId = Eos,
            IncludeLogProbs = true,
            TopLogProbs = 3
        };

        var result = await batcher.GenerateAsync(request);

        Assert.NotNull(result.LogProbs);
        Assert.Equal(result.GeneratedTokens.Count, result.LogProbs!.Count);

        var first = result.LogProbs[0];
        Assert.Equal(5, first.TokenId);                 // greedy picked the top-logit token
        Assert.True(first.LogProb < 0f);                // log of a probability < 1
        Assert.Equal(-0.41f, first.LogProb, 1);         // 10 - logsumexp(10,9,8,0*29) ≈ -0.41

        Assert.Equal(3, first.TopLogProbs.Count);
        Assert.Equal(new[] { 5, 6, 7 }, first.TopLogProbs.Select(t => t.TokenId).ToArray()); // descending
        Assert.Equal(first.LogProb, first.TopLogProbs[0].LogProb); // chosen == top-1
        Assert.True(first.TopLogProbs[0].LogProb > first.TopLogProbs[1].LogProb);
    }
}
