using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Temperature 0 means greedy decoding. Speculative decoding only changes how many target forwards a
/// generation takes, so at temperature 0 it must emit exactly the tokens a plain argmax loop does.
/// Previously temperature 0 was applied as p^(1/0): every probability below 1 became 0, nothing could be
/// sampled, and the decoders emitted the last vocabulary id on every step.
/// </summary>
public class SpeculativeDecodingGreedyTests
{
    private const int Vocab = 32;

    // A peaked but not one-hot causal target: the next token after position i is (7 * tokens[i] + i + 3) % Vocab
    // with probability 0.9 (the rest spread uniformly), so no probability is exactly 1.
    private static Matrix<float> TargetProbs(Vector<int> tokens)
    {
        var probs = new Matrix<float>(tokens.Length, Vocab);
        float rest = 0.1f / (Vocab - 1);
        for (int i = 0; i < tokens.Length; i++)
        {
            int next = (7 * tokens[i] + i + 3) % Vocab;
            for (int v = 0; v < Vocab; v++)
            {
                probs[i, v] = v == next ? 0.9f : rest;
            }
        }
        return probs;
    }

    private static int[] GreedyReference(int[] prompt, int maxNewTokens)
    {
        var tokens = new List<int>(prompt);
        for (int step = 0; step < maxNewTokens; step++)
        {
            var probs = TargetProbs(new Vector<int>(tokens.ToArray()));
            int last = tokens.Count - 1;
            int best = 0;
            for (int v = 1; v < Vocab; v++)
            {
                if (probs[last, v] > probs[last, best]) best = v;
            }
            tokens.Add(best);
        }
        return tokens.Skip(prompt.Length).ToArray();
    }

    [Theory(Timeout = 60000)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    public async Task SpeculativeDecoder_TemperatureZero_MatchesPlainGreedy(int numDraftTokens)
    {
        var prompt = new[] { 4, 9, 2 };
        var decoder = new SpeculativeDecoder<float>(
            new NGramDraftModel<float>(ngramSize: 2, vocabSize: Vocab, seed: 7),
            TargetProbs,
            new SpeculativeDecodingConfig<float> { NumDraftTokens = numDraftTokens, Seed = 11 });

        var result = await decoder.GenerateAsync(new Vector<int>(prompt), maxNewTokens: 12, temperature: 0f);

        Assert.Equal(GreedyReference(prompt, 12), result.NewTokens.ToArray());
    }

    // The tempered-target and residual-sampling path: for a fixed seed the output is reproducible, the requested length,
    // and in the vocabulary (an all-zero tempered row would collapse every token to the last id instead).
    [Fact]
    public async Task SpeculativeDecoder_PositiveTemperature_IsSeededAndInRange()
    {
        var prompt = new[] { 4, 9, 2 };
        async Task<int[]> Run()
        {
            var decoder = new SpeculativeDecoder<float>(
                new NGramDraftModel<float>(ngramSize: 2, vocabSize: Vocab, seed: 7),
                TargetProbs,
                new SpeculativeDecodingConfig<float> { NumDraftTokens = 3, Seed = 11 });
            var result = await decoder.GenerateAsync(new Vector<int>(prompt), maxNewTokens: 12, temperature: 0.7f);
            return result.NewTokens.ToArray();
        }

        var first = await Run();
        var second = await Run();

        Assert.Equal(12, first.Length);
        Assert.Equal(first, second);
        Assert.All(first, t => Assert.InRange(t, 0, Vocab - 1));
        Assert.True(first.Distinct().Count() > 1, "every token is the same id: the tempered distribution collapsed");
    }

    [Fact]
    public async Task TreeSpeculativeDecoder_TemperatureZero_MatchesPlainGreedy()
    {
        var prompt = new[] { 4, 9, 2 };
        var decoder = new TreeSpeculativeDecoder<float>(
            new NGramDraftModel<float>(ngramSize: 2, vocabSize: Vocab, seed: 7),
            sequences => sequences.Select(TargetProbs).ToList(),
            new TreeSpeculativeConfig { BranchFactor = 2, MaxDepth = 3, MaxNodes = 16, Seed = 11 });

        var result = await decoder.GenerateAsync(new Vector<int>(prompt), maxNewTokens: 12, temperature: 0f);

        Assert.Equal(GreedyReference(prompt, 12), result.NewTokens.ToArray());
    }

    [Fact]
    public void NGramDraftModel_TemperatureZero_DraftsTheArgmaxWithCertainty()
    {
        var model = new NGramDraftModel<float>(ngramSize: 2, vocabSize: 10, seed: 3);
        model.Train(new[] { new Vector<int>(new[] { 1, 2, 1, 2, 1, 3 }) });

        var draft = model.GenerateDraft(new Vector<int>(new[] { 1 }), numDraftTokens: 1, temperature: 0f);

        // After 1 the corpus continues with 2 twice and 3 once: greedy picks 2, as a one-hot.
        Assert.Equal(2, draft.Tokens[0]);
        Assert.Equal(1f, draft.TokenProbabilities[0]);
        var row = draft.Probabilities.GetRow(0);
        for (int v = 0; v < row.Length; v++)
        {
            Assert.Equal(v == 2 ? 1f : 0f, row[v]);
        }
    }

    [Fact]
    public void TemperatureSampling_Zero_IsOneHotAtArgmax_AndSamplingReturnsIt()
    {
        var dist = new Vector<float>(new[] { 0.1f, 0.6f, 0.3f });

        var greedy = TemperatureSampling<float>.Apply(dist, 0f);

        Assert.Equal(new[] { 0f, 1f, 0f }, greedy.ToArray());
        var random = new Random(0);
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal(1, TemperatureSampling<float>.Sample(greedy, random));
        }
    }

    [Fact]
    public void TemperatureSampling_PositiveTemperature_SharpensAndRenormalizes()
    {
        var dist = new Vector<float>(new[] { 0.2f, 0.8f });

        var sharpened = TemperatureSampling<float>.Apply(dist, 0.5f);

        // p^2 renormalized: 0.04 / 0.68 and 0.64 / 0.68.
        Assert.Equal(0.04f / 0.68f, sharpened[0], 5);
        Assert.Equal(0.64f / 0.68f, sharpened[1], 5);
    }
}
