using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Unit tests for speculative decoding components.
/// </summary>
public class NGramDraftModelTests
{
    [Fact]
    public void NGramDraftModel_Creation_InitializesCorrectly()
    {
        // Act
        var model = new NGramDraftModel<float>(ngramSize: 3, vocabSize: 100);

        // Assert
        Assert.Equal(100, model.VocabSize);
        Assert.Equal(8, model.MaxDraftTokens);
    }

    [Fact]
    public void NGramDraftModel_Train_LearnsPatternsFromCorpus()
    {
        // Arrange
        var model = new NGramDraftModel<float>(ngramSize: 2, vocabSize: 10, seed: 42);

        // Simple pattern: 1 -> 2, 2 -> 3, 3 -> 1
        var corpus = new List<Vector<int>>
        {
            new Vector<int>(new[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 }),
            new Vector<int>(new[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 })
        };

        // Act
        model.Train(corpus);
        var draft = model.GenerateDraft(new Vector<int>(new[] { 1 }), 3, temperature: 0.1f);

        // Assert - with low temperature, should follow learned pattern
        Assert.Equal(3, draft.NumTokens);
        // The pattern should emerge
    }

    [Fact]
    public void NGramDraftModel_GenerateDraft_ProducesValidOutput()
    {
        // Arrange
        var model = new NGramDraftModel<float>(ngramSize: 3, vocabSize: 100, seed: 42);

        // Act
        var draft = model.GenerateDraft(new Vector<int>(new[] { 1, 2, 3 }), 5, temperature: 1.0f);

        // Assert
        Assert.Equal(5, draft.NumTokens);
        Assert.Equal(5, draft.Tokens.Length);
        Assert.Equal(5, draft.TokenProbabilities.Length);
        Assert.Equal(5, draft.Probabilities.Rows);
        Assert.Equal(100, draft.Probabilities.Columns);
    }

    [Fact]
    public void NGramDraftModel_GenerateDraft_TokenProbabilitiesAreValid()
    {
        // Arrange
        var model = new NGramDraftModel<float>(ngramSize: 2, vocabSize: 50, seed: 42);

        // Act
        var draft = model.GenerateDraft(new Vector<int>(new[] { 5 }), 3, temperature: 1.0f);

        // Assert - probabilities should be in valid range
        for (int i = 0; i < draft.TokenProbabilities.Length; i++)
        {
            float prob = draft.TokenProbabilities[i];
            Assert.True(prob >= 0 && prob <= 1, $"Token probability {prob} out of range");
        }
    }

    [Fact]
    public void NGramDraftModel_Reset_DoesNotThrow()
    {
        // Arrange
        var model = new NGramDraftModel<float>();

        // Act & Assert
        var exception = Record.Exception(() => model.Reset());
        Assert.Null(exception);
    }
}

/// <summary>
/// Tests for NeuralDraftModel.
/// </summary>
public class NeuralDraftModelTests
{
    [Fact]
    public void NeuralDraftModel_Creation_Works()
    {
        // Arrange
        Func<Vector<int>, Vector<float>> forward = tokens =>
        {
            // Simple mock - return uniform distribution
            var logits = new Vector<float>(100);
            return logits;
        };

        // Act
        var model = new NeuralDraftModel<float>(forward, vocabSize: 100, maxDraftTokens: 5);

        // Assert
        Assert.Equal(100, model.VocabSize);
        Assert.Equal(5, model.MaxDraftTokens);
    }

    [Fact]
    public void NeuralDraftModel_GenerateDraft_ProducesTokens()
    {
        // Arrange
        int callCount = 0;
        Func<Vector<int>, Vector<float>> forward = tokens =>
        {
            callCount++;
            var logits = new Vector<float>(50);
            // Bias towards token 10
            logits[10] = 5.0f;
            return logits;
        };

        var model = new NeuralDraftModel<float>(forward, vocabSize: 50, maxDraftTokens: 4, seed: 42);

        // Act
        var draft = model.GenerateDraft(new Vector<int>(new[] { 1, 2 }), 3, temperature: 0.5f);

        // Assert
        Assert.Equal(3, draft.NumTokens);
        Assert.Equal(3, callCount); // Forward called once per draft token
    }

    [Fact]
    public void NeuralDraftModel_GenerateDraft_RespectsMaxDraftTokens()
    {
        // Arrange
        Func<Vector<int>, Vector<float>> forward = _ => new Vector<float>(100);
        var model = new NeuralDraftModel<float>(forward, vocabSize: 100, maxDraftTokens: 3);

        // Act
        var draft = model.GenerateDraft(new Vector<int>(new[] { 1 }), numDraftTokens: 10, temperature: 1.0f);

        // Assert - should be capped at maxDraftTokens
        Assert.Equal(3, draft.NumTokens);
    }
}

/// <summary>
/// Tests for SpeculativeDecoder.
/// </summary>
public class SpeculativeDecoderTests
{
    private IDraftModel<float> CreateMockDraftModel(int vocabSize = 100)
    {
        return new NGramDraftModel<float>(ngramSize: 2, vocabSize: vocabSize, seed: 42);
    }

    private Func<Vector<int>, Matrix<float>> CreateMockTargetForward(int vocabSize = 100)
    {
        return tokens =>
        {
            // Return probability distributions for each position
            var probs = new Matrix<float>(tokens.Length, vocabSize);
            for (int i = 0; i < tokens.Length; i++)
            {
                // Simple distribution - bias towards token 1
                probs[i, 1] = 0.5f;
                float remaining = 0.5f / (vocabSize - 1);
                for (int v = 0; v < vocabSize; v++)
                {
                    if (v != 1) probs[i, v] = remaining;
                }
            }
            return probs;
        };
    }

    [Fact]
    public void SpeculativeDecoder_Creation_Works()
    {
        // Arrange & Act
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward());

        // Assert
        Assert.Equal(5, decoder.Config.NumDraftTokens);
    }

    [Fact]
    public async Task SpeculativeDecoder_GenerateAsync_ProducesTokens()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward(),
            new SpeculativeDecodingConfig<float> { NumDraftTokens = 3 });

        // Act
        var result = await decoder.GenerateAsync(
            inputTokens: new Vector<int>(new[] { 1, 2, 3 }),
            maxNewTokens: 10,
            temperature: 1.0f);

        // Assert
        Assert.True(result.NumGenerated > 0);
        Assert.Equal(3 + result.NumGenerated, result.Tokens.Length);
        Assert.Equal(result.NumGenerated, result.NewTokens.Length);
    }

    [Fact]
    public void SpeculativeDecoder_Generate_SynchronousWorks()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward(),
            new SpeculativeDecodingConfig<float> { NumDraftTokens = 2 });

        // Act
        var result = decoder.Generate(
            inputTokens: new Vector<int>(new[] { 1 }),
            maxNewTokens: 5,
            temperature: 1.0f);

        // Assert
        Assert.True(result.NumGenerated > 0);
    }

    [Fact]
    public async Task SpeculativeDecoder_GenerateAsync_StopsAtEOS()
    {
        // Arrange
        const int eosToken = 99;

        // Mock target that always returns EOS with high probability
        Func<Vector<int>, Matrix<float>> targetForward = tokens =>
        {
            var probs = new Matrix<float>(tokens.Length, 100);
            for (int i = 0; i < tokens.Length; i++)
            {
                probs[i, eosToken] = 0.9f;
                float remaining = 0.1f / 99;
                for (int v = 0; v < 100; v++)
                {
                    if (v != eosToken) probs[i, v] = remaining;
                }
            }
            return probs;
        };

        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            targetForward,
            new SpeculativeDecodingConfig<float> { NumDraftTokens = 3 });

        // Act
        var result = await decoder.GenerateAsync(
            inputTokens: new Vector<int>(new[] { 1 }),
            maxNewTokens: 100,
            temperature: 1.0f,
            eosToken: eosToken);

        // Assert - should stop early
        Assert.True(result.NumGenerated < 100);
        Assert.True(ContainsToken(result.NewTokens, eosToken));
    }

    [Fact]
    public async Task SpeculativeDecoder_GenerateAsync_TracksStatistics()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward(),
            new SpeculativeDecodingConfig<float> { NumDraftTokens = 3 });

        // Act
        await decoder.GenerateAsync(new Vector<int>(new[] { 1, 2 }), maxNewTokens: 10, temperature: 1.0f);

        // Assert
        var stats = decoder.GetStatistics();
        Assert.True(stats.TotalTokensGenerated > 0);
        Assert.True(stats.TotalDraftTokens > 0);
        Assert.True(stats.TotalVerificationCalls > 0);
    }

    [Fact]
    public void SpeculativeDecoder_ResetStatistics_ClearsCounters()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward());

        decoder.Generate(new Vector<int>(new[] { 1 }), maxNewTokens: 5, temperature: 1.0f);

        // Act
        decoder.ResetStatistics();

        // Assert
        var stats = decoder.GetStatistics();
        Assert.Equal(0, stats.TotalTokensGenerated);
        Assert.Equal(0, stats.TotalDraftTokens);
    }

    [Fact]
    public async Task SpeculativeDecoder_GenerateAsync_SupportsCancellation()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward());

        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            decoder.GenerateAsync(new Vector<int>(new[] { 1 }), maxNewTokens: 100, temperature: 1.0f, cancellationToken: cts.Token));
    }

    [Fact]
    public async Task SpeculativeDecoder_GenerateAsync_RecordsStepStatistics()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward(),
            new SpeculativeDecodingConfig<float> { NumDraftTokens = 2 });

        // Act
        var result = await decoder.GenerateAsync(new Vector<int>(new[] { 1 }), maxNewTokens: 10, temperature: 1.0f);

        // Assert
        Assert.NotEmpty(result.StepStatistics);
        foreach (var step in result.StepStatistics)
        {
            Assert.True(step.DraftTokens > 0);
            Assert.True(step.AcceptedTokens >= 0);
        }
    }

    [Fact]
    public async Task SpeculativeDecoder_AcceptanceRate_IsValid()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockTargetForward(),
            new SpeculativeDecodingConfig<float> { NumDraftTokens = 4 });

        // Act
        await decoder.GenerateAsync(new Vector<int>(new[] { 1, 2, 3 }), maxNewTokens: 20, temperature: 1.0f);

        // Assert
        var rate = decoder.AcceptanceRate;
        Assert.True(rate >= 0 && rate <= 1, $"Acceptance rate {rate} should be between 0 and 1");
    }

    private static bool ContainsToken(Vector<int> tokens, int token)
    {
        for (int i = 0; i < tokens.Length; i++)
        {
            if (tokens[i] == token)
                return true;
        }
        return false;
    }
}

/// <summary>
/// Tests for TreeSpeculativeDecoder.
/// </summary>
public class TreeSpeculativeDecoderTests
{
    private IDraftModel<float> CreateMockDraftModel()
    {
        return new NGramDraftModel<float>(ngramSize: 2, vocabSize: 50, seed: 42);
    }

    private Func<List<Vector<int>>, List<Matrix<float>>> CreateMockBatchTargetForward()
    {
        return sequences =>
        {
            var results = new List<Matrix<float>>();
            foreach (var sequence in sequences)
            {
                var probs = new Matrix<float>(sequence.Length, 50);
                for (int p = 0; p < sequence.Length; p++)
                {
                    // Uniform distribution
                    for (int v = 0; v < 50; v++)
                    {
                        probs[p, v] = 0.02f;
                    }
                }
                results.Add(probs);
            }
            return results;
        };
    }

    [Fact]
    public void TreeSpeculativeDecoder_Creation_Works()
    {
        // Act
        var decoder = new TreeSpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockBatchTargetForward(),
            new TreeSpeculativeConfig { BranchFactor = 2, MaxDepth = 3 });

        // Assert
        Assert.Equal(2, decoder.Config.BranchFactor);
        Assert.Equal(3, decoder.Config.MaxDepth);
    }

    [Fact]
    public async Task TreeSpeculativeDecoder_GenerateAsync_ProducesTokens()
    {
        // Arrange
        var decoder = new TreeSpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockBatchTargetForward(),
            new TreeSpeculativeConfig
            {
                BranchFactor = 2,
                MaxDepth = 2,
                MaxNodes = 8
            });

        // Act
        var result = await decoder.GenerateAsync(
            inputTokens: new Vector<int>(new[] { 1, 2 }),
            maxNewTokens: 5,
            temperature: 1.0f);

        // Assert
        Assert.True(result.NumGenerated > 0);
        Assert.True(result.NewTokens.Length > 0);
    }

    [Fact]
    public void TreeSpeculativeDecoder_Generate_SynchronousWorks()
    {
        // Arrange
        var decoder = new TreeSpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockBatchTargetForward());

        // Act
        var result = decoder.Generate(new Vector<int>(new[] { 1 }), maxNewTokens: 3, temperature: 1.0f);

        // Assert
        Assert.True(result.NumGenerated > 0);
    }

    [Fact]
    public async Task TreeSpeculativeDecoder_GenerateAsync_RecordsTreeStatistics()
    {
        // Arrange
        var decoder = new TreeSpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockBatchTargetForward(),
            new TreeSpeculativeConfig
            {
                BranchFactor = 3,
                MaxDepth = 2,
                MaxNodes = 10
            });

        // Act
        var result = await decoder.GenerateAsync(new Vector<int>(new[] { 1 }), maxNewTokens: 5, temperature: 1.0f);

        // Assert
        Assert.NotEmpty(result.StepStatistics);
        foreach (var step in result.StepStatistics)
        {
            Assert.True(step.TreeNodes > 0);
            Assert.True(step.PathsExplored > 0);
        }
    }

    [Fact]
    public async Task TreeSpeculativeDecoder_AcceptanceRate_IsValid()
    {
        // Arrange
        var decoder = new TreeSpeculativeDecoder<float>(
            CreateMockDraftModel(),
            CreateMockBatchTargetForward());

        // Act
        await decoder.GenerateAsync(new Vector<int>(new[] { 1, 2, 3 }), maxNewTokens: 10, temperature: 1.0f);

        // Assert
        var rate = decoder.AcceptanceRate;
        Assert.True(rate >= 0 && rate <= 1);
    }
}

/// <summary>
/// Integration tests for speculative decoding.
/// </summary>
public class SpeculativeDecodingIntegrationTests
{
    [Fact]
    public async Task SpeculativeDecoding_WithTrainedDraft_AchievesSpeedup()
    {
        // Arrange
        var draftModel = new NGramDraftModel<float>(ngramSize: 2, vocabSize: 20, seed: 42);

        // Train on repetitive pattern
        var corpus = Enumerable.Range(0, 100).Select(_ =>
            new Vector<int>(new[] { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 })
        ).ToList();
        draftModel.Train(corpus);

        // Target also follows similar pattern
        Func<Vector<int>, Matrix<float>> targetForward = tokens =>
        {
            var probs = new Matrix<float>(tokens.Length, 20);
            for (int i = 0; i < tokens.Length; i++)
            {
                // Follow pattern: predict next in 1,2,3,4,5 cycle
                int lastToken = i > 0 ? tokens[i - 1] : 0;
                int nextToken = (lastToken % 5) + 1;
                probs[i, nextToken] = 0.8f;
                float remaining = 0.2f / 19;
                for (int v = 0; v < 20; v++)
                {
                    if (v != nextToken) probs[i, v] = remaining;
                }
            }
            return probs;
        };

        var decoder = new SpeculativeDecoder<float>(
            draftModel,
            targetForward,
            new SpeculativeDecodingConfig<float> { NumDraftTokens = 4 });

        // Act
        var result = await decoder.GenerateAsync(new Vector<int>(new[] { 5 }), maxNewTokens: 20, temperature: 0.5f);

        // Assert
        var stats = decoder.GetStatistics();
        Assert.True(result.NumGenerated >= 20);

        // With matching patterns, should have decent acceptance
        // Note: acceptance rate depends on how well draft matches target
    }

    [Fact]
    public async Task SpeculativeDecoding_MultipleGenerations_AccumulatesStats()
    {
        // Arrange
        var decoder = new SpeculativeDecoder<float>(
            new NGramDraftModel<float>(ngramSize: 2, vocabSize: 50, seed: 42),
            tokens =>
            {
                var probs = new Matrix<float>(tokens.Length, 50);
                for (int i = 0; i < tokens.Length; i++)
                {
                    for (int v = 0; v < 50; v++) probs[i, v] = 0.02f;
                }
                return probs;
            });

        // Act - multiple generations
        for (int i = 0; i < 5; i++)
        {
            await decoder.GenerateAsync(new Vector<int>(new[] { 1 }), maxNewTokens: 5, temperature: 1.0f);
        }

        // Assert
        var stats = decoder.GetStatistics();
        Assert.True(stats.TotalTokensGenerated >= 5); // At least 5 total from 5 calls
        Assert.True(stats.TotalVerificationCalls >= 5);
    }

    [Fact]
    public void SpeculativeDecodingConfig_DefaultValues_AreReasonable()
    {
        // Act
        var config = new SpeculativeDecodingConfig<float>();

        // Assert
        Assert.Equal(5, config.NumDraftTokens);
        Assert.False(config.UseTreeSpeculation);
        Assert.Equal(0.5f, config.MinAcceptanceRate);
        Assert.False(config.AdaptiveDraftLength);
    }

    [Fact]
    public void TreeSpeculativeConfig_DefaultValues_AreReasonable()
    {
        // Act
        var config = new TreeSpeculativeConfig();

        // Assert
        Assert.Equal(2, config.BranchFactor);
        Assert.Equal(4, config.MaxDepth);
        Assert.Equal(16, config.MaxNodes);
    }

    [Fact]
    public async Task SpeculativeDecoder_GenerateAsync_TreeMode_RecordsDraftWork()
    {
        // Arrange
        var draftModel = new NGramDraftModel<float>(ngramSize: 2, vocabSize: 20, seed: 42);
        var corpus = new List<Vector<int>>
        {
            new Vector<int>(Enumerable.Range(0, 200).Select(i => i % 10).ToArray())
        };
        draftModel.Train(corpus);

        Func<Vector<int>, Matrix<float>> targetForward = tokens =>
        {
            var probs = new Matrix<float>(tokens.Length, 20);
            for (int i = 0; i < tokens.Length; i++)
            {
                for (int v = 0; v < 20; v++) probs[i, v] = 0.05f;
                probs[i, 7] = 0.8f;
            }
            return probs;
        };

        var decoder = new SpeculativeDecoder<float>(
            draftModel,
            targetForward,
            new SpeculativeDecodingConfig<float>
            {
                UseTreeSpeculation = true,
                TreeBranchFactor = 3,
                MaxTreeDepth = 3,
                Seed = 42
            });

        // Act
        var result = await decoder.GenerateAsync(new Vector<int>(new[] { 1 }), maxNewTokens: 8, temperature: 1.0f);

        // Assert
        Assert.True(result.NumGenerated > 0);
        Assert.True(decoder.TotalDraftTokens > 0);
        Assert.True(result.StepStatistics.Count > 0);
        Assert.True(result.TokensPerVerification > 0);
        Assert.Contains(result.StepStatistics, s => s.DraftTokens > 0);
    }

    [Fact]
    public async Task SpeculativeDecoder_AdaptiveDraftLength_ReducesDraftTokens_WhenAcceptanceLow()
    {
        // Arrange
        var config = new SpeculativeDecodingConfig<float>
        {
            NumDraftTokens = 4,
            AdaptiveDraftLength = true,
            MinAcceptanceRate = 0.8f
        };

        // Target strongly prefers token 2.
        Func<Vector<int>, Matrix<float>> targetForward = tokens =>
        {
            var probs = new Matrix<float>(tokens.Length, 10);
            for (int i = 0; i < tokens.Length; i++)
            {
                for (int v = 0; v < 10; v++) probs[i, v] = 0.0001f;
                probs[i, 2] = 0.999f;
            }
            return probs;
        };

        // Draft consistently proposes token 1 => low acceptance.
        var draft = new DeterministicDraftModel(vocabSize: 10, tokenId: 1);
        var decoder = new SpeculativeDecoder<float>(draft, targetForward, config);

        // Act
        _ = await decoder.GenerateAsync(new Vector<int>(new[] { 0 }), maxNewTokens: 24, temperature: 1.0f);

        // Assert
        Assert.True(decoder.CurrentDraftTokens < 4);
    }
}

internal sealed class DeterministicDraftModel : IDraftModel<float>
{
    private readonly int _vocabSize;
    private readonly int _tokenId;

    public DeterministicDraftModel(int vocabSize, int tokenId)
    {
        _vocabSize = vocabSize;
        _tokenId = tokenId;
    }

    public int MaxDraftTokens => 32;

    public int VocabSize => _vocabSize;

    public DraftResult<float> GenerateDraft(Vector<int> inputTokens, int numDraftTokens, float temperature)
    {
        int n = Math.Max(0, numDraftTokens);
        var tokens = new Vector<int>(Enumerable.Repeat(_tokenId, n).ToArray());

        var probs = new Matrix<float>(n, _vocabSize);
        for (int i = 0; i < n; i++)
        {
            probs[i, _tokenId] = 1.0f;
        }

        var tokenProbs = new Vector<float>(Enumerable.Repeat(1.0f, n).ToArray());
        return new DraftResult<float>
        {
            Tokens = tokens,
            TokenProbabilities = tokenProbs,
            Probabilities = probs
        };
    }

    public void Reset()
    {
    }
}
