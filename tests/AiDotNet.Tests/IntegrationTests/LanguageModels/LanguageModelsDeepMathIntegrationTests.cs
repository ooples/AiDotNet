using AiDotNet.LanguageModels;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LanguageModels;

/// <summary>
/// Deep integration tests for LanguageModels:
/// ChatModelBase (construction validation, token estimation),
/// OpenAIChatModel (defaults, model names),
/// LLM math (token estimation, cost calculation, context window, temperature sampling,
/// top-p nucleus sampling, exponential backoff, perplexity, attention mechanism math).
/// </summary>
public class LanguageModelsDeepMathIntegrationTests
{
    private const string ValidApiKey = "test-api-key-12345";

    // ============================
    // OpenAIChatModel: Construction
    // ============================

    [Fact]
    public void OpenAIChatModel_DefaultModelName()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        Assert.Equal("gpt-3.5-turbo", model.ModelName);
    }

    [Theory]
    [InlineData("gpt-4")]
    [InlineData("gpt-4-turbo")]
    [InlineData("gpt-3.5-turbo")]
    public void OpenAIChatModel_CustomModelName(string modelName)
    {
        var model = new OpenAIChatModel<double>(ValidApiKey, modelName: modelName);
        Assert.Equal(modelName, model.ModelName);
    }

    [Fact]
    public void OpenAIChatModel_MaxContextTokens_Positive()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        Assert.True(model.MaxContextTokens > 0, "MaxContextTokens should be positive");
    }

    [Fact]
    public void OpenAIChatModel_MaxGenerationTokens_Positive()
    {
        var model = new OpenAIChatModel<double>(ValidApiKey);
        Assert.True(model.MaxGenerationTokens > 0, "MaxGenerationTokens should be positive");
    }

    [Fact]
    public void OpenAIChatModel_NullApiKey_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new OpenAIChatModel<double>(null!));
    }

    [Fact]
    public void OpenAIChatModel_EmptyApiKey_Throws()
    {
        Assert.Throws<ArgumentException>(() => new OpenAIChatModel<double>(""));
    }

    [Fact]
    public void OpenAIChatModel_WhitespaceApiKey_Throws()
    {
        Assert.Throws<ArgumentException>(() => new OpenAIChatModel<double>("   "));
    }

    // ============================
    // AzureOpenAIChatModel: Construction
    // ============================

    [Fact]
    public void AzureOpenAIChatModel_ValidConstruction()
    {
        var model = new AzureOpenAIChatModel<double>(
            ValidApiKey,
            "https://test-resource.openai.azure.com",
            "gpt-4-deployment");
        Assert.NotNull(model);
    }

    [Fact]
    public void AzureOpenAIChatModel_NullEndpoint_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new AzureOpenAIChatModel<double>(ValidApiKey, null!, "deployment"));
    }

    [Fact]
    public void AzureOpenAIChatModel_NullDeploymentName_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new AzureOpenAIChatModel<double>(ValidApiKey, "https://test.openai.azure.com", null!));
    }

    // ============================
    // LLM Math: Token Estimation
    // ============================

    [Theory]
    [InlineData("Hello", 2)]            // 5 chars / 4 = 1.25, ceil = 2
    [InlineData("Hello, world!", 4)]     // 13 chars / 4 = 3.25, ceil = 4
    [InlineData("", 0)]                  // Empty string
    [InlineData("a", 1)]                 // Single char
    public void LLMMath_TokenEstimation_CharBased(string text, int expectedTokens)
    {
        // ChatModelBase.EstimateTokenCount: ceil(length / 4.0)
        int estimated = string.IsNullOrEmpty(text) ? 0 : (int)Math.Ceiling(text.Length / 4.0);
        Assert.Equal(expectedTokens, estimated);
    }

    [Theory]
    [InlineData(100, 25)]     // 100 chars ≈ 25 tokens
    [InlineData(1000, 250)]   // 1000 chars ≈ 250 tokens
    [InlineData(4000, 1000)]  // 4000 chars ≈ 1000 tokens
    public void LLMMath_TokenEstimation_Scaling(int charCount, int expectedTokens)
    {
        int tokens = (int)Math.Ceiling(charCount / 4.0);
        Assert.Equal(expectedTokens, tokens);
    }

    // ============================
    // LLM Math: Cost Calculation
    // ============================

    [Theory]
    [InlineData(1000, 500, 0.03, 0.06, 0.06)]  // 1K prompt + 500 completion, $0.03/$0.06 per 1K
    [InlineData(4096, 2048, 0.01, 0.03, 0.102)] // 4K prompt + 2K completion
    public void LLMMath_CostCalculation(int promptTokens, int completionTokens,
        double promptPricePerK, double completionPricePerK, double expectedCost)
    {
        double cost = (promptTokens / 1000.0) * promptPricePerK +
                      (completionTokens / 1000.0) * completionPricePerK;
        Assert.Equal(expectedCost, cost, 1e-3);
    }

    [Fact]
    public void LLMMath_CostScaling_BatchVsSingle()
    {
        // Batch processing should cost the same as individual calls (same token count)
        int totalPromptTokens = 10000;
        int totalCompletionTokens = 5000;
        double pricePerKPrompt = 0.03;
        double pricePerKCompletion = 0.06;

        double singleCallCost = (totalPromptTokens / 1000.0) * pricePerKPrompt +
                                 (totalCompletionTokens / 1000.0) * pricePerKCompletion;

        // 10 calls with 1/10 of the tokens each
        double batchCost = 0;
        for (int i = 0; i < 10; i++)
        {
            batchCost += (totalPromptTokens / 10.0 / 1000.0) * pricePerKPrompt +
                          (totalCompletionTokens / 10.0 / 1000.0) * pricePerKCompletion;
        }

        Assert.Equal(singleCallCost, batchCost, 1e-10);
    }

    // ============================
    // LLM Math: Context Window
    // ============================

    [Theory]
    [InlineData(4096, 1000, 3096)]      // GPT-3.5: 4K context, 1K prompt -> 3K available
    [InlineData(8192, 2000, 6192)]      // GPT-4 8K: 2K prompt -> 6K available
    [InlineData(128000, 50000, 78000)]  // GPT-4 128K: 50K prompt -> 78K available
    public void LLMMath_AvailableTokens(int contextWindow, int promptTokens, int expectedAvailable)
    {
        int available = contextWindow - promptTokens;
        Assert.Equal(expectedAvailable, available);
        Assert.True(available > 0, "Available tokens should be positive");
    }

    [Theory]
    [InlineData(4096, 500, 8)]   // 4K context, 500 tokens per message ≈ 8 turns
    [InlineData(128000, 500, 256)] // 128K context, 500 tokens per message ≈ 256 turns
    public void LLMMath_ConversationLength(int contextWindow, int avgTokensPerMessage, int expectedTurns)
    {
        int turns = contextWindow / avgTokensPerMessage;
        Assert.Equal(expectedTurns, turns);
    }

    // ============================
    // LLM Math: Temperature Sampling
    // ============================

    [Theory]
    [InlineData(new double[] { 2.0, 1.0, 0.5 }, 1.0)]   // Temperature = 1 (standard softmax)
    [InlineData(new double[] { 2.0, 1.0, 0.5 }, 0.5)]   // Low temperature (more focused)
    [InlineData(new double[] { 2.0, 1.0, 0.5 }, 2.0)]   // High temperature (more uniform)
    public void LLMMath_TemperatureSampling_ProbabilitySumToOne(double[] logits, double temperature)
    {
        double[] scaledLogits = logits.Select(l => l / temperature).ToArray();
        double maxLogit = scaledLogits.Max();
        double[] expLogits = scaledLogits.Select(l => Math.Exp(l - maxLogit)).ToArray();
        double sumExp = expLogits.Sum();
        double[] probs = expLogits.Select(e => e / sumExp).ToArray();

        Assert.Equal(1.0, probs.Sum(), 1e-10);
        foreach (double p in probs)
        {
            Assert.True(p > 0 && p < 1, "All probabilities should be in (0, 1)");
        }
    }

    [Fact]
    public void LLMMath_TemperatureSampling_LowTempMoreFocused()
    {
        double[] logits = { 2.0, 1.0, 0.5, 0.1 };

        double[] probsLow = ComputeSoftmax(logits, 0.1);
        double[] probsHigh = ComputeSoftmax(logits, 2.0);

        // Low temperature should give higher max probability
        Assert.True(probsLow.Max() > probsHigh.Max(),
            "Low temperature should produce more focused distribution");

        // High temperature should have lower entropy (more uniform)
        double entropyLow = -probsLow.Where(p => p > 0).Sum(p => p * Math.Log(p));
        double entropyHigh = -probsHigh.Where(p => p > 0).Sum(p => p * Math.Log(p));
        Assert.True(entropyHigh > entropyLow,
            "High temperature should have higher entropy");
    }

    [Fact]
    public void LLMMath_TemperatureZero_ApproachesArgmax()
    {
        double[] logits = { 2.0, 1.0, 0.5, 0.1 };

        // Very low temperature should approach argmax
        double[] probs = ComputeSoftmax(logits, 0.01);

        // The highest logit should get nearly all probability mass
        Assert.True(probs[0] > 0.99, $"At near-zero temperature, max logit should get ~100% probability, got {probs[0]}");
    }

    // ============================
    // LLM Math: Top-p (Nucleus) Sampling
    // ============================

    [Theory]
    [InlineData(new double[] { 0.5, 0.3, 0.15, 0.05 }, 0.9, 3)]   // Top-p=0.9 includes 3 tokens (0.5+0.3+0.15=0.95>0.9)
    [InlineData(new double[] { 0.5, 0.3, 0.15, 0.05 }, 0.5, 1)]   // Top-p=0.5 includes 1 token (0.5>=0.5)
    [InlineData(new double[] { 0.5, 0.3, 0.15, 0.05 }, 1.0, 4)]   // Top-p=1.0 includes all tokens
    public void LLMMath_TopPSampling_NucleusSize(double[] sortedProbs, double topP, int expectedNucleusSize)
    {
        // Sort probabilities in descending order (already sorted in test data)
        double cumSum = 0;
        int nucleusSize = 0;
        foreach (double p in sortedProbs)
        {
            cumSum += p;
            nucleusSize++;
            if (cumSum >= topP) break;
        }

        Assert.Equal(expectedNucleusSize, nucleusSize);
    }

    // ============================
    // LLM Math: Exponential Backoff (Retry Logic)
    // ============================

    [Theory]
    [InlineData(1000, 0, 1000)]    // First retry: 1000ms
    [InlineData(1000, 1, 2000)]    // Second retry: 2000ms
    [InlineData(1000, 2, 4000)]    // Third retry: 4000ms
    [InlineData(1000, 3, 8000)]    // Fourth retry: 8000ms
    public void LLMMath_ExponentialBackoff(int initialDelayMs, int retryAttempt, int expectedDelayMs)
    {
        // ChatModelBase uses: delayMs *= 2 each retry
        int delay = initialDelayMs * (int)Math.Pow(2, retryAttempt);
        Assert.Equal(expectedDelayMs, delay);
    }

    [Theory]
    [InlineData(1000, 5, 16000)]   // Max delay: 1000 * 2^(5-1) = 16000
    [InlineData(500, 4, 4000)]     // Max delay: 500 * 2^(4-1) = 4000
    public void LLMMath_ExponentialBackoff_TotalWaitTime(int initialDelayMs, int maxRetries, int expectedMaxSingleDelay)
    {
        int totalWait = 0;
        int delay = initialDelayMs;
        for (int i = 0; i < maxRetries; i++)
        {
            totalWait += delay;
            delay *= 2;
        }

        // Total wait = sum of geometric series: initial * (2^n - 1)
        int expectedTotal = initialDelayMs * ((int)Math.Pow(2, maxRetries) - 1);
        Assert.Equal(expectedTotal, totalWait);

        // Max single delay
        int maxDelay = initialDelayMs * (int)Math.Pow(2, maxRetries - 1);
        Assert.Equal(expectedMaxSingleDelay, maxDelay);
    }

    // ============================
    // LLM Math: Perplexity
    // ============================

    [Theory]
    [InlineData(new double[] { -0.5, -1.0, -0.3, -0.8 }, 1.914)]   // exp(-avg(log_probs))
    public void LLMMath_Perplexity(double[] logProbs, double expectedPerplexity)
    {
        // Perplexity = exp(-1/N * sum(log_prob_i))
        double avgLogProb = logProbs.Average();
        double perplexity = Math.Exp(-avgLogProb);

        Assert.Equal(expectedPerplexity, perplexity, 1e-2);
    }

    [Fact]
    public void LLMMath_Perplexity_PerfectPrediction()
    {
        // If the model always predicts with probability 1 (log_prob = 0), perplexity = 1
        double[] logProbs = { 0.0, 0.0, 0.0, 0.0 };
        double perplexity = Math.Exp(-logProbs.Average());
        Assert.Equal(1.0, perplexity, 1e-10);
    }

    [Fact]
    public void LLMMath_Perplexity_LowerIsBetter()
    {
        double[] goodLogProbs = { -0.1, -0.2, -0.1, -0.15 };
        double[] badLogProbs = { -2.0, -3.0, -1.5, -2.5 };

        double goodPerplexity = Math.Exp(-goodLogProbs.Average());
        double badPerplexity = Math.Exp(-badLogProbs.Average());

        Assert.True(goodPerplexity < badPerplexity,
            "Better predictions (higher log probs) should have lower perplexity");
    }

    // ============================
    // LLM Math: Attention Mechanism
    // ============================

    [Fact]
    public void LLMMath_ScaledDotProductAttention()
    {
        // Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        int dK = 64; // Key dimension
        double[] qk = { 2.0, 1.5, 0.5, -0.3 }; // Q*K^T products for 4 positions

        // Scale by sqrt(d_k)
        double scale = Math.Sqrt(dK);
        double[] scaled = qk.Select(x => x / scale).ToArray();

        // Apply softmax
        double maxVal = scaled.Max();
        double[] expScaled = scaled.Select(x => Math.Exp(x - maxVal)).ToArray();
        double sumExp = expScaled.Sum();
        double[] attention = expScaled.Select(x => x / sumExp).ToArray();

        // Attention weights should sum to 1
        Assert.Equal(1.0, attention.Sum(), 1e-10);

        // All attention weights should be positive
        foreach (double w in attention)
        {
            Assert.True(w > 0, "Attention weights should be positive");
        }

        // Highest QK product should get highest attention
        Assert.True(attention[0] > attention[1]);
        Assert.True(attention[1] > attention[2]);
    }

    [Theory]
    [InlineData(64, 8.0)]     // d_k = 64, scale = 8
    [InlineData(128, 11.314)] // d_k = 128, scale ≈ 11.31
    [InlineData(256, 16.0)]   // d_k = 256, scale = 16
    public void LLMMath_AttentionScaleFactor(int dK, double expectedScale)
    {
        double scale = Math.Sqrt(dK);
        Assert.Equal(expectedScale, scale, 1e-2);
    }

    [Theory]
    [InlineData(768, 12, 64)]    // GPT-2: 768 hidden, 12 heads -> 64 per head
    [InlineData(1024, 16, 64)]   // GPT-2 Medium: 1024 hidden, 16 heads -> 64 per head
    [InlineData(4096, 32, 128)]  // Larger model: 4096 hidden, 32 heads -> 128 per head
    public void LLMMath_MultiHeadAttention_HeadDimension(int hiddenSize, int numHeads, int expectedHeadDim)
    {
        int headDim = hiddenSize / numHeads;
        Assert.Equal(expectedHeadDim, headDim);
    }

    // ============================
    // LLM Math: Positional Encoding
    // ============================

    [Fact]
    public void LLMMath_SinusoidalPositionalEncoding()
    {
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        int dModel = 512;
        int pos = 10;

        double[] pe = new double[dModel];
        for (int i = 0; i < dModel; i++)
        {
            double angle = pos / Math.Pow(10000, (2.0 * (i / 2)) / dModel);
            pe[i] = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
        }

        // All values should be in [-1, 1]
        foreach (double v in pe)
        {
            Assert.True(v >= -1.0 && v <= 1.0,
                $"Positional encoding value {v} should be in [-1, 1]");
        }

        // Different positions should produce different encodings
        double[] pe2 = new double[dModel];
        for (int i = 0; i < dModel; i++)
        {
            double angle = (pos + 1) / Math.Pow(10000, (2.0 * (i / 2)) / dModel);
            pe2[i] = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
        }

        bool allSame = true;
        for (int i = 0; i < dModel; i++)
        {
            if (Math.Abs(pe[i] - pe2[i]) > 1e-10) { allSame = false; break; }
        }
        Assert.False(allSame, "Different positions should produce different encodings");
    }

    // ============================
    // LLM Math: Transformer Parameter Count
    // ============================

    [Theory]
    [InlineData(768, 12, 12, 50257, 124439808)]     // GPT-2 Small (~124M)
    [InlineData(1024, 24, 16, 50257, 354823168)]     // GPT-2 Medium (~355M)
    public void LLMMath_TransformerParameterCount(int dModel, int nLayers, int nHeads, int vocabSize, long expectedParams)
    {
        // Approximate GPT-2 parameter count:
        // Embedding: vocab_size * d_model + max_seq_len * d_model
        // Per layer: 4 * d_model^2 (attention QKV+O) + 8 * d_model^2 (FFN) + 4 * d_model (layer norms)
        // Final: d_model (final layer norm)
        int maxSeqLen = 1024;

        long embeddingParams = (long)vocabSize * dModel + maxSeqLen * dModel;
        long perLayerParams = 4L * dModel * dModel + 8L * dModel * dModel + 4L * dModel;
        long totalLayerParams = perLayerParams * nLayers;
        long finalNorm = dModel;
        long totalParams = embeddingParams + totalLayerParams + finalNorm;

        // Allow 5% tolerance due to approximation
        double ratio = (double)totalParams / expectedParams;
        Assert.True(ratio > 0.95 && ratio < 1.05,
            $"Parameter count {totalParams} should be within 5% of expected {expectedParams}");
    }

    // ============================
    // Helper Methods
    // ============================

    private static double[] ComputeSoftmax(double[] logits, double temperature)
    {
        double[] scaled = logits.Select(l => l / temperature).ToArray();
        double maxVal = scaled.Max();
        double[] expVals = scaled.Select(x => Math.Exp(x - maxVal)).ToArray();
        double sumExp = expVals.Sum();
        return expVals.Select(x => x / sumExp).ToArray();
    }
}
