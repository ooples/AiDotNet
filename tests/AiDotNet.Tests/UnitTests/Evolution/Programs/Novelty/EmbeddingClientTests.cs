using AiDotNet.Agentic.Embeddings;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs.Novelty;

public sealed class EmbeddingClientTests
{
    [Fact]
    public async Task DeterministicClientReturnsTheSameVectorForTheSameTextEveryTime()
    {
        var client = new DeterministicEmbeddingClient(dimensions: 32);
        EmbeddingBatch first = await client.EmbedAsync(new[] { "def f(x): return x + 1" });
        EmbeddingBatch second = await client.EmbedAsync(new[] { "def f(x): return x + 1" });

        Assert.True(first.Succeeded);
        Assert.True(second.Succeeded);
        Assert.Equal(32, first.Vectors[0].Dimensions);
        Assert.Equal(first.Vectors[0].Components, second.Vectors[0].Components);
        Assert.Equal(1.0, EmbeddingVector.CosineSimilarity(first.Vectors[0], second.Vectors[0]), 10);
    }

    [Fact]
    public void DeterministicClientRatesSimilarTextAboveUnrelatedText()
    {
        var client = new DeterministicEmbeddingClient(dimensions: 128);
        EmbeddingVector baseline = client.Embed("total = 0 for item in items total += item return total");
        EmbeddingVector near = client.Embed("total = 0 for item in items total += item return total + 0");
        EmbeddingVector far = client.Embed("import socket connect listen accept recv send close");

        Assert.True(EmbeddingVector.CosineSimilarity(baseline, near)
            > EmbeddingVector.CosineSimilarity(baseline, far));
    }

    [Fact]
    public void DeterministicClientNeverProducesAZeroMagnitudeVector()
    {
        var client = new DeterministicEmbeddingClient(dimensions: 8);
        foreach (string text in new[] { string.Empty, "   ", "\n\t", "a", "a a a a" })
        {
            EmbeddingVector vector = client.Embed(text);
            Assert.True(vector.Magnitude > 0.0);
            Assert.Equal(1.0, EmbeddingVector.CosineSimilarity(vector, vector), 10);
        }
    }

    [Fact]
    public async Task EveryClientRejectsAnEmptyOrNullBatchAsAnArgumentError()
    {
        var client = new DeterministicEmbeddingClient();
        await Assert.ThrowsAsync<ArgumentException>(async () =>
            await client.EmbedAsync(Array.Empty<string>()));
#pragma warning disable CS8625
        await Assert.ThrowsAsync<ArgumentNullException>(async () => await client.EmbedAsync(null));
        await Assert.ThrowsAsync<ArgumentException>(async () =>
            await client.EmbedAsync(new string[] { null }));
#pragma warning restore CS8625
    }

    [Fact]
    public void EmbeddingVectorRejectsNonFiniteComponentsAndAnEmptyVector()
    {
        Assert.Throws<ArgumentException>(() => new EmbeddingVector(Array.Empty<double>()));
        Assert.Throws<ArgumentException>(() => new EmbeddingVector(new[] { 1.0, double.NaN }));
        Assert.Throws<ArgumentException>(() => new EmbeddingVector(new[] { double.PositiveInfinity }));
    }

    [Fact]
    public void CosineSimilarityIsZeroForMismatchedLengths()
    {
        var shorter = new EmbeddingVector(new[] { 1.0, 0.0 });
        var longer = new EmbeddingVector(new[] { 1.0, 0.0, 0.0 });

        Assert.Equal(0.0, EmbeddingVector.CosineSimilarity(shorter, longer));
        Assert.Equal(0.0, EmbeddingVector.CosineSimilarity(
            new EmbeddingVector(new[] { 0.0, 0.0 }), shorter));
    }

    [Fact]
    public void FailedBatchesCarryABoundedReasonAndNoVectors()
    {
        EmbeddingBatch failure = EmbeddingBatch.Failure(new string('x', 5_000));

        Assert.False(failure.Succeeded);
        Assert.Empty(failure.Vectors);
        Assert.Equal(EmbeddingBatch.MaxFailureReasonLength, failure.FailureReason.Length);
    }

    [Fact]
    public async Task CachingClientAnswersARepeatedCandidateWithoutTouchingTheProvider()
    {
        var provider = new DeterministicEmbeddingClient(dimensions: 16);
        var cache = new CachingEmbeddingClient(provider);
        var texts = new[] { "candidate one", "candidate two" };

        EmbeddingBatch first = await cache.EmbedAsync(texts);
        long callsAfterFirst = provider.Calls;
        EmbeddingBatch second = await cache.EmbedAsync(texts);

        Assert.True(first.Succeeded);
        Assert.True(second.Succeeded);

        // This is the exceed over upstream, which embeds on every insertion decision with no cache anywhere.
        Assert.Equal(1L, callsAfterFirst);
        Assert.Equal(1L, provider.Calls);
        Assert.Equal(1L, cache.InnerCalls);
        Assert.Equal(2L, cache.Hits);
        Assert.Equal(2L, cache.Misses);
        Assert.Equal(first.Vectors[0].Components, second.Vectors[0].Components);
        Assert.Equal(first.Vectors[1].Components, second.Vectors[1].Components);
    }

    [Fact]
    public async Task CachingClientForwardsOnlyTheMissesOfAPartiallyCachedBatch()
    {
        var provider = new DeterministicEmbeddingClient(dimensions: 16);
        var cache = new CachingEmbeddingClient(provider);

        await cache.EmbedAsync(new[] { "alpha" });
        Assert.Equal(1L, provider.TextsEmbedded);

        EmbeddingBatch batch = await cache.EmbedAsync(new[] { "alpha", "beta", "alpha" });

        Assert.True(batch.Succeeded);
        Assert.Equal(3, batch.Vectors.Count);

        // Only "beta" was unknown, and the repeated "alpha" inside the batch was not forwarded twice.
        Assert.Equal(2L, provider.TextsEmbedded);
        Assert.Equal(batch.Vectors[0].Components, batch.Vectors[2].Components);
    }

    [Fact]
    public async Task CachingClientNeverCachesAFailureSoATransientErrorDoesNotPoisonTheRun()
    {
        var flaky = new FlakyEmbeddingClient(failures: 1);
        var cache = new CachingEmbeddingClient(flaky);

        EmbeddingBatch failed = await cache.EmbedAsync(new[] { "candidate" });
        Assert.False(failed.Succeeded);
        Assert.Equal(0, cache.Count);

        EmbeddingBatch recovered = await cache.EmbedAsync(new[] { "candidate" });
        Assert.True(recovered.Succeeded);
        Assert.Equal(1, cache.Count);
        Assert.Equal(2, flaky.Calls);
    }

    [Fact]
    public async Task CachingClientEvictsTheOldestEntryOnceCapacityIsReached()
    {
        var provider = new DeterministicEmbeddingClient(dimensions: 8);
        var cache = new CachingEmbeddingClient(provider, capacity: 2);

        await cache.EmbedAsync(new[] { "one" });
        await cache.EmbedAsync(new[] { "two" });
        await cache.EmbedAsync(new[] { "three" });

        Assert.Equal(2, cache.Count);

        long before = provider.Calls;
        await cache.EmbedAsync(new[] { "one" });
        Assert.Equal(before + 1, provider.Calls);
    }

    [Fact]
    public void CachingClientKeysOnContentSoIdenticalTextSharesOneEntry()
    {
        string key = CachingEmbeddingClient.ComputeKey("def f(): pass");

        Assert.Equal(key, CachingEmbeddingClient.ComputeKey("def f(): pass"));
        Assert.NotEqual(key, CachingEmbeddingClient.ComputeKey("def g(): pass"));
        Assert.Equal(64, key.Length);
    }

    [Fact]
    public void CachingClientRejectsAnInvalidCapacityAndANullInner()
    {
#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() => new CachingEmbeddingClient(null));
#pragma warning restore CS8625
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new CachingEmbeddingClient(new DeterministicEmbeddingClient(), capacity: 0));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4_097)]
    public void DeterministicClientRejectsAnInvalidDimensionCount(int dimensions) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeterministicEmbeddingClient(dimensions));

    [Fact]
    public void HttpEmbeddingClientValidatesItsConfigurationWithoutContactingAnything()
    {
        Assert.Throws<ArgumentException>(() => new OpenAICompatibleEmbeddingClient("   "));
        Assert.Throws<ArgumentException>(() => new OpenAICompatibleEmbeddingClient("key", modelId: " "));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OpenAICompatibleEmbeddingClient("key", maxInputCharacters: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OpenAICompatibleEmbeddingClient("key", maxRetries: 9));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OpenAICompatibleEmbeddingClient("key", initialRetryDelayMilliseconds: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OpenAICompatibleEmbeddingClient("key", timeoutMilliseconds: 0));

        var client = new OpenAICompatibleEmbeddingClient("key");
        Assert.Equal(OpenAICompatibleEmbeddingClient.DefaultEndpoint, client.Endpoint);
        Assert.Equal(OpenAICompatibleEmbeddingClient.DefaultModelId, client.ModelId);
        Assert.Equal(OpenAICompatibleEmbeddingClient.DefaultMaxInputCharacters, client.MaxInputCharacters);

        // Unlike the reference implementation's closed model list, any model name and any compatible endpoint work.
        var custom = new OpenAICompatibleEmbeddingClient(
            "key", "bge-m3", "http://localhost:11434/v1/embeddings");
        Assert.Equal("bge-m3", custom.ModelId);
        Assert.Equal("http://localhost:11434/v1/embeddings", custom.Endpoint);
    }
}
