using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration;

/// <summary>
/// Tests for the end-to-end <see cref="RagPipeline{T}"/> orchestrator: ingest (chunk→embed→store), query
/// (retrieve→rerank→compress→generate), and tenant/namespace isolation.
/// </summary>
public class RagPipelineTests
{
    private const int Dim = 8;

    // Minimal retriever that records the query + filters it was called with and returns a preset doc.
    private sealed class RecordingRetriever : IRetriever<double>
    {
        public string? LastQuery { get; private set; }
        public Dictionary<string, object>? LastFilters { get; private set; }
        public int DefaultTopK => 5;

        public IEnumerable<Document<double>> Retrieve(string query) => Retrieve(query, DefaultTopK, new Dictionary<string, object>());
        public IEnumerable<Document<double>> Retrieve(string query, int topK) => Retrieve(query, topK, new Dictionary<string, object>());
        public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            LastQuery = query;
            LastFilters = metadataFilters;
            return new[] { new Document<double>("d1", "retrieved context") { RelevanceScore = 1.0, HasRelevanceScore = true } };
        }

        public Task<IEnumerable<Document<double>>> RetrieveAsync(string query, int topK, Dictionary<string, object>? metadataFilters = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(Retrieve(query, topK, metadataFilters ?? new Dictionary<string, object>()));
        }
    }

    private static RagPipeline<double> CreatePipeline(RecordingRetriever retriever, string? tenant = null, IChunkingStrategy? chunking = null)
        => new RagPipeline<double>(
            embedding: new StubEmbeddingModel<double>(Dim),
            store: new InMemoryDocumentStore<double>(Dim),
            retriever: retriever,
            chunking: chunking,
            generator: new StubGenerator<double>(),
            tenant: tenant);

    [Fact]
    public async Task IngestAsync_StoresChunkedEmbeddedDocument()
    {
        var store = new InMemoryDocumentStore<double>(Dim);
        var pipeline = new RagPipeline<double>(
            new StubEmbeddingModel<double>(Dim), store, new RecordingRetriever(),
            chunking: new TokenBasedChunkingStrategy(maxTokens: 3, overlapTokens: 0));

        int stored = await pipeline.IngestAsync("doc1", "alpha beta gamma delta epsilon zeta eta theta");

        Assert.True(stored >= 2, "long content should be split into multiple chunks");
        Assert.Equal(stored, store.DocumentCount);
    }

    [Fact]
    public async Task QueryAsync_RunsRetrieveAndGenerate()
    {
        var retriever = new RecordingRetriever();
        var pipeline = CreatePipeline(retriever);

        var result = await pipeline.QueryAsync("what is X?", topK: 3);

        Assert.Equal("what is X?", retriever.LastQuery);
        Assert.Single(result.Contexts);
        Assert.Equal("d1", result.Contexts[0].Id);
        Assert.NotNull(result.Answer); // StubGenerator produced a grounded answer
    }

    [Fact]
    public async Task Tenant_IsStampedOnIngestAndUsedAsRetrievalFilter()
    {
        var retriever = new RecordingRetriever();
        var store = new InMemoryDocumentStore<double>(Dim);
        var pipeline = new RagPipeline<double>(
            new StubEmbeddingModel<double>(Dim), store, retriever, tenant: "acme");

        await pipeline.IngestAsync("doc1", "hello world");
        await pipeline.QueryAsync("q");

        // The retrieval filter carries the tenant so cross-tenant data is isolated.
        Assert.True(retriever.LastFilters!.ContainsKey(RagPipeline<double>.TenantMetadataKey));
        Assert.Equal("acme", retriever.LastFilters![RagPipeline<double>.TenantMetadataKey]);

        // And the stored document was stamped with the tenant.
        var all = store.GetAll().ToList();
        Assert.Single(all);
        Assert.Equal("acme", all[0].Metadata[RagPipeline<double>.TenantMetadataKey]);
    }
}
