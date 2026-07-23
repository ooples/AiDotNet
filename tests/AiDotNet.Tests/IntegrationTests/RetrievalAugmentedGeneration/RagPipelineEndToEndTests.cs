using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Exercises the complete local production path without placeholder embeddings:
/// ingest, embed, HNSW store, dense retrieval, ColBERT reranking, and grounded generation.
/// </summary>
public sealed class RagPipelineEndToEndTests
{
    private sealed class CitingGenerator : GeneratorBase<double>
    {
        public CitingGenerator() : base(maxContextTokens: 2048, maxGenerationTokens: 128) { }

        protected override string GenerateCore(string prompt) =>
            "Paris is the capital of France [1].";
    }

    [Fact(Timeout = 60000)]
    public async Task IngestThenQuery_UsesRealEmbeddingsAndReturnsGroundedCitation()
    {
        var embedding = CreateEmbeddingModel();
        var store = new InMemoryDocumentStore<double>(embedding.EmbeddingDimension);
        var retriever = new DenseRetriever<double>(store, embedding);
        var pipeline = new RagPipeline<double>(
            embedding,
            store,
            retriever,
            reranker: new ColbertReranker<double>(embedding),
            generator: new CitingGenerator(),
            tenant: "travel");

        await pipeline.IngestAsync("france", "paris capital france europe");
        await pipeline.IngestAsync("germany", "berlin capital germany europe");
        await pipeline.IngestAsync("unrelated", "quantum particles physics research");

        var result = await pipeline.QueryAsync("capital france", topK: 2);

        Assert.Equal("france", result.Contexts[0].Id);
        Assert.True(result.Contexts[0].HasRelevanceScore);
        Assert.NotNull(result.Answer);
        Assert.Equal("Paris is the capital of France [1].", result.Answer!.Answer);
        Assert.Equal(new[] { "france" }, result.Answer.Citations);
        Assert.Equal(result.Contexts.Select(document => document.Id),
            result.Answer.SourceDocuments.Select(document => document.Id));
    }

    private static StaticWordEmbeddingModel<double> CreateEmbeddingModel()
    {
        var vocabulary = new Dictionary<string, Vector<double>>
        {
            ["paris"] = new(new[] { 1d, 0d, 0d, 0d, 0d, 0d }),
            ["france"] = new(new[] { 0d, 1d, 0d, 0d, 0d, 0d }),
            ["berlin"] = new(new[] { 0d, 0d, 1d, 0d, 0d, 0d }),
            ["germany"] = new(new[] { 0d, 0d, 0d, 1d, 0d, 0d }),
            ["capital"] = new(new[] { 0d, 0d, 0d, 0d, 1d, 0d }),
            ["europe"] = new(new[] { 0d, 0d, 0d, 0d, 0d, 1d }),
            ["quantum"] = new(new[] { -1d, 0d, 0d, 0d, 0d, 0d }),
            ["particles"] = new(new[] { 0d, -1d, 0d, 0d, 0d, 0d }),
            ["physics"] = new(new[] { 0d, 0d, -1d, 0d, 0d, 0d }),
            ["research"] = new(new[] { 0d, 0d, 0d, -1d, 0d, 0d })
        };

        return new StaticWordEmbeddingModel<double>(vocabulary, dimension: 6);
    }
}
