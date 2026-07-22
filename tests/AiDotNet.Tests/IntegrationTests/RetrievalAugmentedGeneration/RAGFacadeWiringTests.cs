using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Configuration;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Asserts that RAG components wired through the <see cref="AiModelBuilder{T, TInput, TOutput}"/> facade
/// are surfaced on the built <c>AiModelResult</c>. Covers tasks #29 (chunking/compression parameters and
/// the documentStore-dropped bug fix), #30 (ConfigureVectorStore / ConfigureVectorIndex), and #32
/// (ConfigureRAG materialization of the previously orphaned RAGConfiguration).
/// </summary>
public class RAGFacadeWiringTests
{
    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 40, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.15) + (i * 0.01);
            y[i] = Math.Sin((i + cols) * 0.15) + (i * 0.01);
        }

        return (x, y);
    }

    private static IAiModelBuilder<double, Matrix<double>, Vector<double>> NewBuilder()
    {
        var (x, y) = BuildData();
        return new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y));
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureRAG_ChunkingAndCompression_AreSurfacedOnResult()
    {
        var chunking = new FixedSizeChunkingStrategy(chunkSize: 256, chunkOverlap: 32);
        var compressor = new AutoCompressor<double>(maxOutputLength: 200, compressionRatio: 0.5);

        var result = await NewBuilder()
            .ConfigureRetrievalAugmentedGeneration(chunkingStrategy: chunking, contextCompressor: compressor)
            .BuildAsync();

        Assert.Same(chunking, result.ChunkingStrategy);
        Assert.Same(compressor, result.ContextCompressor);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureRAG_DocumentStoreWithoutKnowledgeGraph_BuildsRetriever()
    {
        // The documentStore-dropped bug: a store supplied without a knowledge graph used to be silently
        // discarded. Now it must both surface as DocumentStore and drive a default vector retriever.
        var store = new InMemoryDocumentStore<double>(vectorDimension: 16);

        var result = await NewBuilder()
            .ConfigureRetrievalAugmentedGeneration(documentStore: store)
            .BuildAsync();

        Assert.Same(store, result.DocumentStore);
        Assert.NotNull(result.RagRetriever);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureVectorStore_Alone_YieldsNonNullRetriever()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 16);

        var result = await NewBuilder()
            .ConfigureVectorStore(store)
            .BuildAsync();

        Assert.NotNull(result.RagRetriever);
        Assert.Same(store, result.DocumentStore);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureVectorStore_HonorsConfiguredEmbeddingModel()
    {
        var embedder = new StubEmbeddingModel<double>(embeddingDimension: 16);
        var store = new InMemoryDocumentStore<double>(vectorDimension: 16);

        var result = await NewBuilder()
            .ConfigureEmbeddingModel(embedder)
            .ConfigureVectorStore(store)
            .BuildAsync();

        Assert.NotNull(result.RagRetriever);
        Assert.Same(embedder, result.EmbeddingModel);
    }

    [Theory(Timeout = 120000)]
    [InlineData(VectorIndexKind.Flat)]
    [InlineData(VectorIndexKind.HNSW)]
    [InlineData(VectorIndexKind.IVF)]
    [InlineData(VectorIndexKind.LSH)]
    public async Task ConfigureVectorIndex_BuildsRetrieverAndStore(VectorIndexKind kind)
    {
        var result = await NewBuilder()
            .ConfigureVectorIndex(kind, vectorDimension: 16, metric: new CosineSimilarityMetric<double>())
            .BuildAsync();

        Assert.NotNull(result.RagRetriever);
        Assert.NotNull(result.DocumentStore);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureRAG_MaterializesConfigurationIntoComponents()
    {
        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize", chunkSize: 512, chunkOverlap: 64)
            .WithEmbedding("Stub", embeddingDimension: 32)
            .WithRetrieval("Dense", topK: 7)
            .WithReranking("Diversity", topK: 3)
            .WithContextCompression("Auto", compressionRatio: 0.4, maxLength: 300)
            .Build();

        var result = await NewBuilder()
            .ConfigureRAG(config)
            .BuildAsync();

        Assert.NotNull(result.ChunkingStrategy);
        Assert.NotNull(result.EmbeddingModel);
        Assert.NotNull(result.RagRetriever);
        Assert.NotNull(result.DocumentStore);
        Assert.NotNull(result.RagReranker);
        Assert.NotNull(result.ContextCompressor);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureRAG_WithDisabledOptionalStages_LeavesThemNull()
    {
        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("Recursive")
            .WithEmbedding("Stub", embeddingDimension: 24)
            .WithRetrieval("Dense")
            .Build();

        var result = await NewBuilder()
            .ConfigureRAG(config)
            .BuildAsync();

        Assert.NotNull(result.ChunkingStrategy);
        Assert.NotNull(result.RagRetriever);
        // Reranking and compression were not enabled on the config.
        Assert.Null(result.RagReranker);
        Assert.Null(result.ContextCompressor);
    }
}
