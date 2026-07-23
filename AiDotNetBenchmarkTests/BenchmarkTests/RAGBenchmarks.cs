using AiDotNet.RetrievalAugmentedGeneration.Configuration;
using AiDotNet.RetrievalAugmentedGeneration.Evaluation;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for RAG (Retrieval Augmented Generation) functionality
/// Tests RAG configuration, evaluation, and retrieval performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class RagBenchmarks
{
    // String constants to avoid repetition (SonarCloud S1192)
    private const string ChunkingStrategySentence = "sentence";
    private const string EmbeddingModelSentenceTransformers = "sentence-transformers";
    private const string RetrievalStrategyHybrid = "hybrid";
    private const string DocumentStoreTypeMemory = "memory";

    // BenchmarkDotNet uses Params arrays via reflection - not unused (CodeQL cs/unused-collection false positive)
    [Params(100, 500)]
    public int DocumentCount { get; set; }

    // BenchmarkDotNet uses Params arrays via reflection - not unused (CodeQL cs/unused-collection false positive)
    [Params(5, 10)]
    public int TopK { get; set; }

    private List<string> _documents = new List<string>();

    [GlobalSetup]
    public void Setup()
    {
        _documents = new List<string>();

        // Generate synthetic documents
        for (int i = 0; i < DocumentCount; i++)
        {
            _documents.Add($"Document {i} contains information about topic {i % 10} with details.");
        }
    }

    [Benchmark(Baseline = true)]
    public RAGConfiguration<double> RAG_CreateConfiguration()
    {
        var config = new RAGConfiguration<double>
        {
            DocumentStore = new DocumentStoreConfig
            {
                Type = DocumentStoreTypeMemory,
                Parameters = new Dictionary<string, object>
                {
                    { "max_size", 10000 }
                }
            },
            Chunking = new ChunkingConfig
            {
                Strategy = ChunkingStrategySentence,
                ChunkSize = 512,
                ChunkOverlap = 50
            },
            Embedding = new EmbeddingConfig
            {
                ModelType = EmbeddingModelSentenceTransformers,
                EmbeddingDimension = 768
            },
            Retrieval = new RetrievalConfig
            {
                Strategy = RetrievalStrategyHybrid,
                TopK = TopK
            }
        };
        return config;
    }

    [Benchmark]
    public RAGConfigurationBuilder<double> RAG_BuildConfiguration()
    {
        var builder = new RAGConfigurationBuilder<double>()
            .WithDocumentStore(DocumentStoreTypeMemory)
            .WithChunking(ChunkingStrategySentence, chunkSize: 512, chunkOverlap: 50)
            .WithEmbedding(EmbeddingModelSentenceTransformers, embeddingDimension: 768)
            .WithRetrieval(RetrievalStrategyHybrid, TopK);

        return builder;
    }

    [Benchmark]
    public RAGConfiguration<double> RAG_BuildAndCreate()
    {
        return new RAGConfigurationBuilder<double>()
            .WithDocumentStore(DocumentStoreTypeMemory)
            .WithChunking(ChunkingStrategySentence, chunkSize: 512, chunkOverlap: 50)
            .WithEmbedding(EmbeddingModelSentenceTransformers, embeddingDimension: 768)
            .WithRetrieval(RetrievalStrategyHybrid, TopK)
            .Build();
    }

    [Benchmark]
    public RAGEvaluator<double> RAG_CreateEvaluator()
    {
        var evaluator = new RAGEvaluator<double>();
        return evaluator;
    }

    [Benchmark]
    public double RAG_CalculateRetrievalAccuracy()
    {
        // Simulate retrieval results - use _documents to ensure container is accessed
        var retrievedDocs = _documents.Take(TopK).ToList();

        var relevantDocs = new HashSet<string> { _documents[0], _documents[1] };

        // Calculate accuracy using Count with predicate (SonarCloud S2971)
        int correct = retrievedDocs.Count(doc => relevantDocs.Contains(doc));

        return (double)correct / TopK;
    }

    [Benchmark]
    public ChunkingConfig RAG_CreateChunkingConfig()
    {
        return new ChunkingConfig
        {
            Strategy = ChunkingStrategySentence,
            ChunkSize = 512,
            ChunkOverlap = 100
        };
    }

    [Benchmark]
    public EmbeddingConfig RAG_CreateEmbeddingConfig()
    {
        return new EmbeddingConfig
        {
            ModelType = EmbeddingModelSentenceTransformers,
            ModelPath = "all-MiniLM-L6-v2",
            EmbeddingDimension = 384
        };
    }

    [Benchmark]
    public RetrievalConfig RAG_CreateRetrievalConfig()
    {
        return new RetrievalConfig
        {
            Strategy = RetrievalStrategyHybrid,
            TopK = TopK,
            Parameters = new Dictionary<string, object>
            {
                { "use_reranker", true },
                { "reranker_model", "cross-encoder" }
            }
        };
    }
}
