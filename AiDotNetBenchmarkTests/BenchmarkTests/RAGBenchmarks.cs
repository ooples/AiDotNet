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
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class RAGBenchmarks
{
    [Params(100, 500)]
    public int DocumentCount { get; set; }

    [Params(5, 10)]
    public int TopK { get; set; }

    private List<string> _documents = new List<string>();
    private List<string> _queries = new List<string>();

    [GlobalSetup]
    public void Setup()
    {
        _documents = new List<string>();
        _queries = new List<string>();

        // Generate synthetic documents
        for (int i = 0; i < DocumentCount; i++)
        {
            _documents.Add($"Document {i} contains information about topic {i % 10} with details.");
        }

        // Generate queries
        for (int i = 0; i < 20; i++)
        {
            _queries.Add($"Query about topic {i % 10}");
        }
    }

    [Benchmark(Baseline = true)]
    public RAGConfiguration<double> RAG_CreateConfiguration()
    {
        var config = new RAGConfiguration<double>
        {
            DocumentStore = new DocumentStoreConfig
            {
                Type = "memory",
                Parameters = new Dictionary<string, object>
                {
                    { "max_size", 10000 }
                }
            },
            Chunking = new ChunkingConfig
            {
                Strategy = "sentence",
                ChunkSize = 512,
                ChunkOverlap = 50
            },
            Embedding = new EmbeddingConfig
            {
                ModelType = "sentence-transformers",
                EmbeddingDimension = 768
            },
            Retrieval = new RetrievalConfig
            {
                Strategy = "hybrid",
                TopK = TopK
            }
        };
        return config;
    }

    [Benchmark]
    public RAGConfigurationBuilder<double> RAG_BuildConfiguration()
    {
        var builder = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("memory")
            .WithChunking("sentence", chunkSize: 512, chunkOverlap: 50)
            .WithEmbedding("sentence-transformers", embeddingDimension: 768)
            .WithRetrieval("hybrid", TopK);

        return builder;
    }

    [Benchmark]
    public RAGConfiguration<double> RAG_BuildAndCreate()
    {
        return new RAGConfigurationBuilder<double>()
            .WithDocumentStore("memory")
            .WithChunking("sentence", chunkSize: 512, chunkOverlap: 50)
            .WithEmbedding("sentence-transformers", embeddingDimension: 768)
            .WithRetrieval("hybrid", TopK)
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
        // Simulate retrieval results
        var retrievedDocs = new List<string>();
        for (int i = 0; i < TopK; i++)
        {
            retrievedDocs.Add(_documents[i]);
        }

        var relevantDocs = new List<string> { _documents[0], _documents[1] };

        // Calculate accuracy
        int correct = 0;
        foreach (var doc in retrievedDocs)
        {
            if (relevantDocs.Contains(doc))
            {
                correct++;
            }
        }

        return (double)correct / TopK;
    }

    [Benchmark]
    public ChunkingConfig RAG_CreateChunkingConfig()
    {
        return new ChunkingConfig
        {
            Strategy = "sentence",
            ChunkSize = 512,
            ChunkOverlap = 100
        };
    }

    [Benchmark]
    public EmbeddingConfig RAG_CreateEmbeddingConfig()
    {
        return new EmbeddingConfig
        {
            ModelType = "sentence-transformers",
            ModelPath = "all-MiniLM-L6-v2",
            EmbeddingDimension = 384
        };
    }

    [Benchmark]
    public RetrievalConfig RAG_CreateRetrievalConfig()
    {
        return new RetrievalConfig
        {
            Strategy = "hybrid",
            TopK = TopK,
            Parameters = new Dictionary<string, object>
            {
                { "use_reranker", true },
                { "reranker_model", "cross-encoder" }
            }
        };
    }
}
