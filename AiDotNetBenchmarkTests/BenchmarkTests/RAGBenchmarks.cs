using AiDotNet.RetrievalAugmentedGeneration.Configuration;
using AiDotNet.RetrievalAugmentedGeneration.Evaluation;
using AiDotNet.Agents;
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

    private List<string> _documents = null!;
    private List<string> _queries = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
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
    public RAGConfiguration RAG_CreateConfiguration()
    {
        var config = new RAGConfiguration
        {
            RetrievalTopK = TopK,
            ChunkSize = 512,
            ChunkOverlap = 50,
            EmbeddingModel = "sentence-transformers",
            RerankerModel = "cross-encoder",
            UseHybridSearch = true,
            IncludeMetadata = true
        };
        return config;
    }

    [Benchmark]
    public RAGConfigurationBuilder RAG_BuildConfiguration()
    {
        var builder = new RAGConfigurationBuilder()
            .WithRetrievalTopK(TopK)
            .WithChunkSize(512)
            .WithChunkOverlap(50)
            .WithEmbeddingModel("sentence-transformers")
            .WithHybridSearch(true);

        return builder;
    }

    [Benchmark]
    public RAGConfiguration RAG_BuildAndCreate()
    {
        return new RAGConfigurationBuilder()
            .WithRetrievalTopK(TopK)
            .WithChunkSize(512)
            .WithChunkOverlap(50)
            .WithEmbeddingModel("sentence-transformers")
            .WithHybridSearch(true)
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
}
