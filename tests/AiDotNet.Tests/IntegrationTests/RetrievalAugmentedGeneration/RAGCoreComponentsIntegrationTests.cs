using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Integration tests for RAG core components: VectorSearch metrics, Rerankers,
/// QueryProcessors, and Document model. Uses golden references and edge cases.
/// </summary>
public class RAGCoreComponentsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region VectorSearch Metrics

    [Fact]
    public void CosineSimilarity_IdenticalVectors_ReturnsOne()
    {
        var metric = new CosineSimilarityMetric<double>();
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var similarity = metric.Calculate(v, v);

        Assert.Equal(1.0, similarity, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_ReturnsZero()
    {
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

        var similarity = metric.Calculate(v1, v2);

        Assert.Equal(0.0, similarity, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_OppositeVectors_ReturnsNegativeOne()
    {
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new[] { 1.0, 0.0 });
        var v2 = new Vector<double>(new[] { -1.0, 0.0 });

        var similarity = metric.Calculate(v1, v2);

        Assert.Equal(-1.0, similarity, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_GoldenReference_45Degrees()
    {
        var metric = new CosineSimilarityMetric<double>();
        // cos(45) = 1/sqrt(2) ~ 0.7071
        var v1 = new Vector<double>(new[] { 1.0, 0.0 });
        var v2 = new Vector<double>(new[] { 1.0, 1.0 });

        var similarity = metric.Calculate(v1, v2);

        Assert.Equal(1.0 / Math.Sqrt(2.0), similarity, 1e-4);
    }

    [Fact]
    public void CosineSimilarity_HigherIsBetter_ReturnsTrue()
    {
        var metric = new CosineSimilarityMetric<double>();
        Assert.True(metric.HigherIsBetter);
    }

    [Fact]
    public void CosineSimilarity_ScaleInvariant()
    {
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2Scaled = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        var similarity = metric.Calculate(v1, v2Scaled);

        Assert.Equal(1.0, similarity, 1e-4); // same direction, should be 1.0
    }

    [Fact]
    public void EuclideanDistance_IdenticalVectors_ReturnsZero()
    {
        var metric = new EuclideanDistanceMetric<double>();
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var distance = metric.Calculate(v, v);

        Assert.Equal(0.0, distance, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_GoldenReference_345Triangle()
    {
        var metric = new EuclideanDistanceMetric<double>();
        var v1 = new Vector<double>(new[] { 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0 });

        var distance = metric.Calculate(v1, v2);

        Assert.Equal(5.0, distance, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_HigherIsBetter_ReturnsFalse()
    {
        var metric = new EuclideanDistanceMetric<double>();
        Assert.False(metric.HigherIsBetter);
    }

    [Fact]
    public void EuclideanDistance_IsSymmetric()
    {
        var metric = new EuclideanDistanceMetric<double>();
        var v1 = new Vector<double>(new[] { 1.0, 5.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 2.0, 7.0 });

        var d12 = metric.Calculate(v1, v2);
        var d21 = metric.Calculate(v2, v1);

        Assert.Equal(d12, d21, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_TriangleInequality()
    {
        var metric = new EuclideanDistanceMetric<double>();
        var a = new Vector<double>(new[] { 0.0, 0.0 });
        var b = new Vector<double>(new[] { 3.0, 0.0 });
        var c = new Vector<double>(new[] { 3.0, 4.0 });

        double dAB = metric.Calculate(a, b);
        double dBC = metric.Calculate(b, c);
        double dAC = metric.Calculate(a, c);

        // Triangle inequality: d(A,C) <= d(A,B) + d(B,C)
        Assert.True(dAC <= dAB + dBC + Tolerance);
    }

    #endregion

    #region QueryProcessors

    [Fact]
    public void StopWordRemoval_RemovesCommonWords()
    {
        var processor = new StopWordRemovalQueryProcessor();

        var result = processor.ProcessQuery("What are the main features of the new iPhone?");

        Assert.DoesNotContain("What", result, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(" are ", result, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(" the ", result, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(" of ", result, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("main", result, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("features", result, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("iPhone", result);
    }

    [Fact]
    public void StopWordRemoval_EmptyQuery_ThrowsArgumentException()
    {
        var processor = new StopWordRemovalQueryProcessor();

        Assert.Throws<ArgumentException>(() => processor.ProcessQuery(""));
    }

    [Fact]
    public void StopWordRemoval_AllStopWords_ReturnsEmpty()
    {
        var processor = new StopWordRemovalQueryProcessor();

        var result = processor.ProcessQuery("the and is of");
        Assert.Equal("", result.Trim());
    }

    [Fact]
    public void StopWordRemoval_NoStopWords_PreservesAll()
    {
        var processor = new StopWordRemovalQueryProcessor();

        var result = processor.ProcessQuery("quantum entanglement physics");
        Assert.Equal("quantum entanglement physics", result);
    }

    [Fact]
    public void StopWordRemoval_CustomStopWords()
    {
        var customStopWords = new HashSet<string> { "hello", "world" };
        var processor = new StopWordRemovalQueryProcessor(customStopWords);

        var result = processor.ProcessQuery("hello world programming");
        Assert.Equal("programming", result.Trim());
    }

    [Fact]
    public void StopWordRemoval_PreserveFirstWord()
    {
        var processor = new StopWordRemovalQueryProcessor(preserveFirstWord: true);

        var result = processor.ProcessQuery("What is machine learning?");
        // "What" should be preserved even though it's a stop word
        Assert.StartsWith("What", result);
    }

    [Fact]
    public void KeywordExtraction_ExtractsKeyTerms()
    {
        var processor = new KeywordExtractionQueryProcessor();

        var result = processor.ProcessQuery("Can you tell me about machine learning algorithms?");

        Assert.Contains("machine", result);
        Assert.Contains("learning", result);
        Assert.Contains("algorithms", result);
        Assert.DoesNotContain("can", result);
        Assert.DoesNotContain("you", result);
        Assert.DoesNotContain("tell", result);
    }

    [Fact]
    public void KeywordExtraction_EmptyQuery_ThrowsArgumentException()
    {
        var processor = new KeywordExtractionQueryProcessor();

        Assert.Throws<ArgumentException>(() => processor.ProcessQuery(""));
    }

    [Fact]
    public void KeywordExtraction_MinWordLength_FiltersShortWords()
    {
        var processor = new KeywordExtractionQueryProcessor(minWordLength: 4);

        var result = processor.ProcessQuery("use big neural network model");
        Assert.DoesNotContain("use", result);
        Assert.DoesNotContain("big", result);
        Assert.Contains("neural", result);
        Assert.Contains("network", result);
        Assert.Contains("model", result);
    }

    [Fact]
    public void KeywordExtraction_RemovesPunctuation()
    {
        var processor = new KeywordExtractionQueryProcessor();

        var result = processor.ProcessQuery("What's machine-learning?");

        // Should split on non-word chars and extract keywords
        Assert.Contains("machine", result);
        Assert.Contains("learning", result);
    }

    #endregion

    #region DiversityReranker

    [Fact]
    public void DiversityReranker_PromotesDiverseDocuments()
    {
        var reranker = new DiversityReranker<double>();

        var docs = new List<Document<double>>
        {
            CreateDoc("1", "Python programming language basics guide", 0.9),
            CreateDoc("2", "Python programming language tutorial introduction", 0.85),
            CreateDoc("3", "Machine learning with neural networks deep learning", 0.7),
            CreateDoc("4", "Python programming language getting started", 0.65),
        };

        var reranked = reranker.Rerank("Python programming", docs).ToList();

        Assert.Equal(4, reranked.Count);
        // First should be most relevant
        Assert.Equal("1", reranked[0].Id);
        // Machine learning doc should be promoted due to diversity
        // (it's less similar to the first pick than the other Python docs)
        int mlPosition = reranked.FindIndex(d => d.Id == "3");
        Assert.True(mlPosition < 3, "Diverse ML doc should be promoted above position 3");
    }

    [Fact]
    public void DiversityReranker_SingleDocument_ReturnsSame()
    {
        var reranker = new DiversityReranker<double>();
        var docs = new List<Document<double>>
        {
            CreateDoc("1", "Only document", 0.5),
        };

        var reranked = reranker.Rerank("query", docs).ToList();

        Assert.Single(reranked);
        Assert.Equal("1", reranked[0].Id);
    }

    [Fact]
    public void DiversityReranker_EmptyDocs_ReturnsEmpty()
    {
        var reranker = new DiversityReranker<double>();

        var reranked = reranker.Rerank("query", new List<Document<double>>()).ToList();

        Assert.Empty(reranked);
    }

    [Fact]
    public void DiversityReranker_LambdaOne_PureRelevance()
    {
        var reranker = new DiversityReranker<double>(1.0);

        var docs = new List<Document<double>>
        {
            CreateDoc("1", "same text", 0.5),
            CreateDoc("2", "same text", 0.9),
            CreateDoc("3", "different content entirely", 0.3),
        };

        var reranked = reranker.Rerank("query", docs).ToList();

        // Lambda=1 means pure relevance, highest score first
        Assert.Equal("2", reranked[0].Id);
    }

    [Fact]
    public void DiversityReranker_InvalidLambda_Throws()
    {
        Assert.Throws<ArgumentException>(() => new DiversityReranker<double>(-0.1));
        Assert.Throws<ArgumentException>(() => new DiversityReranker<double>(1.1));
    }

    [Fact]
    public void DiversityReranker_AssignsNewRelevanceScores()
    {
        var reranker = new DiversityReranker<double>();
        var docs = new List<Document<double>>
        {
            CreateDoc("1", "First document about topic A", 0.9),
            CreateDoc("2", "Second document about topic B", 0.8),
        };

        var reranked = reranker.Rerank("query", docs).ToList();

        // All should have relevance scores set
        Assert.True(reranked.All(d => d.HasRelevanceScore));
        // First rank should have highest score
        Assert.True(reranked[0].RelevanceScore >= reranked[1].RelevanceScore);
    }

    #endregion

    #region MaximalMarginalRelevanceReranker

    [Fact]
    public void MMR_PromotesDiverseEmbeddings()
    {
        var docs = new List<Document<double>>
        {
            CreateDocWithEmbedding("1", "Climate temperature effects", 0.9, new[] { 1.0, 0.0, 0.0 }),
            CreateDocWithEmbedding("2", "Climate temperature patterns", 0.85, new[] { 0.95, 0.05, 0.0 }),
            CreateDocWithEmbedding("3", "Ocean acidification effects", 0.7, new[] { 0.3, 0.9, 0.0 }),
            CreateDocWithEmbedding("4", "Renewable energy solutions", 0.6, new[] { 0.0, 0.1, 0.95 }),
        };

        var reranker = new MaximalMarginalRelevanceReranker<double>(
            doc => doc.Embedding ?? new Vector<double>(3),
            lambda: 0.5);

        var reranked = reranker.Rerank("climate", docs).ToList();

        Assert.Equal(4, reranked.Count);
        Assert.Equal("1", reranked[0].Id); // Most relevant

        // Doc 3 (ocean, different embedding) should be promoted over doc 2 (similar to doc 1)
        int oceanPos = reranked.FindIndex(d => d.Id == "3");
        int tempPatternPos = reranked.FindIndex(d => d.Id == "2");
        Assert.True(oceanPos < tempPatternPos,
            "Diverse ocean doc should be ranked above similar temperature doc");
    }

    [Fact]
    public void MMR_LambdaOne_IsRankByRelevance()
    {
        var docs = new List<Document<double>>
        {
            CreateDocWithEmbedding("1", "text", 0.5, new[] { 1.0, 0.0 }),
            CreateDocWithEmbedding("2", "text", 0.9, new[] { 1.0, 0.0 }),
            CreateDocWithEmbedding("3", "text", 0.3, new[] { 0.0, 1.0 }),
        };

        var reranker = new MaximalMarginalRelevanceReranker<double>(
            doc => doc.Embedding ?? new Vector<double>(2),
            lambda: 1.0);

        var reranked = reranker.Rerank("query", docs).ToList();

        // Pure relevance: 0.9, 0.5, 0.3
        Assert.Equal("2", reranked[0].Id);
    }

    [Fact]
    public void MMR_InvalidLambda_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new MaximalMarginalRelevanceReranker<double>(d => new Vector<double>(1), lambda: -0.1));
        Assert.Throws<ArgumentException>(() =>
            new MaximalMarginalRelevanceReranker<double>(d => new Vector<double>(1), lambda: 1.1));
    }

    [Fact]
    public void MMR_NullEmbeddingFunc_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new MaximalMarginalRelevanceReranker<double>(null!, lambda: 0.7));
    }

    [Fact]
    public void MMR_SingleDocument_ReturnsSame()
    {
        var docs = new List<Document<double>>
        {
            CreateDocWithEmbedding("1", "only doc", 0.9, new[] { 1.0, 0.0 }),
        };

        var reranker = new MaximalMarginalRelevanceReranker<double>(
            doc => doc.Embedding ?? new Vector<double>(2), lambda: 0.7);

        var reranked = reranker.Rerank("query", docs).ToList();
        Assert.Single(reranked);
    }

    #endregion

    #region ReciprocalRankFusion

    [Fact]
    public void RRF_GoldenReference_ScoreFormula()
    {
        // RRF score = 1 / (k + rank + 1), with k=60, rank is 0-based
        var rrf = new ReciprocalRankFusion<double>(k: 60);

        var docs = new List<Document<double>>
        {
            CreateDoc("1", "First", 0.0),
            CreateDoc("2", "Second", 0.0),
            CreateDoc("3", "Third", 0.0),
        };

        var reranked = rrf.Rerank("query", docs).ToList();

        // Score for rank 0: 1/(60+0+1) = 1/61
        // Score for rank 1: 1/(60+1+1) = 1/62
        // Score for rank 2: 1/(60+2+1) = 1/63
        Assert.Equal(1.0 / 61.0, reranked[0].RelevanceScore, 1e-8);
        Assert.Equal(1.0 / 62.0, reranked[1].RelevanceScore, 1e-8);
        Assert.Equal(1.0 / 63.0, reranked[2].RelevanceScore, 1e-8);
    }

    [Fact]
    public void RRF_FuseRankings_CombinesMultipleLists()
    {
        var rrf = new ReciprocalRankFusion<double>(k: 60);

        var list1 = new List<Document<double>>
        {
            CreateDoc("A", "Doc A", 0.0),
            CreateDoc("B", "Doc B", 0.0),
            CreateDoc("C", "Doc C", 0.0),
        };
        var list2 = new List<Document<double>>
        {
            CreateDoc("B", "Doc B", 0.0),
            CreateDoc("C", "Doc C", 0.0),
            CreateDoc("D", "Doc D", 0.0),
        };

        var fused = rrf.FuseRankings(new List<List<Document<double>>> { list1, list2 }, topK: 4);

        Assert.Equal(4, fused.Count);
        // B appears at rank 0 in list1 (1/61) and rank 0 in list2 (1/61): 2/61
        // B should have highest fused score
        Assert.Equal("B", fused[0].Id);
    }

    [Fact]
    public void RRF_FuseRankings_DocumentAppearsOnce_GetsPartialScore()
    {
        var rrf = new ReciprocalRankFusion<double>(k: 60);

        var list1 = new List<Document<double>> { CreateDoc("A", "Doc A", 0.0) };
        var list2 = new List<Document<double>> { CreateDoc("B", "Doc B", 0.0) };

        var fused = rrf.FuseRankings(new List<List<Document<double>>> { list1, list2 }, topK: 2);

        Assert.Equal(2, fused.Count);
        // Both should have same score: 1/61
        Assert.Equal(fused[0].RelevanceScore, fused[1].RelevanceScore, Tolerance);
    }

    [Fact]
    public void RRF_InvalidK_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReciprocalRankFusion<double>(k: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReciprocalRankFusion<double>(k: -1));
    }

    [Fact]
    public void RRF_FuseRankings_NullLists_Throws()
    {
        var rrf = new ReciprocalRankFusion<double>();
        Assert.Throws<ArgumentException>(() => rrf.FuseRankings(null!, 5));
        Assert.Throws<ArgumentException>(() =>
            rrf.FuseRankings(new List<List<Document<double>>>(), 5));
    }

    [Fact]
    public void RRF_FuseRankings_InvalidTopK_Throws()
    {
        var rrf = new ReciprocalRankFusion<double>();
        var lists = new List<List<Document<double>>>
        {
            new() { CreateDoc("1", "test", 0.0) }
        };
        Assert.Throws<ArgumentOutOfRangeException>(() => rrf.FuseRankings(lists, topK: 0));
    }

    #endregion

    #region Document Model

    [Fact]
    public void Document_DefaultConstruction_HasExpectedDefaults()
    {
        var doc = new Document<double>();

        Assert.Equal(string.Empty, doc.Id);
        Assert.Equal(string.Empty, doc.Content);
        Assert.NotNull(doc.Metadata);
        Assert.Empty(doc.Metadata);
        Assert.False(doc.HasRelevanceScore);
        Assert.Null(doc.Embedding);
    }

    [Fact]
    public void Document_SetProperties_Persists()
    {
        var doc = new Document<double>
        {
            Id = "doc-001",
            Content = "This is a test document about machine learning.",
            RelevanceScore = 0.95,
            HasRelevanceScore = true,
            Metadata = new Dictionary<string, object>
            {
                ["author"] = "Test Author",
                ["date"] = "2024-01-01"
            }
        };

        Assert.Equal("doc-001", doc.Id);
        Assert.Equal("This is a test document about machine learning.", doc.Content);
        Assert.Equal(0.95, doc.RelevanceScore, Tolerance);
        Assert.True(doc.HasRelevanceScore);
        Assert.Equal("Test Author", doc.Metadata["author"]);
    }

    [Fact]
    public void Document_Embedding_CanBeSet()
    {
        var doc = new Document<double>
        {
            Embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 })
        };

        Assert.NotNull(doc.Embedding);
        Assert.Equal(3, doc.Embedding.Length);
        Assert.Equal(0.1, doc.Embedding[0], Tolerance);
    }

    #endregion

    #region LostInTheMiddleReranker

    [Fact]
    public void LostInTheMiddle_ReordersToAlternateEnds()
    {
        var reranker = new LostInTheMiddleReranker<double>();

        var docs = Enumerable.Range(1, 5).Select(i =>
            CreateDoc(i.ToString(), $"Document {i}", 1.0 - i * 0.1)).ToList();

        var reranked = reranker.Rerank("query", docs).ToList();

        Assert.Equal(5, reranked.Count);
        // The most relevant docs should be at the beginning and end
        // (not buried in the middle, hence the name)
    }

    #endregion

    #region IdentityReranker

    [Fact]
    public void IdentityReranker_ReturnsDocsInOriginalOrder()
    {
        var reranker = new IdentityReranker<double>();

        var docs = new List<Document<double>>
        {
            CreateDoc("1", "First", 0.5),
            CreateDoc("2", "Second", 0.9),
            CreateDoc("3", "Third", 0.1),
        };

        var reranked = reranker.Rerank("query", docs).ToList();

        Assert.Equal("1", reranked[0].Id);
        Assert.Equal("2", reranked[1].Id);
        Assert.Equal("3", reranked[2].Id);
    }

    #endregion

    #region Helpers

    private static Document<double> CreateDoc(string id, string content, double score)
    {
        return new Document<double>
        {
            Id = id,
            Content = content,
            RelevanceScore = score,
            HasRelevanceScore = true
        };
    }

    private static Document<double> CreateDocWithEmbedding(string id, string content, double score, double[] embedding)
    {
        return new Document<double>
        {
            Id = id,
            Content = content,
            RelevanceScore = score,
            HasRelevanceScore = true,
            Embedding = new Vector<double>(embedding)
        };
    }

    #endregion
}
