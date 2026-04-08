using AiDotNet.RetrievalAugmentedGeneration.Configuration;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Deep integration tests for RAG:
/// RAGConfiguration defaults, ChunkingConfig, RetrievalConfig, EmbeddingConfig,
/// Document model, VectorDocument model,
/// RAG math (chunking overlap, cosine similarity, BM25, NDCG, Reciprocal Rank Fusion).
/// </summary>
public class RAGDeepMathIntegrationTests
{
    // ============================
    // RAGConfiguration: Defaults
    // ============================

    [Fact]
    public void RAGConfiguration_Defaults()
    {
        var config = new RAGConfiguration<double>();
        Assert.NotNull(config.DocumentStore);
        Assert.NotNull(config.Chunking);
        Assert.NotNull(config.Embedding);
        Assert.NotNull(config.Retrieval);
        Assert.NotNull(config.Reranking);
        Assert.NotNull(config.QueryExpansion);
        Assert.NotNull(config.ContextCompression);
    }

    // ============================
    // ChunkingConfig: Defaults
    // ============================

    [Fact]
    public void ChunkingConfig_Defaults()
    {
        var config = new ChunkingConfig();
        Assert.Equal(string.Empty, config.Strategy);
        Assert.Equal(1000, config.ChunkSize);
        Assert.Equal(200, config.ChunkOverlap);
        Assert.NotNull(config.Parameters);
        Assert.Empty(config.Parameters);
    }

    [Fact]
    public void ChunkingConfig_OverlapLessThanChunkSize()
    {
        var config = new ChunkingConfig();
        Assert.True(config.ChunkOverlap < config.ChunkSize,
            $"Overlap ({config.ChunkOverlap}) must be less than chunk size ({config.ChunkSize})");
    }

    // ============================
    // RetrievalConfig: Defaults
    // ============================

    [Fact]
    public void RetrievalConfig_Defaults()
    {
        var config = new RetrievalConfig();
        Assert.Equal(string.Empty, config.Strategy);
        Assert.Equal(10, config.TopK);
        Assert.NotNull(config.Parameters);
        Assert.Empty(config.Parameters);
    }

    // ============================
    // Document: Construction
    // ============================

    [Fact]
    public void Document_DefaultConstructor()
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
    public void Document_ConstructorWithIdAndContent()
    {
        var doc = new Document<double>("doc-1", "This is a test document.");
        Assert.Equal("doc-1", doc.Id);
        Assert.Equal("This is a test document.", doc.Content);
    }

    [Fact]
    public void Document_ConstructorWithMetadata()
    {
        var metadata = new Dictionary<string, object> { ["source"] = "web", ["date"] = "2025-01-01" };
        var doc = new Document<double>("doc-1", "Test content", metadata);
        Assert.Equal(2, doc.Metadata.Count);
        Assert.Equal("web", doc.Metadata["source"]);
    }

    [Fact]
    public void Document_RelevanceScore_DefaultIsZero()
    {
        var doc = new Document<double>();
        Assert.Equal(0.0, doc.RelevanceScore);
    }

    // ============================
    // VectorDocument: Construction
    // ============================

    [Fact]
    public void VectorDocument_DefaultConstructor()
    {
        var vDoc = new VectorDocument<double>();
        Assert.NotNull(vDoc.Document);
        Assert.NotNull(vDoc.Embedding);
    }

    // ============================
    // RAG Math: Chunking
    // ============================

    [Theory]
    [InlineData(5000, 1000, 200, 6)]   // 5000 chars, chunk 1000, overlap 200: ceil((5000-200)/(1000-200))=6
    [InlineData(3000, 500, 100, 8)]    // 3000 chars, chunk 500, overlap 100: ceil((3000-100)/(500-100))=8
    [InlineData(1000, 1000, 200, 1)]   // Exactly one chunk
    [InlineData(500, 1000, 200, 1)]    // Smaller than chunk size: 1 chunk
    public void RAGMath_ChunkCount(int textLength, int chunkSize, int overlap, int expectedChunks)
    {
        int effectiveStep = chunkSize - overlap;
        int chunks;
        if (textLength <= chunkSize)
        {
            chunks = 1;
        }
        else
        {
            chunks = (int)Math.Ceiling((double)(textLength - overlap) / effectiveStep);
        }

        Assert.Equal(expectedChunks, chunks);
    }

    [Theory]
    [InlineData(1000, 200, 800)]   // Effective step = 1000 - 200 = 800
    [InlineData(500, 100, 400)]    // Effective step = 500 - 100 = 400
    [InlineData(512, 0, 512)]      // No overlap: full step
    public void RAGMath_EffectiveChunkStep(int chunkSize, int overlap, int expectedStep)
    {
        int step = chunkSize - overlap;
        Assert.Equal(expectedStep, step);
    }

    // ============================
    // RAG Math: Cosine Similarity
    // ============================

    [Theory]
    [InlineData(new double[] { 1, 0, 0 }, new double[] { 1, 0, 0 }, 1.0)]       // Identical
    [InlineData(new double[] { 1, 0, 0 }, new double[] { 0, 1, 0 }, 0.0)]       // Orthogonal
    [InlineData(new double[] { 1, 0, 0 }, new double[] { -1, 0, 0 }, -1.0)]     // Opposite
    [InlineData(new double[] { 1, 1 }, new double[] { 1, 1 }, 1.0)]             // Same direction
    public void RAGMath_CosineSimilarity(double[] a, double[] b, double expected)
    {
        double dotProduct = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double similarity = dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.Equal(expected, similarity, 1e-10);
    }

    [Fact]
    public void RAGMath_CosineSimilarity_RangeIsMinusOneToOne()
    {
        // Random vectors should have cosine similarity in [-1, 1]
        double[] a = { 0.5, -0.3, 0.8, 0.1, -0.6 };
        double[] b = { -0.2, 0.7, 0.1, -0.4, 0.3 };

        double dotProduct = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double similarity = dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(similarity >= -1.0 - 1e-10 && similarity <= 1.0 + 1e-10,
            $"Cosine similarity {similarity} should be in [-1, 1]");
    }

    // ============================
    // RAG Math: BM25 Scoring
    // ============================

    [Theory]
    [InlineData(3, 100, 10, 5, 1.2, 0.75, 50)]   // tf=3, N=100, df=10, docLen=50
    [InlineData(1, 1000, 100, 5, 1.2, 0.75, 300)]  // tf=1, large corpus
    public void RAGMath_BM25_Score(int tf, int N, int df, int docLen, double k1, double b, double avgDl)
    {
        // BM25: score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        double idf = Math.Log((N - df + 0.5) / (df + 0.5) + 1);
        double tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * docLen / avgDl));
        double score = idf * tfNorm;

        Assert.True(score > 0, "BM25 score should be positive for matching terms");
        Assert.True(idf > 0, "IDF should be positive for terms not in all documents");
    }

    [Fact]
    public void RAGMath_BM25_HigherTF_HigherScore()
    {
        int N = 100, df = 10, docLen = 50;
        double k1 = 1.2, b = 0.75, avgDl = 50;

        double idf = Math.Log((N - df + 0.5) / (df + 0.5) + 1);

        double score1 = idf * (1 * (k1 + 1)) / (1 + k1 * (1 - b + b * docLen / avgDl));
        double score5 = idf * (5 * (k1 + 1)) / (5 + k1 * (1 - b + b * docLen / avgDl));

        Assert.True(score5 > score1, "Higher TF should give higher BM25 score");
    }

    // ============================
    // RAG Math: NDCG (Normalized Discounted Cumulative Gain)
    // ============================

    [Theory]
    [InlineData(new double[] { 3, 2, 3, 0, 1, 2 })]
    public void RAGMath_DCG_Calculation(double[] relevanceScores)
    {
        // DCG = sum(rel_i / log2(i + 1)) for i = 0..n-1
        double dcg = 0;
        for (int i = 0; i < relevanceScores.Length; i++)
        {
            dcg += relevanceScores[i] / (Math.Log(i + 2) / Math.Log(2)); // log2(rank + 1) where rank starts at 1
        }

        Assert.True(dcg > 0, "DCG should be positive for non-zero relevance");

        // NDCG = DCG / IDCG (ideal DCG from sorted relevance)
        var sorted = relevanceScores.OrderByDescending(x => x).ToArray();
        double idcg = 0;
        for (int i = 0; i < sorted.Length; i++)
        {
            idcg += sorted[i] / (Math.Log(i + 2) / Math.Log(2));
        }

        double ndcg = dcg / idcg;
        Assert.True(ndcg >= 0.0 && ndcg <= 1.0 + 1e-10,
            $"NDCG ({ndcg}) should be in [0, 1]");
    }

    [Fact]
    public void RAGMath_NDCG_PerfectRanking_IsOne()
    {
        // If results are already in ideal order, NDCG = 1.0
        double[] relevance = { 3, 2, 1, 0 }; // Already sorted

        double dcg = 0, idcg = 0;
        for (int i = 0; i < relevance.Length; i++)
        {
            double discount = (Math.Log(i + 2) / Math.Log(2));
            dcg += relevance[i] / discount;
            idcg += relevance[i] / discount; // Same order
        }

        double ndcg = dcg / idcg;
        Assert.Equal(1.0, ndcg, 1e-10);
    }

    // ============================
    // RAG Math: Reciprocal Rank Fusion (RRF)
    // ============================

    [Fact]
    public void RAGMath_ReciprocalRankFusion()
    {
        // RRF score for document d: sum(1 / (k + rank_i(d))) across all rankers
        int k = 60; // Standard RRF constant

        // Document appears at rank 1 in ranker A, rank 5 in ranker B
        double rrfScore = 1.0 / (k + 1) + 1.0 / (k + 5);

        Assert.True(rrfScore > 0, "RRF score should be positive");
        Assert.True(rrfScore < 2.0 / k, "RRF score should be bounded by 2/k for 2 rankers");
    }

    [Fact]
    public void RAGMath_RRF_HigherRank_HigherScore()
    {
        int k = 60;

        // Doc A: rank 1 in both rankers
        double scoreA = 1.0 / (k + 1) + 1.0 / (k + 1);

        // Doc B: rank 10 in both rankers
        double scoreB = 1.0 / (k + 10) + 1.0 / (k + 10);

        Assert.True(scoreA > scoreB, "Higher-ranked document should have higher RRF score");
    }

    // ============================
    // RAG Math: Mean Reciprocal Rank (MRR)
    // ============================

    [Theory]
    [InlineData(new int[] { 1, 3, 2, 1, 5 }, 0.6067)]   // MRR = (1/1 + 1/3 + 1/2 + 1/1 + 1/5) / 5
    public void RAGMath_MRR_Calculation(int[] firstRelevantRanks, double expectedMRR)
    {
        double sumReciprocalRanks = 0;
        foreach (int rank in firstRelevantRanks)
        {
            sumReciprocalRanks += 1.0 / rank;
        }
        double mrr = sumReciprocalRanks / firstRelevantRanks.Length;

        Assert.Equal(expectedMRR, mrr, 1e-3);
    }

    // ============================
    // RAG Math: Precision and Recall at K
    // ============================

    [Theory]
    [InlineData(3, 5, 0.6)]     // 3 relevant out of 5 retrieved: P@5 = 0.6
    [InlineData(5, 10, 0.5)]    // 5 relevant out of 10 retrieved: P@10 = 0.5
    [InlineData(1, 1, 1.0)]     // 1 relevant out of 1 retrieved: P@1 = 1.0
    public void RAGMath_PrecisionAtK(int relevantRetrieved, int k, double expectedPrecision)
    {
        double precision = (double)relevantRetrieved / k;
        Assert.Equal(expectedPrecision, precision, 1e-10);
    }

    [Theory]
    [InlineData(3, 10, 0.3)]    // 3 relevant retrieved out of 10 total relevant: R@K = 0.3
    [InlineData(10, 10, 1.0)]   // All relevant documents retrieved
    public void RAGMath_RecallAtK(int relevantRetrieved, int totalRelevant, double expectedRecall)
    {
        double recall = (double)relevantRetrieved / totalRelevant;
        Assert.Equal(expectedRecall, recall, 1e-10);
    }

    // ============================
    // RAG Math: Context Window Usage
    // ============================

    [Theory]
    [InlineData(4096, 512, 200, 17)]    // (4096 - 512) / 200 = 17 chunks max
    [InlineData(8192, 1024, 500, 14)]   // (8192 - 1024) / 500 = 14 chunks max
    public void RAGMath_MaxChunksInContextWindow(int contextSize, int reservedForQuery, int avgChunkTokens, int expectedMaxChunks)
    {
        int availableTokens = contextSize - reservedForQuery;
        int maxChunks = availableTokens / avgChunkTokens;
        Assert.Equal(expectedMaxChunks, maxChunks);
    }
}
