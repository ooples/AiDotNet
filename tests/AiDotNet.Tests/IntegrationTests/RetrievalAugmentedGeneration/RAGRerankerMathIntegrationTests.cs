using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.RerankingStrategies;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Deep mathematical correctness tests for RAG rerankers.
/// Each test verifies the reranker's algorithm against hand-computed golden references.
/// </summary>
public class RAGRerankerMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Reciprocal Rank Fusion - Golden Reference Tests

    [Fact]
    public void RRF_SingleList_GoldenReference_ScoresCorrect()
    {
        // Given 3 documents in a single list with k=60:
        // Rank 0: score = 1/(60+0+1) = 1/61
        // Rank 1: score = 1/(60+1+1) = 1/62
        // Rank 2: score = 1/(60+2+1) = 1/63
        var rrf = new ReciprocalRankFusion<double>(k: 60);

        var docs = new List<Document<double>>
        {
            new("doc1", "first document") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "second document") { RelevanceScore = 0.7, HasRelevanceScore = true },
            new("doc3", "third document") { RelevanceScore = 0.5, HasRelevanceScore = true },
        };

        var result = rrf.Rerank("query", docs).ToList();

        // Verify RRF scores match golden reference exactly
        double expectedScore0 = 1.0 / 61.0;  // rank 0
        double expectedScore1 = 1.0 / 62.0;  // rank 1
        double expectedScore2 = 1.0 / 63.0;  // rank 2

        // The first doc should have the highest RRF score
        Assert.Equal(expectedScore0, Convert.ToDouble(result[0].RelevanceScore), Tolerance);
        Assert.Equal(expectedScore1, Convert.ToDouble(result[1].RelevanceScore), Tolerance);
        Assert.Equal(expectedScore2, Convert.ToDouble(result[2].RelevanceScore), Tolerance);
    }

    [Fact]
    public void RRF_FuseRankings_TwoLists_DocumentInBoth_GoldenReference()
    {
        // Doc "shared" appears in both lists:
        // List 1: shared at rank 0, other1 at rank 1
        // List 2: other2 at rank 0, shared at rank 1
        //
        // With k=60:
        // shared: 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62
        // other1: 1/(60+1+1) = 1/62 (only in list 1 at rank 1)
        // other2: 1/(60+0+1) = 1/61 (only in list 2 at rank 0)
        var rrf = new ReciprocalRankFusion<double>(k: 60);

        var list1 = new List<Document<double>>
        {
            new("shared", "shared doc") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("other1", "other doc 1") { RelevanceScore = 0.5, HasRelevanceScore = true },
        };

        var list2 = new List<Document<double>>
        {
            new("other2", "other doc 2") { RelevanceScore = 0.8, HasRelevanceScore = true },
            new("shared", "shared doc") { RelevanceScore = 0.6, HasRelevanceScore = true },
        };

        var rankings = new List<List<Document<double>>> { list1, list2 };
        var result = rrf.FuseRankings(rankings, topK: 3);

        double expectedSharedScore = 1.0 / 61.0 + 1.0 / 62.0;
        double expectedOther1Score = 1.0 / 62.0;
        double expectedOther2Score = 1.0 / 61.0;

        // Shared doc should be first (highest combined RRF score)
        var sharedDoc = result.First(d => d.Id == "shared");
        Assert.Equal(expectedSharedScore, Convert.ToDouble(sharedDoc.RelevanceScore), Tolerance);

        // other2 should be second (appears at rank 0 in list 2)
        var other2Doc = result.First(d => d.Id == "other2");
        Assert.Equal(expectedOther2Score, Convert.ToDouble(other2Doc.RelevanceScore), Tolerance);

        // other1 should be third (appears at rank 1 in list 1)
        var other1Doc = result.First(d => d.Id == "other1");
        Assert.Equal(expectedOther1Score, Convert.ToDouble(other1Doc.RelevanceScore), Tolerance);

        // Verify ordering: shared > other2 > other1
        Assert.Equal("shared", result[0].Id);
        Assert.Equal("other2", result[1].Id);
        Assert.Equal("other1", result[2].Id);
    }

    [Fact]
    public void RRF_FuseRankings_ThreeLists_GoldenReference()
    {
        // Three lists with k=60:
        // List 1: [A(r0), B(r1), C(r2)]
        // List 2: [B(r0), C(r1), A(r2)]
        // List 3: [C(r0), A(r1), B(r2)]
        //
        // A: 1/61 + 1/63 + 1/62 = all three contribute
        // B: 1/62 + 1/61 + 1/63 = same sum (symmetric)
        // C: 1/63 + 1/62 + 1/61 = same sum (symmetric)
        // All three docs should have the SAME total RRF score!
        var rrf = new ReciprocalRankFusion<double>(k: 60);

        var list1 = new List<Document<double>>
        {
            new("A", "doc A"),
            new("B", "doc B"),
            new("C", "doc C"),
        };

        var list2 = new List<Document<double>>
        {
            new("B", "doc B"),
            new("C", "doc C"),
            new("A", "doc A"),
        };

        var list3 = new List<Document<double>>
        {
            new("C", "doc C"),
            new("A", "doc A"),
            new("B", "doc B"),
        };

        var rankings = new List<List<Document<double>>> { list1, list2, list3 };
        var result = rrf.FuseRankings(rankings, topK: 3);

        double expectedScore = 1.0 / 61.0 + 1.0 / 62.0 + 1.0 / 63.0;

        // All three should have the same score
        foreach (var doc in result)
        {
            Assert.Equal(expectedScore, Convert.ToDouble(doc.RelevanceScore), Tolerance);
        }
    }

    [Fact]
    public void RRF_CustomK_GoldenReference()
    {
        // With k=1 (small k emphasizes top ranks more):
        // Rank 0: 1/(1+0+1) = 1/2 = 0.5
        // Rank 1: 1/(1+1+1) = 1/3 ≈ 0.333
        // Difference = 0.167 (large gap)
        //
        // With k=1000 (large k reduces rank emphasis):
        // Rank 0: 1/(1000+0+1) = 1/1001
        // Rank 1: 1/(1000+1+1) = 1/1002
        // Difference is tiny
        var rrfSmallK = new ReciprocalRankFusion<double>(k: 1);

        var docs = new List<Document<double>>
        {
            new("doc1", "first") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "second") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = rrfSmallK.Rerank("query", docs).ToList();

        Assert.Equal(1.0 / 2.0, Convert.ToDouble(result[0].RelevanceScore), Tolerance);
        Assert.Equal(1.0 / 3.0, Convert.ToDouble(result[1].RelevanceScore), Tolerance);
    }

    [Fact]
    public void RRF_InvalidK_ThrowsArgumentOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReciprocalRankFusion<double>(k: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReciprocalRankFusion<double>(k: -1));
    }

    [Fact]
    public void RRF_FuseRankings_NullOrEmpty_ThrowsArgumentException()
    {
        var rrf = new ReciprocalRankFusion<double>();

        Assert.Throws<ArgumentException>(() =>
            rrf.FuseRankings(new List<List<Document<double>>>(), topK: 1));
    }

    [Fact]
    public void RRF_FuseRankings_TopKLimitsOutput()
    {
        var rrf = new ReciprocalRankFusion<double>(k: 60);

        var list1 = new List<Document<double>>
        {
            new("A", "doc A"),
            new("B", "doc B"),
            new("C", "doc C"),
            new("D", "doc D"),
            new("E", "doc E"),
        };

        var rankings = new List<List<Document<double>>> { list1 };
        var result = rrf.FuseRankings(rankings, topK: 2);

        Assert.Equal(2, result.Count);
        // First two should be ranked highest
        Assert.Equal("A", result[0].Id);
        Assert.Equal("B", result[1].Id);
    }

    [Fact]
    public void RRF_FuseRankings_InvalidTopK_Throws()
    {
        var rrf = new ReciprocalRankFusion<double>(k: 60);
        var rankings = new List<List<Document<double>>>
        {
            new() { new("A", "doc") }
        };

        Assert.Throws<ArgumentOutOfRangeException>(() => rrf.FuseRankings(rankings, topK: 0));
    }

    #endregion

    #region DiversityReranker - Jaccard Similarity Golden References

    [Fact]
    public void DiversityReranker_JaccardSimilarity_GoldenReference()
    {
        // Text1: "the cat sat on the mat" → words: {the, cat, sat, on, mat} (5 unique)
        // Text2: "the cat sat on a hat"   → words: {the, cat, sat, on, a, hat} (6 unique)
        // Intersection: {the, cat, sat, on} = 4
        // Union: {the, cat, sat, on, mat, a, hat} = 7
        // Jaccard = 4/7 ≈ 0.5714
        //
        // We verify this indirectly by observing the reranker's behavior
        var lambda = 0.5;
        var reranker = new DiversityReranker<double>(lambda);

        var docs = new List<Document<double>>
        {
            new("doc1", "the cat sat on the mat") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "the cat sat on a hat") { RelevanceScore = 0.85, HasRelevanceScore = true },
            new("doc3", "dogs run in the park daily") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("query", docs).ToList();

        // doc1 should be first (highest relevance)
        Assert.Equal("doc1", result[0].Id);

        // doc3 should be second (lower relevance but very different from doc1)
        // doc2 is very similar to doc1 (Jaccard ≈ 0.57) so diversity penalty hurts it
        // doc3 shares only "the" with doc1: intersection={the}=1, union≈10, Jaccard≈0.1
        // doc2 score: 0.5*0.85 - 0.5*0.5714 ≈ 0.425 - 0.286 = 0.139
        // doc3 score: 0.5*0.7  - 0.5*0.1    ≈ 0.35  - 0.05  = 0.30
        // doc3 > doc2, so doc3 should be second
        Assert.Equal("doc3", result[1].Id);
        Assert.Equal("doc2", result[2].Id);
    }

    [Fact]
    public void DiversityReranker_IdenticalDocuments_PenalizedHeavily()
    {
        // Two identical documents: Jaccard = 1.0
        // With lambda=0.5: score = 0.5*rel - 0.5*1.0
        // The second identical doc gets a heavy penalty
        var reranker = new DiversityReranker<double>(0.5);

        var docs = new List<Document<double>>
        {
            new("doc1", "climate change effects on polar bears") { RelevanceScore = 0.95, HasRelevanceScore = true },
            new("doc2", "climate change effects on polar bears") { RelevanceScore = 0.93, HasRelevanceScore = true },
            new("doc3", "renewable energy solar panels wind turbines") { RelevanceScore = 0.60, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("climate", docs).ToList();

        // doc1 first (highest relevance)
        Assert.Equal("doc1", result[0].Id);
        // doc3 should be second despite lower relevance (Jaccard with doc1 ≈ 0)
        // doc2 has Jaccard=1.0 with doc1, so score = 0.5*0.93 - 0.5*1.0 = -0.035
        // doc3 has Jaccard≈0 with doc1, so score = 0.5*0.60 - 0.5*0 = 0.30
        Assert.Equal("doc3", result[1].Id);
        Assert.Equal("doc2", result[2].Id);
    }

    [Fact]
    public void DiversityReranker_LambdaOne_PureRelevance()
    {
        // Lambda=1.0 means only relevance matters, no diversity penalty
        var reranker = new DiversityReranker<double>(1.0);

        var docs = new List<Document<double>>
        {
            new("doc1", "same words same words") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "same words same words") { RelevanceScore = 0.8, HasRelevanceScore = true },
            new("doc3", "totally different content here") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("query", docs).ToList();

        // Pure relevance order regardless of similarity
        Assert.Equal("doc1", result[0].Id);
        Assert.Equal("doc2", result[1].Id);
        Assert.Equal("doc3", result[2].Id);
    }

    [Fact]
    public void DiversityReranker_LambdaZero_PureDiversity()
    {
        // Lambda=0.0 means only diversity matters
        var reranker = new DiversityReranker<double>(0.0);

        var docs = new List<Document<double>>
        {
            new("doc1", "the cat sat") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "the cat sat on mat") { RelevanceScore = 0.85, HasRelevanceScore = true },
            new("doc3", "dogs run in park") { RelevanceScore = 0.3, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("query", docs).ToList();

        // doc1 still first (highest relevance for initial pick)
        Assert.Equal("doc1", result[0].Id);

        // doc3 second (most diverse from doc1 despite lowest relevance)
        // With lambda=0: score = 0*rel - 1.0*sim → purely penalizing similarity
        Assert.Equal("doc3", result[1].Id);
    }

    [Fact]
    public void DiversityReranker_InvalidLambda_Throws()
    {
        Assert.Throws<ArgumentException>(() => new DiversityReranker<double>(-0.1));
        Assert.Throws<ArgumentException>(() => new DiversityReranker<double>(1.1));
    }

    [Fact]
    public void DiversityReranker_FinalScores_AreRankBased()
    {
        // After diversity reranking, scores are set to (totalDocs - rank + 1) / totalDocs
        // For 3 docs: rank 1 → (3-1+1)/3 = 3/3 = 1.0
        //             rank 2 → (3-2+1)/3 = 2/3 ≈ 0.667
        //             rank 3 → (3-3+1)/3 = 1/3 ≈ 0.333
        var reranker = new DiversityReranker<double>(0.5);

        var docs = new List<Document<double>>
        {
            new("doc1", "aaa bbb ccc") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "ddd eee fff") { RelevanceScore = 0.8, HasRelevanceScore = true },
            new("doc3", "ggg hhh iii") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("query", docs).ToList();

        Assert.Equal(3.0 / 3.0, Convert.ToDouble(result[0].RelevanceScore), Tolerance);
        Assert.Equal(2.0 / 3.0, Convert.ToDouble(result[1].RelevanceScore), Tolerance);
        Assert.Equal(1.0 / 3.0, Convert.ToDouble(result[2].RelevanceScore), Tolerance);
    }

    [Fact]
    public void DiversityReranker_SingleDocument_ReturnsAsIs()
    {
        var reranker = new DiversityReranker<double>(0.5);

        var docs = new List<Document<double>>
        {
            new("doc1", "only document") { RelevanceScore = 0.9, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("query", docs).ToList();

        Assert.Single(result);
        Assert.Equal("doc1", result[0].Id);
    }

    [Fact]
    public void DiversityReranker_EmptyContent_JaccardIsZero()
    {
        // Empty content should produce Jaccard similarity of 0
        var reranker = new DiversityReranker<double>(0.5);

        var docs = new List<Document<double>>
        {
            new("doc1", "hello world") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "") { RelevanceScore = 0.85, HasRelevanceScore = true },
            new("doc3", "hello world again") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("query", docs).ToList();

        // Empty doc has zero similarity to everything, so it gets no diversity penalty
        Assert.Equal("doc1", result[0].Id);
        // doc2 (empty) has 0 Jaccard with doc1: score = 0.5*0.85 - 0.5*0 = 0.425
        // doc3 shares {hello, world}: intersection=2, union≈4 → Jaccard≈0.5
        // doc3 score: 0.5*0.7 - 0.5*0.5 = 0.35 - 0.25 = 0.10
        // doc2 > doc3
        Assert.Equal("doc2", result[1].Id);
    }

    #endregion

    #region MaximalMarginalRelevanceReranker - Cosine Similarity Golden References

    [Fact]
    public void MMR_CosineSimilarity_GoldenReference()
    {
        // Vectors:
        // A = [1, 0, 0] (relevance = 0.9)
        // B = [1, 0, 0] (relevance = 0.85) - identical to A
        // C = [0, 1, 0] (relevance = 0.7)  - orthogonal to A
        //
        // Cosine(A,B) = 1.0 (identical)
        // Cosine(A,C) = 0.0 (orthogonal)
        //
        // Lambda=0.5, first pick = A (highest relevance)
        // Next: B score = 0.5*0.85 - 0.5*1.0 = 0.425 - 0.5 = -0.075
        //       C score = 0.5*0.70 - 0.5*0.0 = 0.35  - 0   =  0.35
        // C wins! (diversity beats high similarity)
        var embeddings = new Dictionary<string, Vector<double>>
        {
            { "A", new Vector<double>(new double[] { 1, 0, 0 }) },
            { "B", new Vector<double>(new double[] { 1, 0, 0 }) },
            { "C", new Vector<double>(new double[] { 0, 1, 0 }) },
        };

        Func<Document<double>, Vector<double>> getEmb = doc => embeddings[doc.Id];
        var mmr = new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: 0.5);

        var docs = new List<Document<double>>
        {
            new("A", "doc A") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("B", "doc B") { RelevanceScore = 0.85, HasRelevanceScore = true },
            new("C", "doc C") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = mmr.Rerank("query", docs).ToList();

        Assert.Equal("A", result[0].Id); // Most relevant first
        Assert.Equal("C", result[1].Id); // Orthogonal beats identical
        Assert.Equal("B", result[2].Id); // Identical to A, penalized
    }

    [Fact]
    public void MMR_LambdaOne_PureRelevanceOrdering()
    {
        // Lambda=1.0: MMR = 1.0*relevance - 0.0*similarity = pure relevance
        var embeddings = new Dictionary<string, Vector<double>>
        {
            { "A", new Vector<double>(new double[] { 1, 0 }) },
            { "B", new Vector<double>(new double[] { 1, 0 }) },
            { "C", new Vector<double>(new double[] { 0, 1 }) },
        };

        Func<Document<double>, Vector<double>> getEmb = doc => embeddings[doc.Id];
        var mmr = new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: 1.0);

        var docs = new List<Document<double>>
        {
            new("A", "doc A") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("B", "doc B") { RelevanceScore = 0.8, HasRelevanceScore = true },
            new("C", "doc C") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = mmr.Rerank("query", docs).ToList();

        // Pure relevance order
        Assert.Equal("A", result[0].Id);
        Assert.Equal("B", result[1].Id);
        Assert.Equal("C", result[2].Id);
    }

    [Fact]
    public void MMR_FinalScores_ReflectMMRPosition()
    {
        // After MMR reranking, scores are set to 1.0 - (i / count)
        // For 3 docs: position 0 → 1.0 - 0/3 = 1.0
        //             position 1 → 1.0 - 1/3 ≈ 0.667
        //             position 2 → 1.0 - 2/3 ≈ 0.333
        var embeddings = new Dictionary<string, Vector<double>>
        {
            { "A", new Vector<double>(new double[] { 1, 0 }) },
            { "B", new Vector<double>(new double[] { 0, 1 }) },
            { "C", new Vector<double>(new double[] { 1, 1 }) },
        };

        Func<Document<double>, Vector<double>> getEmb = doc => embeddings[doc.Id];
        var mmr = new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: 0.7);

        var docs = new List<Document<double>>
        {
            new("A", "doc A") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("B", "doc B") { RelevanceScore = 0.8, HasRelevanceScore = true },
            new("C", "doc C") { RelevanceScore = 0.7, HasRelevanceScore = true },
        };

        var result = mmr.Rerank("query", docs).ToList();

        Assert.Equal(1.0, Convert.ToDouble(result[0].RelevanceScore), Tolerance);
        Assert.Equal(1.0 - 1.0 / 3.0, Convert.ToDouble(result[1].RelevanceScore), Tolerance);
        Assert.Equal(1.0 - 2.0 / 3.0, Convert.ToDouble(result[2].RelevanceScore), Tolerance);
    }

    [Fact]
    public void MMR_OrthogonalVectors_MaxDiversity()
    {
        // 4 orthogonal vectors: cosine similarity between any pair = 0
        // So diversity penalty is always 0, and MMR degrades to pure relevance
        var embeddings = new Dictionary<string, Vector<double>>
        {
            { "A", new Vector<double>(new double[] { 1, 0, 0, 0 }) },
            { "B", new Vector<double>(new double[] { 0, 1, 0, 0 }) },
            { "C", new Vector<double>(new double[] { 0, 0, 1, 0 }) },
            { "D", new Vector<double>(new double[] { 0, 0, 0, 1 }) },
        };

        Func<Document<double>, Vector<double>> getEmb = doc => embeddings[doc.Id];
        var mmr = new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: 0.5);

        var docs = new List<Document<double>>
        {
            new("A", "doc A") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("B", "doc B") { RelevanceScore = 0.8, HasRelevanceScore = true },
            new("C", "doc C") { RelevanceScore = 0.7, HasRelevanceScore = true },
            new("D", "doc D") { RelevanceScore = 0.6, HasRelevanceScore = true },
        };

        var result = mmr.Rerank("query", docs).ToList();

        // With orthogonal vectors and lambda=0.5, relevance determines order
        Assert.Equal("A", result[0].Id);
        Assert.Equal("B", result[1].Id);
        Assert.Equal("C", result[2].Id);
        Assert.Equal("D", result[3].Id);
    }

    [Fact]
    public void MMR_HighSimilarityCluster_PromotesDiverseDoc()
    {
        // 3 docs with very similar embeddings (close to [1,0]) and 1 diverse doc [0,1]
        // Even though the diverse doc has lower relevance, MMR should promote it
        var embeddings = new Dictionary<string, Vector<double>>
        {
            { "A", new Vector<double>(new double[] { 1.0, 0.0 }) },
            { "B", new Vector<double>(new double[] { 0.99, 0.14 }) },  // cos(A,B) ≈ 0.99
            { "C", new Vector<double>(new double[] { 0.98, 0.20 }) },  // cos(A,C) ≈ 0.98
            { "D", new Vector<double>(new double[] { 0.0, 1.0 }) },    // cos(A,D) = 0.0
        };

        Func<Document<double>, Vector<double>> getEmb = doc => embeddings[doc.Id];
        var mmr = new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: 0.5);

        var docs = new List<Document<double>>
        {
            new("A", "doc A") { RelevanceScore = 0.95, HasRelevanceScore = true },
            new("B", "doc B") { RelevanceScore = 0.90, HasRelevanceScore = true },
            new("C", "doc C") { RelevanceScore = 0.85, HasRelevanceScore = true },
            new("D", "doc D") { RelevanceScore = 0.50, HasRelevanceScore = true },
        };

        var result = mmr.Rerank("query", docs).ToList();

        // A first (highest relevance)
        Assert.Equal("A", result[0].Id);
        // D second (orthogonal to A despite lower relevance)
        // B: 0.5*0.9 - 0.5*0.99 ≈ 0.45 - 0.495 = -0.045
        // D: 0.5*0.5 - 0.5*0.0  = 0.25  - 0     =  0.25
        Assert.Equal("D", result[1].Id);
    }

    [Fact]
    public void MMR_InvalidLambda_Throws()
    {
        var getEmb = (Document<double> doc) => new Vector<double>(new double[] { 1 });
        Assert.Throws<ArgumentException>(() => new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: -0.1));
        Assert.Throws<ArgumentException>(() => new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: 1.1));
    }

    [Fact]
    public void MMR_NullEmbeddingFunc_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new MaximalMarginalRelevanceReranker<double>(null!, lambda: 0.5));
    }

    [Fact]
    public void MMR_SingleDocument_ReturnsIt()
    {
        var embeddings = new Dictionary<string, Vector<double>>
        {
            { "A", new Vector<double>(new double[] { 1, 0 }) },
        };

        Func<Document<double>, Vector<double>> getEmb = doc => embeddings[doc.Id];
        var mmr = new MaximalMarginalRelevanceReranker<double>(getEmb, lambda: 0.7);

        var docs = new List<Document<double>>
        {
            new("A", "doc A") { RelevanceScore = 0.9, HasRelevanceScore = true },
        };

        var result = mmr.Rerank("query", docs).ToList();

        Assert.Single(result);
        Assert.Equal("A", result[0].Id);
    }

    #endregion

    #region Reranker Base Validation Tests

    [Fact]
    public void Reranker_EmptyQuery_ThrowsArgumentException()
    {
        var reranker = new DiversityReranker<double>(0.5);
        var docs = new List<Document<double>> { new("doc1", "content") };

        Assert.Throws<ArgumentException>(() => reranker.Rerank("", docs));
        Assert.Throws<ArgumentException>(() => reranker.Rerank("  ", docs));
    }

    [Fact]
    public void Reranker_NullDocuments_ThrowsArgumentNullException()
    {
        var reranker = new DiversityReranker<double>(0.5);

        Assert.Throws<ArgumentNullException>(() => reranker.Rerank("query", null!));
    }

    [Fact]
    public void Reranker_TopK_ReturnsLimitedResults()
    {
        var reranker = new DiversityReranker<double>(0.5);

        var docs = new List<Document<double>>
        {
            new("doc1", "aaa") { RelevanceScore = 0.9, HasRelevanceScore = true },
            new("doc2", "bbb") { RelevanceScore = 0.8, HasRelevanceScore = true },
            new("doc3", "ccc") { RelevanceScore = 0.7, HasRelevanceScore = true },
            new("doc4", "ddd") { RelevanceScore = 0.6, HasRelevanceScore = true },
            new("doc5", "eee") { RelevanceScore = 0.5, HasRelevanceScore = true },
        };

        var result = reranker.Rerank("query", docs, topK: 3).ToList();

        Assert.Equal(3, result.Count);
    }

    [Fact]
    public void Reranker_InvalidTopK_ThrowsArgumentException()
    {
        var reranker = new DiversityReranker<double>(0.5);
        var docs = new List<Document<double>> { new("doc1", "content") };

        Assert.Throws<ArgumentException>(() => reranker.Rerank("query", docs, topK: 0));
        Assert.Throws<ArgumentException>(() => reranker.Rerank("query", docs, topK: -1));
    }

    #endregion
}
