using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Deep mathematical correctness tests for HNSW index and vector search metrics.
/// Verifies cosine similarity, euclidean distance golden references, HNSW graph
/// properties (recall, topology, batch operations), and edge cases.
/// </summary>
public class RAGVectorSearchIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region CosineSimilarityMetric - Golden References

    [Fact]
    public void CosineSimilarity_IdenticalVectors_IsOne()
    {
        var metric = new CosineSimilarityMetric<double>();
        var v = new Vector<double>(new double[] { 1, 2, 3 });

        var result = Convert.ToDouble(metric.Calculate(v, v));

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_IsZero()
    {
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 0, 0 });
        var v2 = new Vector<double>(new double[] { 0, 1, 0 });

        var result = Convert.ToDouble(metric.Calculate(v1, v2));

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_OppositeVectors_IsNegativeOne()
    {
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { -1, -2, -3 });

        var result = Convert.ToDouble(metric.Calculate(v1, v2));

        Assert.Equal(-1.0, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_45Degrees_GoldenReference()
    {
        // cos(45°) = 1/√2 ≈ 0.7071
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 0 });
        var v2 = new Vector<double>(new double[] { 1, 1 });

        var result = Convert.ToDouble(metric.Calculate(v1, v2));

        double expected = 1.0 / Math.Sqrt(2.0);
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_ScaleInvariant()
    {
        // Cosine similarity should be independent of vector magnitude
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 2, 4, 6 }); // v1 * 2
        var v3 = new Vector<double>(new double[] { 100, 200, 300 }); // v1 * 100

        double sim12 = Convert.ToDouble(metric.Calculate(v1, v2));
        double sim13 = Convert.ToDouble(metric.Calculate(v1, v3));

        Assert.Equal(1.0, sim12, Tolerance);
        Assert.Equal(1.0, sim13, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_HigherIsBetter_IsTrue()
    {
        var metric = new CosineSimilarityMetric<double>();
        Assert.True(metric.HigherIsBetter);
    }

    [Fact]
    public void CosineSimilarity_3DVectors_GoldenReference()
    {
        // v1 = [1, 2, 3], v2 = [4, 5, 6]
        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        // |v1| = sqrt(1+4+9) = sqrt(14)
        // |v2| = sqrt(16+25+36) = sqrt(77)
        // cos = 32 / (sqrt(14) * sqrt(77)) = 32 / sqrt(1078)
        var metric = new CosineSimilarityMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 4, 5, 6 });

        double result = Convert.ToDouble(metric.Calculate(v1, v2));
        double expected = 32.0 / Math.Sqrt(14.0 * 77.0);

        Assert.Equal(expected, result, Tolerance);
    }

    #endregion

    #region EuclideanDistanceMetric - Golden References

    [Fact]
    public void EuclideanDistance_IdenticalVectors_IsZero()
    {
        var metric = new EuclideanDistanceMetric<double>();
        var v = new Vector<double>(new double[] { 1, 2, 3 });

        var result = Convert.ToDouble(metric.Calculate(v, v));

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_UnitVectors_GoldenReference()
    {
        // |[1,0,0] - [0,1,0]| = sqrt(1+1+0) = sqrt(2) ≈ 1.4142
        var metric = new EuclideanDistanceMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 0, 0 });
        var v2 = new Vector<double>(new double[] { 0, 1, 0 });

        var result = Convert.ToDouble(metric.Calculate(v1, v2));

        Assert.Equal(Math.Sqrt(2.0), result, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_3D_GoldenReference()
    {
        // |[1,2,3] - [4,5,6]| = sqrt(9+9+9) = sqrt(27) = 3*sqrt(3)
        var metric = new EuclideanDistanceMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 4, 5, 6 });

        var result = Convert.ToDouble(metric.Calculate(v1, v2));
        double expected = Math.Sqrt(27.0);

        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_HigherIsBetter_IsFalse()
    {
        var metric = new EuclideanDistanceMetric<double>();
        Assert.False(metric.HigherIsBetter);
    }

    [Fact]
    public void EuclideanDistance_TriangleInequality()
    {
        // d(A,C) <= d(A,B) + d(B,C)
        var metric = new EuclideanDistanceMetric<double>();
        var a = new Vector<double>(new double[] { 0, 0 });
        var b = new Vector<double>(new double[] { 3, 0 });
        var c = new Vector<double>(new double[] { 3, 4 });

        double dAB = Convert.ToDouble(metric.Calculate(a, b)); // 3
        double dBC = Convert.ToDouble(metric.Calculate(b, c)); // 4
        double dAC = Convert.ToDouble(metric.Calculate(a, c)); // 5

        Assert.Equal(3.0, dAB, Tolerance);
        Assert.Equal(4.0, dBC, Tolerance);
        Assert.Equal(5.0, dAC, Tolerance);
        Assert.True(dAC <= dAB + dBC + Tolerance);
    }

    [Fact]
    public void EuclideanDistance_Symmetry()
    {
        var metric = new EuclideanDistanceMetric<double>();
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 4, 5, 6 });

        double d12 = Convert.ToDouble(metric.Calculate(v1, v2));
        double d21 = Convert.ToDouble(metric.Calculate(v2, v1));

        Assert.Equal(d12, d21, Tolerance);
    }

    #endregion

    #region HNSW Index - Graph Properties and Recall

    [Fact]
    public void HNSW_AddSingle_CountIsOne()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        index.Add("doc1", new Vector<double>(new double[] { 1, 0, 0 }));

        Assert.Equal(1, index.Count);
    }

    [Fact]
    public void HNSW_AddMultiple_CountCorrect()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        for (int i = 0; i < 10; i++)
        {
            index.Add($"doc{i}", new Vector<double>(new double[] { Math.Cos(i), Math.Sin(i), 0 }));
        }

        Assert.Equal(10, index.Count);
    }

    [Fact]
    public void HNSW_Search_IdenticalVector_ReturnsSelf()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        var target = new Vector<double>(new double[] { 1, 0, 0 });
        index.Add("target", target);
        index.Add("other", new Vector<double>(new double[] { 0, 1, 0 }));

        var results = index.Search(target, k: 1);

        Assert.Single(results);
        Assert.Equal("target", results[0].Id);
    }

    [Fact]
    public void HNSW_Search_OrthogonalVectors_CorrectRanking()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        index.Add("x", new Vector<double>(new double[] { 1, 0, 0 }));
        index.Add("y", new Vector<double>(new double[] { 0, 1, 0 }));
        index.Add("z", new Vector<double>(new double[] { 0, 0, 1 }));

        // Search for [1, 0, 0] → should find "x" first
        var results = index.Search(new Vector<double>(new double[] { 1, 0, 0 }), k: 3);

        Assert.Equal(3, results.Count);
        Assert.Equal("x", results[0].Id);
        double topScore = Convert.ToDouble(results[0].Score);
        Assert.True(topScore > 0.9, $"Top score {topScore} should be close to 1.0");
    }

    [Fact]
    public void HNSW_Search_EmptyIndex_ReturnsEmpty()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        var results = index.Search(new Vector<double>(new double[] { 1, 0 }), k: 5);

        Assert.Empty(results);
    }

    [Fact]
    public void HNSW_Search_KLargerThanCount_ReturnsAll()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        index.Add("d1", new Vector<double>(new double[] { 1, 0 }));
        index.Add("d2", new Vector<double>(new double[] { 0, 1 }));

        var results = index.Search(new Vector<double>(new double[] { 1, 0 }), k: 100);

        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void HNSW_Remove_DecreasesCount()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        index.Add("d1", new Vector<double>(new double[] { 1, 0 }));
        index.Add("d2", new Vector<double>(new double[] { 0, 1 }));
        Assert.Equal(2, index.Count);

        index.Remove("d1");
        Assert.Equal(1, index.Count);
    }

    [Fact]
    public void HNSW_Remove_ExcludesFromSearch()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        index.Add("keep", new Vector<double>(new double[] { 1, 0 }));
        index.Add("remove", new Vector<double>(new double[] { 0.99, 0.1 }));

        index.Remove("remove");

        var results = index.Search(new Vector<double>(new double[] { 1, 0 }), k: 10);

        Assert.Single(results);
        Assert.Equal("keep", results[0].Id);
    }

    [Fact]
    public void HNSW_Clear_ResetsIndex()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        for (int i = 0; i < 20; i++)
        {
            index.Add($"d{i}", new Vector<double>(new double[] { i * 0.1, 1 - i * 0.1 }));
        }

        index.Clear();
        Assert.Equal(0, index.Count);

        var results = index.Search(new Vector<double>(new double[] { 1, 0 }), k: 5);
        Assert.Empty(results);
    }

    [Fact]
    public void HNSW_AddBatch_EquivalentToIndividualAdds()
    {
        var index1 = new HNSWIndex<double>(new CosineSimilarityMetric<double>(), seed: 42);
        var index2 = new HNSWIndex<double>(new CosineSimilarityMetric<double>(), seed: 42);

        var vectors = new Dictionary<string, Vector<double>>
        {
            { "a", new Vector<double>(new double[] { 1, 0, 0 }) },
            { "b", new Vector<double>(new double[] { 0, 1, 0 }) },
            { "c", new Vector<double>(new double[] { 0, 0, 1 }) },
        };

        // Add individually to index1
        foreach (var kvp in vectors)
        {
            index1.Add(kvp.Key, kvp.Value);
        }

        // Add as batch to index2
        index2.AddBatch(vectors);

        Assert.Equal(index1.Count, index2.Count);

        // Both should find the same nearest neighbor
        var query = new Vector<double>(new double[] { 1, 0, 0 });
        var results1 = index1.Search(query, k: 3);
        var results2 = index2.Search(query, k: 3);

        Assert.Equal(results1[0].Id, results2[0].Id);
    }

    [Fact]
    public void HNSW_Recall_50Vectors_HighRecall()
    {
        // HNSW should achieve high recall for approximate nearest neighbors
        var index = new HNSWIndex<double>(
            new CosineSimilarityMetric<double>(),
            maxConnections: 16, efConstruction: 200, efSearch: 50, seed: 42);

        var metric = new CosineSimilarityMetric<double>();

        // Add 50 random unit vectors
        var vectors = new Dictionary<string, Vector<double>>();
        var random = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            double angle = random.NextDouble() * 2 * Math.PI;
            double z = random.NextDouble() * 2 - 1;
            double r = Math.Sqrt(1 - z * z);
            var v = new Vector<double>(new double[] { r * Math.Cos(angle), r * Math.Sin(angle), z });
            vectors[$"d{i}"] = v;
            index.Add($"d{i}", v);
        }

        // For 10 random queries, check that HNSW finds the true nearest neighbor
        int correct = 0;
        for (int q = 0; q < 10; q++)
        {
            double angle = random.NextDouble() * 2 * Math.PI;
            double z = random.NextDouble() * 2 - 1;
            double r = Math.Sqrt(1 - z * z);
            var query = new Vector<double>(new double[] { r * Math.Cos(angle), r * Math.Sin(angle), z });

            // Brute-force find true nearest neighbor
            string trueBest = vectors
                .OrderByDescending(kvp => Convert.ToDouble(metric.Calculate(query, kvp.Value)))
                .First().Key;

            // HNSW search
            var hnswResults = index.Search(query, k: 5);

            // Check if true nearest neighbor is in top-5 HNSW results
            if (hnswResults.Any(r => r.Id == trueBest))
            {
                correct++;
            }
        }

        // With good HNSW parameters, recall@5 should be very high
        Assert.True(correct >= 8,
            $"HNSW recall@5 is {correct}/10 - expected >= 8 for 50 vectors");
    }

    [Fact]
    public void HNSW_InvalidK_Throws()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        Assert.Throws<ArgumentException>(() =>
            index.Search(new Vector<double>(new double[] { 1, 0 }), k: 0));
    }

    [Fact]
    public void HNSW_InvalidConstructorParams_Throws()
    {
        var metric = new CosineSimilarityMetric<double>();

        Assert.Throws<ArgumentException>(() =>
            new HNSWIndex<double>(metric, maxConnections: 1)); // Must be >= 2
        Assert.Throws<ArgumentException>(() =>
            new HNSWIndex<double>(metric, efConstruction: 0));
        Assert.Throws<ArgumentException>(() =>
            new HNSWIndex<double>(metric, efSearch: 0));
    }

    [Fact]
    public void HNSW_UpdateExisting_OverwritesVector()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        index.Add("d1", new Vector<double>(new double[] { 1, 0 }));
        index.Add("d1", new Vector<double>(new double[] { 0, 1 })); // Overwrite

        // Search for [0, 1] should find d1 with high similarity
        var results = index.Search(new Vector<double>(new double[] { 0, 1 }), k: 1);
        Assert.Equal("d1", results[0].Id);

        double score = Convert.ToDouble(results[0].Score);
        Assert.True(score > 0.9, $"After overwrite, score {score} should be high");
    }

    [Fact]
    public void HNSW_SearchResults_SortedByScore()
    {
        var index = new HNSWIndex<double>(new CosineSimilarityMetric<double>());

        for (int i = 0; i < 20; i++)
        {
            double angle = i * Math.PI / 10;
            index.Add($"d{i}", new Vector<double>(new double[] { Math.Cos(angle), Math.Sin(angle) }));
        }

        var results = index.Search(new Vector<double>(new double[] { 1, 0 }), k: 10);

        // Results should be sorted by decreasing score
        for (int i = 1; i < results.Count; i++)
        {
            double prevScore = Convert.ToDouble(results[i - 1].Score);
            double currScore = Convert.ToDouble(results[i].Score);
            Assert.True(prevScore >= currScore - Tolerance,
                $"Results not sorted: [{i-1}]={prevScore}, [{i}]={currScore}");
        }
    }

    #endregion
}
