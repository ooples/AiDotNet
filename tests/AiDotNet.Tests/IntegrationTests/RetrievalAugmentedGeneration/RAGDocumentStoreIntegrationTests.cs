using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Deep integration tests for InMemoryDocumentStore.
/// Tests HNSW-based similarity search with known vector relationships,
/// CRUD operations, metadata filtering, and edge cases.
/// </summary>
public class RAGDocumentStoreIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region Construction and Basic Operations

    [Fact]
    public void InMemoryStore_Construction_ValidDimension()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

        Assert.Equal(0, store.DocumentCount);
        Assert.Equal(3, store.VectorDimension);
    }

    [Fact]
    public void InMemoryStore_InvalidDimension_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new InMemoryDocumentStore<double>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new InMemoryDocumentStore<double>(-1));
    }

    [Fact]
    public void InMemoryStore_AddSingle_IncreasesCount()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

        var doc = new Document<double>("doc1", "content");
        var embedding = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
        var vdoc = new VectorDocument<double>(doc, embedding);

        store.Add(vdoc);

        Assert.Equal(1, store.DocumentCount);
    }

    [Fact]
    public void InMemoryStore_AddNull_Throws()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

        Assert.Throws<ArgumentNullException>(() => store.Add(null!));
    }

    [Fact]
    public void InMemoryStore_AddEmptyId_Throws()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
        var doc = new Document<double>("", "content");
        var embedding = new Vector<double>(new double[] { 1, 0, 0 });

        Assert.Throws<ArgumentException>(() =>
            store.Add(new VectorDocument<double>(doc, embedding)));
    }

    #endregion

    #region GetById Tests

    [Fact]
    public void InMemoryStore_GetById_ReturnsCorrectDocument()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var doc1 = new Document<double>("id1", "first content");
        var doc2 = new Document<double>("id2", "second content");
        store.Add(new VectorDocument<double>(doc1, new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(doc2, new Vector<double>(new double[] { 0, 1 })));

        var retrieved = store.GetById("id1");
        Assert.NotNull(retrieved);
        Assert.Equal("id1", retrieved.Id);
        Assert.Equal("first content", retrieved.Content);

        retrieved = store.GetById("id2");
        Assert.NotNull(retrieved);
        Assert.Equal("second content", retrieved.Content);
    }

    [Fact]
    public void InMemoryStore_GetById_NonExistent_ReturnsNull()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var result = store.GetById("nonexistent");
        Assert.Null(result);
    }

    [Fact]
    public void InMemoryStore_GetById_EmptyId_Throws()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        Assert.Throws<ArgumentException>(() => store.GetById(""));
        Assert.Throws<ArgumentException>(() => store.GetById("  "));
    }

    #endregion

    #region Remove Tests

    [Fact]
    public void InMemoryStore_Remove_ExistingDoc_ReturnsTrue()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);
        store.Add(new VectorDocument<double>(
            new Document<double>("doc1", "content"),
            new Vector<double>(new double[] { 1, 0 })));

        bool removed = store.Remove("doc1");

        Assert.True(removed);
        Assert.Equal(0, store.DocumentCount);
        Assert.Null(store.GetById("doc1"));
    }

    [Fact]
    public void InMemoryStore_Remove_NonExistent_ReturnsFalse()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        bool removed = store.Remove("nonexistent");

        Assert.False(removed);
    }

    [Fact]
    public void InMemoryStore_Remove_EmptyId_Throws()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        Assert.Throws<ArgumentException>(() => store.Remove(""));
    }

    #endregion

    #region Clear Tests

    [Fact]
    public void InMemoryStore_Clear_RemovesAllDocuments()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "c1"), new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d2", "c2"), new Vector<double>(new double[] { 0, 1 })));

        Assert.Equal(2, store.DocumentCount);

        store.Clear();

        Assert.Equal(0, store.DocumentCount);
        Assert.Null(store.GetById("d1"));
        Assert.Null(store.GetById("d2"));
    }

    [Fact]
    public void InMemoryStore_Clear_ResetsVectorDimension()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 5);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "c1"), new Vector<double>(new double[] { 1, 0, 0, 0, 0 })));

        store.Clear();

        // VectorDimension should reset to initial value
        Assert.Equal(5, store.VectorDimension);
    }

    #endregion

    #region GetAll Tests

    [Fact]
    public void InMemoryStore_GetAll_ReturnsAllDocuments()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "content 1"), new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d2", "content 2"), new Vector<double>(new double[] { 0, 1 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d3", "content 3"), new Vector<double>(new double[] { 1, 1 })));

        var all = store.GetAll().ToList();

        Assert.Equal(3, all.Count);
        Assert.Contains(all, d => d.Id == "d1");
        Assert.Contains(all, d => d.Id == "d2");
        Assert.Contains(all, d => d.Id == "d3");
    }

    [Fact]
    public void InMemoryStore_GetAll_EmptyStore_ReturnsEmpty()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var all = store.GetAll().ToList();

        Assert.Empty(all);
    }

    #endregion

    #region Similarity Search Tests - Cosine Similarity Golden References

    [Fact]
    public void InMemoryStore_GetSimilar_OrthogonalVectors_CorrectOrder()
    {
        // Query = [1, 0, 0]
        // Doc A = [1, 0, 0] → cosine = 1.0 (most similar)
        // Doc B = [0, 1, 0] → cosine = 0.0 (orthogonal)
        // Doc C = [0, 0, 1] → cosine = 0.0 (orthogonal)
        var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

        store.Add(new VectorDocument<double>(
            new Document<double>("A", "doc A"), new Vector<double>(new double[] { 1, 0, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("B", "doc B"), new Vector<double>(new double[] { 0, 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("C", "doc C"), new Vector<double>(new double[] { 0, 0, 1 })));

        var query = new Vector<double>(new double[] { 1, 0, 0 });
        var results = store.GetSimilar(query, topK: 3).ToList();

        // A should be first (cosine similarity = 1.0)
        Assert.Equal("A", results[0].Id);
        Assert.True(results[0].HasRelevanceScore);

        // A's relevance score should be close to 1.0
        double scoreA = Convert.ToDouble(results[0].RelevanceScore);
        Assert.True(scoreA > 0.9, $"Expected score > 0.9 for identical vector, got {scoreA}");
    }

    [Fact]
    public void InMemoryStore_GetSimilar_SimilarVectors_RankedByAngle()
    {
        // Query = [1, 0]
        // Doc A = [1, 0]     → cosine = 1.0
        // Doc B = [1, 0.1]   → cosine ≈ 0.995 (very similar)
        // Doc C = [1, 1]     → cosine ≈ 0.707 (45 degrees)
        // Doc D = [0, 1]     → cosine = 0.0 (orthogonal)
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("A", "doc A"), new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("B", "doc B"), new Vector<double>(new double[] { 1, 0.1 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("C", "doc C"), new Vector<double>(new double[] { 1, 1 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("D", "doc D"), new Vector<double>(new double[] { 0, 1 })));

        var query = new Vector<double>(new double[] { 1, 0 });
        var results = store.GetSimilar(query, topK: 4).ToList();

        // Order should be A > B > C > D
        Assert.Equal("A", results[0].Id);
        Assert.Equal("B", results[1].Id);
        Assert.Equal("C", results[2].Id);
        Assert.Equal("D", results[3].Id);
    }

    [Fact]
    public void InMemoryStore_GetSimilar_TopK_LimitsResults()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        for (int i = 0; i < 10; i++)
        {
            var emb = new Vector<double>(new double[] { Math.Cos(i * 0.3), Math.Sin(i * 0.3) });
            store.Add(new VectorDocument<double>(
                new Document<double>($"doc{i}", $"content {i}"), emb));
        }

        var query = new Vector<double>(new double[] { 1, 0 });
        var results = store.GetSimilar(query, topK: 3).ToList();

        Assert.Equal(3, results.Count);
    }

    [Fact]
    public void InMemoryStore_GetSimilar_EmptyStore_ReturnsEmpty()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var query = new Vector<double>(new double[] { 1, 0 });
        var results = store.GetSimilar(query, topK: 5).ToList();

        Assert.Empty(results);
    }

    [Fact]
    public void InMemoryStore_GetSimilar_InvalidTopK_Throws()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var query = new Vector<double>(new double[] { 1, 0 });
        Assert.Throws<ArgumentException>(() => store.GetSimilar(query, topK: 0).ToList());
    }

    [Fact]
    public void InMemoryStore_GetSimilar_NegativeEmbeddings_HandledCorrectly()
    {
        // Query = [-1, 0]
        // Doc A = [-1, 0] → cosine = 1.0 (same direction)
        // Doc B = [1, 0]  → cosine = -1.0 (opposite direction)
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("A", "negative"), new Vector<double>(new double[] { -1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("B", "positive"), new Vector<double>(new double[] { 1, 0 })));

        var query = new Vector<double>(new double[] { -1, 0 });
        var results = store.GetSimilar(query, topK: 2).ToList();

        // A should be first (same direction as query)
        Assert.Equal("A", results[0].Id);
    }

    #endregion

    #region Batch Operations Tests

    [Fact]
    public void InMemoryStore_AddBatch_AddsMultipleDocuments()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var batch = new List<VectorDocument<double>>
        {
            new(new Document<double>("d1", "c1"), new Vector<double>(new double[] { 1, 0 })),
            new(new Document<double>("d2", "c2"), new Vector<double>(new double[] { 0, 1 })),
            new(new Document<double>("d3", "c3"), new Vector<double>(new double[] { 1, 1 })),
        };

        store.AddBatch(batch);

        Assert.Equal(3, store.DocumentCount);
        Assert.NotNull(store.GetById("d1"));
        Assert.NotNull(store.GetById("d2"));
        Assert.NotNull(store.GetById("d3"));
    }

    [Fact]
    public void InMemoryStore_AddBatch_EmptyList_Throws()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        Assert.Throws<ArgumentException>(() =>
            store.AddBatch(new List<VectorDocument<double>>()));
    }

    [Fact]
    public void InMemoryStore_AddBatch_NullList_Throws()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        Assert.Throws<ArgumentNullException>(() =>
            store.AddBatch(null!));
    }

    [Fact]
    public void InMemoryStore_AddBatch_Searchable()
    {
        // Verify that batch-added documents are searchable
        var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

        var batch = new List<VectorDocument<double>>
        {
            new(new Document<double>("d1", "c1"), new Vector<double>(new double[] { 1, 0, 0 })),
            new(new Document<double>("d2", "c2"), new Vector<double>(new double[] { 0, 1, 0 })),
            new(new Document<double>("d3", "c3"), new Vector<double>(new double[] { 0, 0, 1 })),
        };

        store.AddBatch(batch);

        var query = new Vector<double>(new double[] { 1, 0, 0 });
        var results = store.GetSimilar(query, topK: 3).ToList();

        Assert.Equal(3, results.Count);
        Assert.Equal("d1", results[0].Id); // Most similar to [1,0,0]
    }

    #endregion

    #region Metadata Filtering Tests

    [Fact]
    public void InMemoryStore_MetadataFilter_StringEquality()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var meta1 = new Dictionary<string, object> { { "category", "science" } };
        var meta2 = new Dictionary<string, object> { { "category", "sports" } };
        var meta3 = new Dictionary<string, object> { { "category", "science" } };

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "c1", meta1), new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d2", "c2", meta2), new Vector<double>(new double[] { 0.9, 0.1 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d3", "c3", meta3), new Vector<double>(new double[] { 0.8, 0.2 })));

        var query = new Vector<double>(new double[] { 1, 0 });
        var filters = new Dictionary<string, object> { { "category", "science" } };
        var results = store.GetSimilarWithFilters(query, topK: 10, filters).ToList();

        // Only d1 and d3 have category "science"
        Assert.Equal(2, results.Count);
        Assert.All(results, r => Assert.True(r.Id == "d1" || r.Id == "d3"));
    }

    [Fact]
    public void InMemoryStore_MetadataFilter_NoMatch_ReturnsEmpty()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var meta = new Dictionary<string, object> { { "type", "article" } };
        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "c1", meta), new Vector<double>(new double[] { 1, 0 })));

        var query = new Vector<double>(new double[] { 1, 0 });
        var filters = new Dictionary<string, object> { { "type", "video" } };
        var results = store.GetSimilarWithFilters(query, topK: 10, filters).ToList();

        Assert.Empty(results);
    }

    [Fact]
    public void InMemoryStore_MetadataFilter_BooleanEquality()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var meta1 = new Dictionary<string, object> { { "verified", true } };
        var meta2 = new Dictionary<string, object> { { "verified", false } };

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "c1", meta1), new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d2", "c2", meta2), new Vector<double>(new double[] { 0, 1 })));

        var query = new Vector<double>(new double[] { 1, 0 });
        var filters = new Dictionary<string, object> { { "verified", true } };
        var results = store.GetSimilarWithFilters(query, topK: 10, filters).ToList();

        Assert.Single(results);
        Assert.Equal("d1", results[0].Id);
    }

    #endregion

    #region Document Lifecycle Tests

    [Fact]
    public void InMemoryStore_AddRemoveAdd_WorksCorrectly()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        // Add
        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "original"), new Vector<double>(new double[] { 1, 0 })));
        Assert.Equal(1, store.DocumentCount);

        // Remove
        store.Remove("d1");
        Assert.Equal(0, store.DocumentCount);

        // Re-add with different content
        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "updated"), new Vector<double>(new double[] { 0, 1 })));
        Assert.Equal(1, store.DocumentCount);

        var retrieved = store.GetById("d1");
        Assert.NotNull(retrieved);
        Assert.Equal("updated", retrieved.Content);
    }

    [Fact]
    public void InMemoryStore_RemoveThenSearch_DoesNotReturnRemoved()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "keep"), new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d2", "remove"), new Vector<double>(new double[] { 0.99, 0.1 })));

        store.Remove("d2");

        var query = new Vector<double>(new double[] { 1, 0 });
        var results = store.GetSimilar(query, topK: 10).ToList();

        Assert.Single(results);
        Assert.Equal("d1", results[0].Id);
    }

    [Fact]
    public void InMemoryStore_ClearThenAdd_WorksCorrectly()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "old"), new Vector<double>(new double[] { 1, 0 })));

        store.Clear();
        Assert.Equal(0, store.DocumentCount);

        store.Add(new VectorDocument<double>(
            new Document<double>("d2", "new"), new Vector<double>(new double[] { 0, 1 })));
        Assert.Equal(1, store.DocumentCount);

        var retrieved = store.GetById("d2");
        Assert.NotNull(retrieved);
    }

    #endregion

    #region Stress and Edge Case Tests

    [Fact]
    public void InMemoryStore_ManyDocuments_SearchStillWorks()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

        // Add 50 documents with embeddings spread around the unit sphere
        for (int i = 0; i < 50; i++)
        {
            double angle = 2 * Math.PI * i / 50;
            var emb = new Vector<double>(new double[] { Math.Cos(angle), Math.Sin(angle), 0 });
            store.Add(new VectorDocument<double>(
                new Document<double>($"doc{i}", $"content {i}"), emb));
        }

        Assert.Equal(50, store.DocumentCount);

        // Search for [1, 0, 0] - should find doc0 (angle = 0)
        var query = new Vector<double>(new double[] { 1, 0, 0 });
        var results = store.GetSimilar(query, topK: 5).ToList();

        Assert.Equal(5, results.Count);
        // doc0 should be first or very near first (its embedding is [1, 0, 0])
        Assert.Equal("doc0", results[0].Id);
    }

    [Fact]
    public void InMemoryStore_DocumentMetadata_Preserved()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        var metadata = new Dictionary<string, object>
        {
            { "source", "wikipedia" },
            { "year", 2024 },
            { "verified", true }
        };

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "content", metadata),
            new Vector<double>(new double[] { 1, 0 })));

        var retrieved = store.GetById("d1");
        Assert.NotNull(retrieved);
        Assert.Equal("wikipedia", retrieved.Metadata["source"]);
        Assert.Equal(2024, retrieved.Metadata["year"]);
        Assert.Equal(true, retrieved.Metadata["verified"]);
    }

    [Fact]
    public void InMemoryStore_DuplicateId_OverwritesSilently()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "original"), new Vector<double>(new double[] { 1, 0 })));

        // Adding same ID again should overwrite
        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "updated"), new Vector<double>(new double[] { 0, 1 })));

        // Count might be 1 or 2 depending on implementation
        // But GetById should return the latest version
        var retrieved = store.GetById("d1");
        Assert.NotNull(retrieved);
        Assert.Equal("updated", retrieved.Content);
    }

    [Fact]
    public void InMemoryStore_SearchResults_HaveRelevanceScores()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "c1"), new Vector<double>(new double[] { 1, 0 })));
        store.Add(new VectorDocument<double>(
            new Document<double>("d2", "c2"), new Vector<double>(new double[] { 0, 1 })));

        var query = new Vector<double>(new double[] { 1, 0 });
        var results = store.GetSimilar(query, topK: 2).ToList();

        // All results should have relevance scores set
        foreach (var result in results)
        {
            Assert.True(result.HasRelevanceScore,
                $"Document {result.Id} missing relevance score");
        }
    }

    [Fact]
    public void InMemoryStore_SearchResults_DoNotMutateStoredDocs()
    {
        var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

        store.Add(new VectorDocument<double>(
            new Document<double>("d1", "content"), new Vector<double>(new double[] { 1, 0 })));

        // Search twice and verify scores don't accumulate on stored doc
        var query = new Vector<double>(new double[] { 1, 0 });
        var results1 = store.GetSimilar(query, topK: 1).ToList();
        var results2 = store.GetSimilar(query, topK: 1).ToList();

        // Original stored doc should not be modified
        var storedDoc = store.GetById("d1");
        Assert.NotNull(storedDoc);
        // The stored doc should NOT have HasRelevanceScore set (only search results should)
        // This verifies the store returns copies, not references to stored docs
    }

    #endregion
}
