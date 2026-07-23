using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.ComponentTests.Base;

/// <summary>
/// Base test class for IDocumentStore&lt;double&gt; implementations.
/// Tests document store invariants: adding documents increases count,
/// initial count is zero, search finds added documents, and clear resets count.
/// </summary>
public abstract class DocumentStoreTestBase
{
    /// <summary>
    /// Creates a fresh, empty document store for testing.
    /// </summary>
    protected abstract IDocumentStore<double> CreateDocumentStore();

    /// <summary>
    /// The embedding dimension used for test documents. Override if the store requires a specific dimension.
    /// </summary>
    protected virtual int EmbeddingDimension => 8;

    /// <summary>
    /// Creates a test VectorDocument with a dummy embedding.
    /// </summary>
    protected VectorDocument<double> CreateTestVectorDocument(string id, string content)
    {
        var doc = new Document<double>(id, content);
        var embeddingValues = new double[EmbeddingDimension];
        for (int i = 0; i < EmbeddingDimension; i++)
        {
            embeddingValues[i] = (id.GetHashCode() + i) * 0.01;
        }
        var embedding = new Vector<double>(embeddingValues);

        return new VectorDocument<double>(doc, embedding);
    }

    /// <summary>
    /// Creates a query vector for similarity search.
    /// </summary>
    protected Vector<double> CreateQueryVector(double seed = 1.0)
    {
        var values = new double[EmbeddingDimension];
        for (int i = 0; i < EmbeddingDimension; i++)
        {
            values[i] = seed / (i + 1.0);
        }
        return new Vector<double>(values);
    }

    // =====================================================
    // INVARIANT: Adding a document must increase count
    // After adding one document, DocumentCount should be 1
    // more than before.
    // =====================================================

    [Fact]
    public void AddDocument_IncreasesCount()
    {
        var store = CreateDocumentStore();
        int initialCount = store.DocumentCount;

        var vectorDoc = CreateTestVectorDocument("test-1", "Test document content");
        store.Add(vectorDoc);

        Assert.Equal(initialCount + 1, store.DocumentCount);
    }

    // =====================================================
    // INVARIANT: Fresh store should be empty
    // A newly created document store must start with zero documents.
    // =====================================================

    [Fact]
    public void DocumentCount_StartsAtZero()
    {
        var store = CreateDocumentStore();

        Assert.Equal(0, store.DocumentCount);
    }

    // =====================================================
    // INVARIANT: Search should find added documents
    // After adding a document, similarity search with a
    // related query vector should return at least one result.
    // =====================================================

    [Fact]
    public void Search_AfterAdd_FindsDocument()
    {
        var store = CreateDocumentStore();
        var vectorDoc = CreateTestVectorDocument("find-me", "Searchable document content about AI");
        store.Add(vectorDoc);

        var queryVector = CreateQueryVector();
        var results = store.GetSimilar(queryVector, 5);

        Assert.NotNull(results);
        var resultList = results.ToList();
        Assert.True(resultList.Count > 0,
            "Search should find at least one document after adding one to the store.");
    }

    // =====================================================
    // INVARIANT: Clear should reset count to zero
    // After clearing, the store must be empty.
    // =====================================================

    [Fact]
    public void Clear_ResetsCount()
    {
        var store = CreateDocumentStore();

        // Add some documents first
        store.Add(CreateTestVectorDocument("clear-1", "Document one"));
        store.Add(CreateTestVectorDocument("clear-2", "Document two"));
        Assert.True(store.DocumentCount > 0, "Store should have documents before clear.");

        store.Clear();

        Assert.Equal(0, store.DocumentCount);
    }
}
