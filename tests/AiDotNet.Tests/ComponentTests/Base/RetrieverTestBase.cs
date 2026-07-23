using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.ComponentTests.Base;

/// <summary>
/// Base test class for IRetriever&lt;double&gt; implementations.
/// Tests retrieval invariants: valid queries return results, empty queries are handled
/// gracefully, DefaultTopK is positive, and result count respects the TopK limit.
/// </summary>
public abstract class RetrieverTestBase
{
    /// <summary>
    /// Creates the retriever under test. The retriever should be connected to
    /// a document store that already contains documents (use <see cref="CreateDocumentStoreWithDocuments"/>).
    /// </summary>
    protected abstract IRetriever<double> CreateRetriever();

    /// <summary>
    /// Creates a document store pre-populated with test documents so the retriever has something to search.
    /// </summary>
    protected abstract IDocumentStore<double> CreateDocumentStoreWithDocuments();

    /// <summary>
    /// The number of test documents that <see cref="CreateDocumentStoreWithDocuments"/> adds.
    /// Derived classes should override if they add a different number.
    /// </summary>
    protected virtual int TestDocumentCount => 5;

    /// <summary>
    /// Creates a set of test documents with dummy content and embeddings.
    /// </summary>
    protected static List<VectorDocument<double>> CreateTestVectorDocuments(int count = 5, int embeddingDimension = 8)
    {
        var documents = new List<VectorDocument<double>>();
        for (int i = 0; i < count; i++)
        {
            var doc = new Document<double>($"doc-{i}", $"Test document number {i} about topic {i % 3}");
            doc.Metadata["category"] = $"category-{i % 3}";
            doc.Metadata["index"] = i;

            var embeddingValues = new double[embeddingDimension];
            for (int j = 0; j < embeddingDimension; j++)
            {
                embeddingValues[j] = (i + 1.0) / (j + 1.0) * 0.1;
            }
            var embedding = new Vector<double>(embeddingValues);

            documents.Add(new VectorDocument<double>(doc, embedding));
        }

        return documents;
    }

    // =====================================================
    // INVARIANT: Basic retrieval should return results
    // A retriever connected to a populated document store
    // must return at least one result for a valid query.
    // =====================================================

    [Fact]
    public void Retrieve_WithValidQuery_ReturnsResults()
    {
        var retriever = CreateRetriever();

        var results = retriever.Retrieve("test document about topic");

        Assert.NotNull(results);
        var resultList = results.ToList();
        Assert.True(resultList.Count > 0,
            "Retriever should return at least one result for a valid query against a populated store.");
    }

    // =====================================================
    // INVARIANT: Empty query should not crash
    // An empty string query must not throw; it may return
    // empty results or default results, but must not fail.
    // =====================================================

    [Fact]
    public void Retrieve_WithEmptyQuery_HandlesGracefully()
    {
        var retriever = CreateRetriever();

        var exception = Record.Exception(() =>
        {
            var results = retriever.Retrieve(string.Empty);
            // Force enumeration to ensure no deferred exceptions
            _ = results?.ToList();
        });

        Assert.Null(exception);
    }

    // =====================================================
    // INVARIANT: DefaultTopK must be positive
    // A non-positive TopK is nonsensical for retrieval.
    // =====================================================

    [Fact]
    public void DefaultTopK_ShouldBePositive()
    {
        var retriever = CreateRetriever();

        Assert.True(retriever.DefaultTopK > 0,
            $"DefaultTopK should be > 0 but was {retriever.DefaultTopK}.");
    }

    // =====================================================
    // INVARIANT: Result count should not exceed TopK
    // When retrieving with a specific topK, the number of
    // results returned must not exceed that limit.
    // =====================================================

    [Fact]
    public void Retrieve_ResultCount_ShouldNotExceedTopK()
    {
        var retriever = CreateRetriever();
        int topK = 2;

        var results = retriever.Retrieve("test document", topK);

        Assert.NotNull(results);
        var resultList = results.ToList();
        Assert.True(resultList.Count <= topK,
            $"Retrieve returned {resultList.Count} results but topK was {topK}. " +
            "Result count must not exceed the requested topK limit.");
    }
}
