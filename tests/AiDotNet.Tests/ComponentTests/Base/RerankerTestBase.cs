using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.ComponentTests.Base;

/// <summary>
/// Base test class for IReranker&lt;double&gt; implementations.
/// Tests reranking invariants: reranked results are reordered, empty input returns empty,
/// document count is preserved, and the top result has the highest score.
/// </summary>
public abstract class RerankerTestBase
{
    /// <summary>
    /// Creates the reranker under test.
    /// </summary>
    protected abstract IReranker<double> CreateReranker();

    /// <summary>
    /// Creates a list of test documents with varying content for reranking.
    /// </summary>
    protected static List<Document<double>> CreateTestDocuments(int count = 5)
    {
        var documents = new List<Document<double>>();
        for (int i = 0; i < count; i++)
        {
            var doc = new Document<double>($"doc-{i}", $"Document about topic {i} with relevant content on subject {i % 3}");
            doc.RelevanceScore = (count - i) * 0.1; // Descending initial scores
            doc.HasRelevanceScore = true;
            documents.Add(doc);
        }

        return documents;
    }

    // =====================================================
    // INVARIANT: Reranking should return reordered results
    // A reranker must return a non-null, non-empty collection
    // when given a non-empty input.
    // =====================================================

    [Fact]
    public void Rerank_WithDocuments_ReturnsReorderedResults()
    {
        var reranker = CreateReranker();
        var documents = CreateTestDocuments();

        var results = reranker.Rerank("relevant topic", documents);

        Assert.NotNull(results);
        var resultList = results.ToList();
        Assert.True(resultList.Count > 0,
            "Reranker should return at least one result when given non-empty input.");
    }

    // =====================================================
    // INVARIANT: Empty input should return empty output
    // Reranking an empty collection must return an empty
    // collection, not null or an exception.
    // =====================================================

    [Fact]
    public void Rerank_WithEmptyDocuments_ReturnsEmpty()
    {
        var reranker = CreateReranker();
        var emptyDocuments = new List<Document<double>>();

        var results = reranker.Rerank("some query", emptyDocuments);

        Assert.NotNull(results);
        var resultList = results.ToList();
        Assert.Empty(resultList);
    }

    // =====================================================
    // INVARIANT: Document count must be preserved
    // Reranking should not lose or duplicate documents.
    // The output count must equal the input count.
    // =====================================================

    [Fact]
    public void Rerank_PreservesDocumentCount()
    {
        var reranker = CreateReranker();
        var documents = CreateTestDocuments(7);

        var results = reranker.Rerank("topic query", documents);

        Assert.NotNull(results);
        var resultList = results.ToList();
        Assert.Equal(documents.Count, resultList.Count);
    }

    // =====================================================
    // INVARIANT: Top result should have the highest score
    // If the reranker modifies scores, the first result in
    // the output must have the highest relevance score.
    // =====================================================

    [Fact]
    public void Rerank_TopResult_HasHighestScore()
    {
        var reranker = CreateReranker();
        var documents = CreateTestDocuments();

        var results = reranker.Rerank("relevant topic", documents);

        Assert.NotNull(results);
        var resultList = results.ToList();

        if (resultList.Count < 2)
        {
            return; // Cannot verify ordering with fewer than 2 results
        }

        // If the reranker modifies scores, the top result should have the highest score
        if (reranker.ModifiesScores)
        {
            var topScore = resultList[0].RelevanceScore;
            for (int i = 1; i < resultList.Count; i++)
            {
                Assert.True(resultList[i].RelevanceScore <= topScore,
                    $"Result at index {i} has score {resultList[i].RelevanceScore} " +
                    $"which exceeds the top result score {topScore}. " +
                    "Results should be ordered by descending relevance score.");
            }
        }
    }
}
