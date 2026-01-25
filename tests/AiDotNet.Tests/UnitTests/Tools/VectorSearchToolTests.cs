using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tools;
using Xunit;

namespace AiDotNetTests.UnitTests.Tools;

/// <summary>
/// Unit tests for the VectorSearchTool class.
/// </summary>
public class VectorSearchToolTests
{
    #region PR #756 Bug Fix Tests - Parameter Validation

    [Fact]
    public void Constructor_ThrowsOnNullRetriever()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new VectorSearchTool<double>(null!));
    }

    [Fact]
    public void Constructor_ThrowsOnZeroTopK()
    {
        var mockRetriever = new MockRetriever<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new VectorSearchTool<double>(mockRetriever, topK: 0));
    }

    [Fact]
    public void Constructor_ThrowsOnNegativeTopK()
    {
        var mockRetriever = new MockRetriever<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new VectorSearchTool<double>(mockRetriever, topK: -1));
    }

    [Fact]
    public void Constructor_ThrowsOnExcessiveTopK()
    {
        var mockRetriever = new MockRetriever<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new VectorSearchTool<double>(mockRetriever, topK: 101));
    }

    [Fact]
    public void Constructor_AcceptsMaxAllowedTopK()
    {
        var mockRetriever = new MockRetriever<double>();

        var tool = new VectorSearchTool<double>(mockRetriever, topK: 100);

        Assert.NotNull(tool);
        Assert.Equal("VectorSearch", tool.Name);
    }

    [Fact]
    public void Constructor_AcceptsMinAllowedTopK()
    {
        var mockRetriever = new MockRetriever<double>();

        var tool = new VectorSearchTool<double>(mockRetriever, topK: 1);

        Assert.NotNull(tool);
    }

    [Fact]
    public void Execute_EmptyInput_ReturnsError()
    {
        var mockRetriever = new MockRetriever<double>();
        var tool = new VectorSearchTool<double>(mockRetriever);

        var result = tool.Execute("");

        Assert.Contains("Error", result);
        Assert.Contains("empty", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Execute_WhitespaceInput_ReturnsError()
    {
        var mockRetriever = new MockRetriever<double>();
        var tool = new VectorSearchTool<double>(mockRetriever);

        var result = tool.Execute("   ");

        Assert.Contains("Error", result);
    }

    [Fact]
    public void Execute_ValidQuery_ReturnsResults()
    {
        var mockRetriever = new MockRetriever<double>();
        mockRetriever.AddDocument("doc1", "Test content about machine learning");
        var tool = new VectorSearchTool<double>(mockRetriever);

        var result = tool.Execute("machine learning");

        Assert.Contains("Test content", result);
    }

    [Fact]
    public void Execute_WithTopKParameter_RespectsLimit()
    {
        var mockRetriever = new MockRetriever<double>();
        mockRetriever.AddDocument("doc1", "Content 1");
        mockRetriever.AddDocument("doc2", "Content 2");
        var tool = new VectorSearchTool<double>(mockRetriever, topK: 5);

        var result = tool.Execute("test|topK=1");

        // Should only show 1 document
        Assert.Contains("Found 1", result);
    }

    [Fact]
    public void Execute_NoResults_ReturnsNoDocumentsMessage()
    {
        var mockRetriever = new MockRetriever<double>(returnEmpty: true);
        var tool = new VectorSearchTool<double>(mockRetriever);

        var result = tool.Execute("some query");

        Assert.Contains("No relevant documents", result);
    }

    #endregion

    /// <summary>
    /// Mock retriever for testing purposes.
    /// </summary>
    private class MockRetriever<T> : IRetriever<T>
    {
        private readonly List<Document<T>> _documents = [];
        private readonly bool _returnEmpty;

        public MockRetriever(bool returnEmpty = false)
        {
            _returnEmpty = returnEmpty;
        }

        public int DefaultTopK => 5;

        public void AddDocument(string id, string content)
        {
            _documents.Add(new Document<T>
            {
                Id = id,
                Content = content,
                Metadata = new Dictionary<string, object> { ["source"] = "test" }
            });
        }

        public IEnumerable<Document<T>> Retrieve(string query)
        {
            return Retrieve(query, DefaultTopK);
        }

        public IEnumerable<Document<T>> Retrieve(string query, int topK)
        {
            if (_returnEmpty)
                return Enumerable.Empty<Document<T>>();

            return _documents.Take(topK);
        }

        public IEnumerable<Document<T>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            return Retrieve(query, topK);
        }
    }
}
