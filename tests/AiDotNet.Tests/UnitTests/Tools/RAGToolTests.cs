using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tools;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.Tools;

/// <summary>
/// Unit tests for the RAGTool class.
/// </summary>
public class RAGToolTests
{
    #region PR #756 Bug Fix Tests - Parameter Validation

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsOnNullRetriever()
    {
        var mockGenerator = new MockGenerator<double>();

        Assert.Throws<ArgumentNullException>(() =>
            new RAGTool<double>(null!, null, mockGenerator));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsOnNullGenerator()
    {
        var mockRetriever = new MockRetriever<double>();

        Assert.Throws<ArgumentNullException>(() =>
            new RAGTool<double>(mockRetriever, null, null!));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsOnZeroTopK()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 0));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsOnNegativeTopK()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RAGTool<double>(mockRetriever, null, mockGenerator, topK: -1));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsOnExcessiveTopK()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 101));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsOnZeroTopKAfterRerank()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 10, topKAfterRerank: 0));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsOnNegativeTopKAfterRerank()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 10, topKAfterRerank: -1));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsWhenTopKAfterRerankExceedsTopK()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 5, topKAfterRerank: 10));
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_AcceptsMaxAllowedTopK()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 100);

        Assert.NotNull(tool);
        Assert.Equal("RAG", tool.Name);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_AcceptsMinAllowedTopK()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        // topKAfterRerank must be null or <= topK
        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 1, topKAfterRerank: 1);

        Assert.NotNull(tool);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_AcceptsNullReranker()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator);

        Assert.NotNull(tool);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_AcceptsNullTopKAfterRerank()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();

        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator, topK: 10, topKAfterRerank: null);

        Assert.NotNull(tool);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_EmptyInput_ReturnsError()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();
        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator);

        var result = tool.Execute("");

        Assert.Contains("Error", result);
        Assert.Contains("empty", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_WhitespaceInput_ReturnsError()
    {
        var mockRetriever = new MockRetriever<double>();
        var mockGenerator = new MockGenerator<double>();
        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator);

        var result = tool.Execute("   ");

        Assert.Contains("Error", result);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_ValidQuery_ReturnsAnswer()
    {
        var mockRetriever = new MockRetriever<double>();
        mockRetriever.AddDocument("doc1", "Test content about AI");
        var mockGenerator = new MockGenerator<double>();
        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator);

        var result = tool.Execute("What is AI?");

        Assert.Contains("Answer", result);
    }

    [Fact(Timeout = 60000)]
    public async Task Execute_NoDocumentsFound_ReturnsNoDocumentsMessage()
    {
        var mockRetriever = new MockRetriever<double>(returnEmpty: true);
        var mockGenerator = new MockGenerator<double>();
        var tool = new RAGTool<double>(mockRetriever, null, mockGenerator);

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

        public int DefaultTopK => 10;

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

    /// <summary>
    /// Mock generator for testing purposes.
    /// </summary>
    private class MockGenerator<T> : IGenerator<T>
    {
        public int MaxContextTokens => 4096;
        public int MaxGenerationTokens => 1024;

        public string Generate(string prompt)
        {
            return $"Generated answer for: {prompt}";
        }

        public GroundedAnswer<T> GenerateGrounded(string query, IEnumerable<Document<T>> context)
        {
            return new GroundedAnswer<T>
            {
                Answer = $"Generated grounded answer for: {query}",
                Citations = context.Select(d => d.Id).ToList(),
                ConfidenceScore = 0.95
            };
        }
    }
}
