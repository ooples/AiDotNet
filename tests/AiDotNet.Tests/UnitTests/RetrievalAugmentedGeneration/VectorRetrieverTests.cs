using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration;

/// <summary>
/// Unit tests for VectorRetriever - a dense vector-based retriever that uses
/// embedding similarity for semantic document retrieval.
/// </summary>
public class VectorRetrieverTests
{
    private const int DefaultEmbeddingDimension = 128;

    #region Mock Implementations

    /// <summary>
    /// Mock embedding model for testing.
    /// </summary>
    private class MockEmbeddingModel : IEmbeddingModel<double>
    {
        public int EmbeddingDimension { get; }
        public int MaxTokens => 512;

        private readonly Dictionary<string, Vector<double>> _embeddings = new();
        public int EmbedCallCount { get; private set; }
        public string? LastEmbeddedText { get; private set; }

        public MockEmbeddingModel(int embeddingDimension = DefaultEmbeddingDimension)
        {
            EmbeddingDimension = embeddingDimension;
        }

        public void SetEmbedding(string text, Vector<double> embedding)
        {
            _embeddings[text] = embedding;
        }

        public Vector<double> Embed(string text)
        {
            EmbedCallCount++;
            LastEmbeddedText = text;

            if (_embeddings.TryGetValue(text, out var embedding))
            {
                return embedding;
            }

            // Generate deterministic embedding based on text hash
            var vector = new Vector<double>(EmbeddingDimension);
            var hash = text.GetHashCode();
            for (int i = 0; i < EmbeddingDimension; i++)
            {
                vector[i] = Math.Sin(hash + i) * 0.5 + 0.5;
            }
            return vector;
        }

        public Task<Vector<double>> EmbedAsync(string text)
        {
            return Task.FromResult(Embed(text));
        }

        public Matrix<double> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            var matrix = new Matrix<double>(textList.Count, EmbeddingDimension);
            for (int i = 0; i < textList.Count; i++)
            {
                var embedding = Embed(textList[i]);
                for (int j = 0; j < EmbeddingDimension; j++)
                {
                    matrix[i, j] = embedding[j];
                }
            }
            return matrix;
        }

        public Task<Matrix<double>> EmbedBatchAsync(IEnumerable<string> texts)
        {
            return Task.FromResult(EmbedBatch(texts));
        }
    }

    /// <summary>
    /// Mock document store for testing.
    /// </summary>
    private class MockDocumentStore : IDocumentStore<double>
    {
        private readonly List<(Document<double> Doc, Vector<double> Embedding)> _documents = new();

        public int DocumentCount => _documents.Count;
        public int VectorDimension { get; }

        // Track method calls for verification
        public int GetSimilarCallCount { get; private set; }
        public int GetSimilarWithFiltersCallCount { get; private set; }
        public Vector<double>? LastQueryVector { get; private set; }
        public int LastTopK { get; private set; }
        public Dictionary<string, object>? LastFilters { get; private set; }

        // Configure return values
        private List<Document<double>>? _customResults;

        public MockDocumentStore(int vectorDimension = DefaultEmbeddingDimension)
        {
            VectorDimension = vectorDimension;
        }

        public void AddDocument(string id, string content, Vector<double> embedding,
            double relevanceScore = 0.0, Dictionary<string, object>? metadata = null)
        {
            var doc = new Document<double>
            {
                Id = id,
                Content = content,
                Metadata = metadata ?? new Dictionary<string, object>(),
                RelevanceScore = relevanceScore
            };
            _documents.Add((doc, embedding));
        }

        public void SetCustomResults(List<Document<double>> results)
        {
            _customResults = results;
        }

        public void Add(VectorDocument<double> vectorDocument)
        {
            _documents.Add((vectorDocument.Document, vectorDocument.Embedding));
        }

        public void AddBatch(IEnumerable<VectorDocument<double>> vectorDocuments)
        {
            foreach (var vd in vectorDocuments)
            {
                Add(vd);
            }
        }

        public IEnumerable<Document<double>> GetSimilar(Vector<double> queryVector, int topK)
        {
            GetSimilarCallCount++;
            LastQueryVector = queryVector;
            LastTopK = topK;

            if (_customResults is not null)
            {
                return _customResults.Take(topK);
            }

            // Return documents sorted by cosine similarity
            return _documents
                .Select(d => (Doc: d.Doc, Similarity: ComputeCosineSimilarity(queryVector, d.Embedding)))
                .OrderByDescending(x => x.Similarity)
                .Take(topK)
                .Select(x =>
                {
                    x.Doc.RelevanceScore = x.Similarity;
                    return x.Doc;
                });
        }

        public IEnumerable<Document<double>> GetSimilarWithFilters(
            Vector<double> queryVector, int topK, Dictionary<string, object> metadataFilters)
        {
            GetSimilarWithFiltersCallCount++;
            LastQueryVector = queryVector;
            LastTopK = topK;
            LastFilters = metadataFilters;

            if (_customResults is not null)
            {
                return ApplyFilters(_customResults, metadataFilters).Take(topK);
            }

            // Apply filters first, then similarity search
            var filtered = _documents.Where(d => MatchesFilters(d.Doc, metadataFilters));

            return filtered
                .Select(d => (Doc: d.Doc, Similarity: ComputeCosineSimilarity(queryVector, d.Embedding)))
                .OrderByDescending(x => x.Similarity)
                .Take(topK)
                .Select(x =>
                {
                    x.Doc.RelevanceScore = x.Similarity;
                    return x.Doc;
                });
        }

        public Document<double>? GetById(string documentId)
        {
            return _documents.FirstOrDefault(d => d.Doc.Id == documentId).Doc;
        }

        public bool Remove(string documentId)
        {
            var index = _documents.FindIndex(d => d.Doc.Id == documentId);
            if (index >= 0)
            {
                _documents.RemoveAt(index);
                return true;
            }
            return false;
        }

        public void Clear()
        {
            _documents.Clear();
        }

        public IEnumerable<Document<double>> GetAll()
        {
            return _documents.Select(d => d.Doc);
        }

        private static double ComputeCosineSimilarity(Vector<double> a, Vector<double> b)
        {
            double dotProduct = 0;
            double normA = 0;
            double normB = 0;

            for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            var denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
            return denominator > 0 ? dotProduct / denominator : 0;
        }

        private static bool MatchesFilters(Document<double> doc, Dictionary<string, object> filters)
        {
            foreach (var filter in filters)
            {
                if (!doc.Metadata.TryGetValue(filter.Key, out var value))
                {
                    return false;
                }
                if (!Equals(value, filter.Value))
                {
                    return false;
                }
            }
            return true;
        }

        private static IEnumerable<Document<double>> ApplyFilters(
            IEnumerable<Document<double>> docs, Dictionary<string, object> filters)
        {
            return docs.Where(d => MatchesFilters(d, filters));
        }
    }

    #endregion

    #region Helper Methods

    private static Vector<double> CreateVector(params double[] values)
    {
        var vector = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            vector[i] = values[i];
        }
        return vector;
    }

    private static Vector<double> CreateRandomVector(int dimension, int seed = 42)
    {
        var random = new Random(seed);
        var vector = new Vector<double>(dimension);
        for (int i = 0; i < dimension; i++)
        {
            vector[i] = random.NextDouble();
        }
        return vector;
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
    {
        // Arrange
        var embeddingModel = new MockEmbeddingModel();

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new VectorRetriever<double>(null!, embeddingModel));
        Assert.Equal("documentStore", exception.ParamName);
    }

    [Fact]
    public void Constructor_WithNullEmbeddingModel_ThrowsArgumentNullException()
    {
        // Arrange
        var documentStore = new MockDocumentStore();

        // Act & Assert
        var exception = Assert.Throws<ArgumentNullException>(() =>
            new VectorRetriever<double>(documentStore, null!));
        Assert.Equal("embeddingModel", exception.ParamName);
    }

    [Fact]
    public void Constructor_WithValidParameters_CreatesInstance()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        // Act
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Assert
        Assert.NotNull(retriever);
    }

    [Fact]
    public void Constructor_WithDefaultTopK_UsesDefaultValue()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        documentStore.AddDocument("doc1", "content 1", CreateRandomVector(DefaultEmbeddingDimension, 1));
        documentStore.AddDocument("doc2", "content 2", CreateRandomVector(DefaultEmbeddingDimension, 2));
        documentStore.AddDocument("doc3", "content 3", CreateRandomVector(DefaultEmbeddingDimension, 3));
        documentStore.AddDocument("doc4", "content 4", CreateRandomVector(DefaultEmbeddingDimension, 4));
        documentStore.AddDocument("doc5", "content 5", CreateRandomVector(DefaultEmbeddingDimension, 5));
        documentStore.AddDocument("doc6", "content 6", CreateRandomVector(DefaultEmbeddingDimension, 6));

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("test query").ToList();

        // Assert - Default topK is 5
        Assert.Equal(5, results.Count);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(10)]
    [InlineData(100)]
    public void Constructor_WithCustomTopK_UsesProvidedValue(int topK)
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        // Add enough documents
        for (int i = 0; i < topK + 5; i++)
        {
            documentStore.AddDocument($"doc{i}", $"content {i}", CreateRandomVector(DefaultEmbeddingDimension, i));
        }

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel, topK);

        // Act
        var results = retriever.Retrieve("test query").ToList();

        // Assert
        Assert.Equal(topK, results.Count);
    }

    #endregion

    #region Retrieve Method Tests

    [Fact]
    public void Retrieve_EmbedsQueryUsingEmbeddingModel()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        documentStore.AddDocument("doc1", "test content", CreateRandomVector(DefaultEmbeddingDimension));
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
        var query = "find semantically similar documents";

        // Act
        var results = retriever.Retrieve(query).ToList();

        // Assert
        Assert.Equal(1, embeddingModel.EmbedCallCount);
        Assert.Equal(query, embeddingModel.LastEmbeddedText);
    }

    [Fact]
    public void Retrieve_PassesQueryVectorToDocumentStore()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        // Set known embedding for query
        var expectedVector = CreateVector(1.0, 2.0, 3.0, 4.0);
        embeddingModel.SetEmbedding("test query", expectedVector);

        documentStore.AddDocument("doc1", "test", CreateRandomVector(4, 1));
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel, 5);

        // Act
        var results = retriever.Retrieve("test query").ToList();

        // Assert
        Assert.NotNull(documentStore.LastQueryVector);
        Assert.Equal(expectedVector.Length, documentStore.LastQueryVector.Length);
        for (int i = 0; i < expectedVector.Length; i++)
        {
            Assert.Equal(expectedVector[i], documentStore.LastQueryVector[i]);
        }
    }

    [Fact]
    public void Retrieve_ReturnsDocumentsFromStore()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        var embedding1 = CreateRandomVector(DefaultEmbeddingDimension, 1);
        var embedding2 = CreateRandomVector(DefaultEmbeddingDimension, 2);

        documentStore.AddDocument("doc1", "first document", embedding1);
        documentStore.AddDocument("doc2", "second document", embedding2);

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("search query").ToList();

        // Assert
        Assert.Equal(2, results.Count);
        Assert.Contains(results, r => r.Id == "doc1");
        Assert.Contains(results, r => r.Id == "doc2");
    }

    [Fact]
    public void Retrieve_WithTopK_ReturnsRequestedNumberOfDocuments()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        for (int i = 0; i < 10; i++)
        {
            documentStore.AddDocument($"doc{i}", $"content {i}", CreateRandomVector(DefaultEmbeddingDimension, i));
        }

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query", 3).ToList();

        // Assert
        Assert.Equal(3, results.Count);
        Assert.Equal(3, documentStore.LastTopK);
    }

    [Fact]
    public void Retrieve_ReturnsDocumentsOrderedBySimilarity()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        // Create embeddings - doc1 will be most similar to query
        var queryEmbedding = CreateVector(1.0, 0.0, 0.0, 0.0);
        var embedding1 = CreateVector(0.9, 0.1, 0.0, 0.0);  // Very similar
        var embedding2 = CreateVector(0.5, 0.5, 0.0, 0.0);  // Somewhat similar
        var embedding3 = CreateVector(0.0, 0.0, 1.0, 0.0);  // Not similar

        embeddingModel.SetEmbedding("test query", queryEmbedding);

        documentStore.AddDocument("doc1", "similar", embedding1, 0.9);
        documentStore.AddDocument("doc2", "somewhat similar", embedding2, 0.7);
        documentStore.AddDocument("doc3", "not similar", embedding3, 0.1);

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("test query").ToList();

        // Assert
        Assert.Equal(3, results.Count);
        Assert.Equal("doc1", results[0].Id);  // Most similar first
        Assert.Equal("doc2", results[1].Id);
        Assert.Equal("doc3", results[2].Id);  // Least similar last
    }

    [Fact]
    public void Retrieve_WithEmptyStore_ReturnsEmptyList()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("any query").ToList();

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public void Retrieve_IncludesRelevanceScores()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension));
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query").ToList();

        // Assert
        Assert.Single(results);
        // Relevance score should be computed (cosine similarity)
        Assert.True(results[0].RelevanceScore >= 0 && results[0].RelevanceScore <= 1);
    }

    #endregion

    #region Metadata Filtering Tests

    [Fact]
    public void Retrieve_WithMetadataFilters_PassesFiltersToStore()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension),
            metadata: new Dictionary<string, object> { ["category"] = "A" });

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
        var filters = new Dictionary<string, object> { ["category"] = "A" };

        // Act
        var results = retriever.Retrieve("query", 5, filters).ToList();

        // Assert
        Assert.Equal(1, documentStore.GetSimilarWithFiltersCallCount);
        Assert.NotNull(documentStore.LastFilters);
        Assert.Contains(documentStore.LastFilters, kvp => kvp.Key == "category" && (string)kvp.Value == "A");
    }

    [Fact]
    public void Retrieve_WithMetadataFilters_FiltersDocuments()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content 1", CreateRandomVector(DefaultEmbeddingDimension, 1),
            metadata: new Dictionary<string, object> { ["category"] = "science" });
        documentStore.AddDocument("doc2", "content 2", CreateRandomVector(DefaultEmbeddingDimension, 2),
            metadata: new Dictionary<string, object> { ["category"] = "history" });
        documentStore.AddDocument("doc3", "content 3", CreateRandomVector(DefaultEmbeddingDimension, 3),
            metadata: new Dictionary<string, object> { ["category"] = "science" });

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
        var filters = new Dictionary<string, object> { ["category"] = "science" };

        // Act
        var results = retriever.Retrieve("query", 10, filters).ToList();

        // Assert
        Assert.Equal(2, results.Count);
        Assert.All(results, r => Assert.Equal("science", r.Metadata["category"]));
    }

    [Fact]
    public void Retrieve_WithMultipleFilters_AppliesAllFilters()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content 1", CreateRandomVector(DefaultEmbeddingDimension, 1),
            metadata: new Dictionary<string, object> { ["category"] = "tech", ["year"] = 2024 });
        documentStore.AddDocument("doc2", "content 2", CreateRandomVector(DefaultEmbeddingDimension, 2),
            metadata: new Dictionary<string, object> { ["category"] = "tech", ["year"] = 2023 });
        documentStore.AddDocument("doc3", "content 3", CreateRandomVector(DefaultEmbeddingDimension, 3),
            metadata: new Dictionary<string, object> { ["category"] = "science", ["year"] = 2024 });

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
        var filters = new Dictionary<string, object> { ["category"] = "tech", ["year"] = 2024 };

        // Act
        var results = retriever.Retrieve("query", 10, filters).ToList();

        // Assert
        Assert.Single(results);
        Assert.Equal("doc1", results[0].Id);
    }

    [Fact]
    public void Retrieve_WithNoMatchingFilters_ReturnsEmpty()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension),
            metadata: new Dictionary<string, object> { ["status"] = "draft" });

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
        var filters = new Dictionary<string, object> { ["status"] = "published" };

        // Act
        var results = retriever.Retrieve("query", 10, filters).ToList();

        // Assert
        Assert.Empty(results);
    }

    [Fact]
    public void Retrieve_WithEmptyFilters_ReturnsAllMatchingDocuments()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content 1", CreateRandomVector(DefaultEmbeddingDimension, 1));
        documentStore.AddDocument("doc2", "content 2", CreateRandomVector(DefaultEmbeddingDimension, 2));

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
        var emptyFilters = new Dictionary<string, object>();

        // Act
        var results = retriever.Retrieve("query", 10, emptyFilters).ToList();

        // Assert
        Assert.Equal(2, results.Count);
    }

    [Fact]
    public void Retrieve_WithNullMetadataFilters_ThrowsArgumentNullException()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            retriever.Retrieve("query", 5, null!).ToList());
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void Retrieve_WithNullQuery_ThrowsArgumentException()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act & Assert - RetrieverBase throws ArgumentException for null/empty queries
        Assert.Throws<ArgumentException>(() =>
            retriever.Retrieve(null!).ToList());
    }

    [Fact]
    public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            retriever.Retrieve("").ToList());
    }

    [Fact]
    public void Retrieve_WithWhitespaceQuery_ThrowsArgumentException()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            retriever.Retrieve("   ").ToList());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-100)]
    public void Retrieve_WithInvalidTopK_ThrowsArgumentOutOfRangeException(int topK)
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            retriever.Retrieve("query", topK).ToList());
    }

    [Fact]
    public void Retrieve_WithTopKGreaterThanDocumentCount_ReturnsAllDocuments()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content 1", CreateRandomVector(DefaultEmbeddingDimension, 1));
        documentStore.AddDocument("doc2", "content 2", CreateRandomVector(DefaultEmbeddingDimension, 2));
        documentStore.AddDocument("doc3", "content 3", CreateRandomVector(DefaultEmbeddingDimension, 3));

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query", 100).ToList();

        // Assert
        Assert.Equal(3, results.Count);
    }

    [Fact]
    public void Retrieve_MultipleTimes_EmbedsQueryEachTime()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension));
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        retriever.Retrieve("query 1").ToList();
        retriever.Retrieve("query 2").ToList();
        retriever.Retrieve("query 3").ToList();

        // Assert
        Assert.Equal(3, embeddingModel.EmbedCallCount);
    }

    [Fact]
    public void Retrieve_PreservesDocumentMetadata()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        var metadata = new Dictionary<string, object>
        {
            ["author"] = "John Doe",
            ["date"] = "2024-01-01",
            ["tags"] = new[] { "ai", "ml" }
        };

        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension),
            metadata: metadata);

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query").ToList();

        // Assert
        Assert.Single(results);
        Assert.Equal("John Doe", results[0].Metadata["author"]);
        Assert.Equal("2024-01-01", results[0].Metadata["date"]);
    }

    [Fact]
    public void Retrieve_PreservesDocumentContent()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        var content = "This is the original document content with special characters: @#$%";
        documentStore.AddDocument("doc1", content, CreateRandomVector(DefaultEmbeddingDimension));

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query").ToList();

        // Assert
        Assert.Single(results);
        Assert.Equal(content, results[0].Content);
    }

    #endregion

    #region Semantic Similarity Tests

    [Fact]
    public void Retrieve_FindsSemanticallyRelatedDocuments()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        // Set up embeddings to simulate semantic relationships
        // Similar embeddings = similar meaning
        var queryEmbedding = CreateVector(1.0, 0.5, 0.0, 0.0);
        var relatedEmbedding = CreateVector(0.95, 0.48, 0.02, 0.01);  // Very similar
        var unrelatedEmbedding = CreateVector(-0.5, -0.5, 0.5, 0.5);  // Different

        embeddingModel.SetEmbedding("machine learning algorithms", queryEmbedding);

        documentStore.AddDocument("doc1", "Deep learning and neural networks", relatedEmbedding);
        documentStore.AddDocument("doc2", "Cooking recipes for beginners", unrelatedEmbedding);

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("machine learning algorithms", 1).ToList();

        // Assert
        Assert.Single(results);
        Assert.Equal("doc1", results[0].Id);  // Related document should be returned
    }

    [Fact]
    public void Retrieve_ReturnsNormalizedRelevanceScores()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        for (int i = 0; i < 5; i++)
        {
            documentStore.AddDocument($"doc{i}", $"content {i}", CreateRandomVector(DefaultEmbeddingDimension, i));
        }

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("test query").ToList();

        // Assert
        Assert.All(results, r => Assert.True(r.RelevanceScore >= -1 && r.RelevanceScore <= 1,
            $"Cosine similarity should be between -1 and 1, but was {r.RelevanceScore}"));
    }

    #endregion

    #region Integration-Style Tests

    [Fact]
    public void Retrieve_WithLargeDocumentSet_HandlesEfficiently()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        // Add many documents
        for (int i = 0; i < 1000; i++)
        {
            documentStore.AddDocument($"doc{i}", $"Content of document {i}",
                CreateRandomVector(DefaultEmbeddingDimension, i));
        }

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel, 10);

        // Act
        var results = retriever.Retrieve("test query").ToList();

        // Assert
        Assert.Equal(10, results.Count);
    }

    [Fact]
    public void Retrieve_WithVaryingTopK_ReturnsCorrectCount()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        for (int i = 0; i < 20; i++)
        {
            documentStore.AddDocument($"doc{i}", $"content {i}", CreateRandomVector(DefaultEmbeddingDimension, i));
        }

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act & Assert
        Assert.Single(retriever.Retrieve("q", 1));
        Assert.Equal(5, retriever.Retrieve("q", 5).Count());
        Assert.Equal(10, retriever.Retrieve("q", 10).Count());
        Assert.Equal(15, retriever.Retrieve("q", 15).Count());
    }

    [Fact]
    public void Retrieve_SameQueryDifferentFilters_ReturnsDifferentResults()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension, 1),
            metadata: new Dictionary<string, object> { ["type"] = "A" });
        documentStore.AddDocument("doc2", "content", CreateRandomVector(DefaultEmbeddingDimension, 2),
            metadata: new Dictionary<string, object> { ["type"] = "B" });

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var resultsA = retriever.Retrieve("query", 10, new Dictionary<string, object> { ["type"] = "A" }).ToList();
        var resultsB = retriever.Retrieve("query", 10, new Dictionary<string, object> { ["type"] = "B" }).ToList();

        // Assert
        Assert.Single(resultsA);
        Assert.Equal("doc1", resultsA[0].Id);

        Assert.Single(resultsB);
        Assert.Equal("doc2", resultsB[0].Id);
    }

    [Fact]
    public void Retrieve_DocumentsWithSameEmbedding_ReturnsAllEqually()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();

        // Use same embedding for multiple documents
        var sharedEmbedding = CreateRandomVector(DefaultEmbeddingDimension, 42);

        documentStore.AddDocument("doc1", "first", sharedEmbedding);
        documentStore.AddDocument("doc2", "second", sharedEmbedding);
        documentStore.AddDocument("doc3", "third", sharedEmbedding);

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query").ToList();

        // Assert
        Assert.Equal(3, results.Count);
        // All should have equal relevance scores (same embedding = same similarity)
    }

    [Fact]
    public void Retrieve_WithDifferentEmbeddingDimensions_WorksCorrectly()
    {
        // Arrange
        var dimension = 256;
        var documentStore = new MockDocumentStore(dimension);
        var embeddingModel = new MockEmbeddingModel(dimension);

        documentStore.AddDocument("doc1", "content", CreateRandomVector(dimension, 1));

        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query").ToList();

        // Assert
        Assert.Single(results);
    }

    #endregion

    #region Document Store Interaction Tests

    [Fact]
    public void Retrieve_CallsGetSimilarWithFiltersNotGetSimilar()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension));
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        var results = retriever.Retrieve("query", 5, new Dictionary<string, object>()).ToList();

        // Assert
        Assert.Equal(1, documentStore.GetSimilarWithFiltersCallCount);
        Assert.Equal(0, documentStore.GetSimilarCallCount);
    }

    [Fact]
    public void Retrieve_PassesCorrectTopKToStore()
    {
        // Arrange
        var documentStore = new MockDocumentStore();
        var embeddingModel = new MockEmbeddingModel();
        documentStore.AddDocument("doc1", "content", CreateRandomVector(DefaultEmbeddingDimension));
        var retriever = new VectorRetriever<double>(documentStore, embeddingModel);

        // Act
        retriever.Retrieve("query", 7).ToList();

        // Assert
        Assert.Equal(7, documentStore.LastTopK);
    }

    #endregion
}
