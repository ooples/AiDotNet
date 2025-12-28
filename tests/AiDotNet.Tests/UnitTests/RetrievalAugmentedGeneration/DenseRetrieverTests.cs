using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for DenseRetriever which uses vector similarity search.
    /// </summary>
    public class DenseRetrieverTests
    {
        #region Test Helpers

        /// <summary>
        /// Mock embedding model for testing.
        /// </summary>
        private class MockEmbeddingModel : IEmbeddingModel<double>
        {
            public int EmbeddingDimension => 128;
            public int MaxTokens => 512;

            public Vector<double> Embed(string text)
            {
                // Simple embedding: use character sum as a deterministic embedding
                var embedding = new double[EmbeddingDimension];
                if (!string.IsNullOrEmpty(text))
                {
                    var charSum = text.Sum(c => (double)c);
                    for (int i = 0; i < EmbeddingDimension; i++)
                    {
                        embedding[i] = Math.Sin(charSum + i) * 0.5 + 0.5;
                    }
                }
                return new Vector<double>(embedding);
            }

            public Matrix<double> EmbedBatch(IEnumerable<string> texts)
            {
                var textList = texts.ToList();
                var data = new double[textList.Count, EmbeddingDimension];
                for (int row = 0; row < textList.Count; row++)
                {
                    var embedding = Embed(textList[row]);
                    for (int col = 0; col < EmbeddingDimension; col++)
                    {
                        data[row, col] = embedding[col];
                    }
                }
                return new Matrix<double>(data);
            }
        }

        /// <summary>
        /// Mock document store for testing.
        /// </summary>
        private class MockDocumentStore : IDocumentStore<double>
        {
            private readonly List<Document<double>> _documents = new();
            private readonly MockEmbeddingModel _embeddingModel = new();

            public int DocumentCount => _documents.Count;
            public int VectorDimension => 128;

            public void Add(VectorDocument<double> vectorDocument)
            {
                _documents.Add(vectorDocument.Document);
            }

            public void AddBatch(IEnumerable<VectorDocument<double>> vectorDocuments)
            {
                foreach (var vd in vectorDocuments)
                {
                    _documents.Add(vd.Document);
                }
            }

            public void AddDocument(Document<double> document)
            {
                _documents.Add(document);
            }

            public IEnumerable<Document<double>> GetAll() => _documents;

            public IEnumerable<Document<double>> GetSimilar(Vector<double> queryVector, int topK)
            {
                // Simple similarity based on vector distance
                var scored = _documents.Select(d =>
                {
                    var docVector = _embeddingModel.Embed(d.Content);
                    var similarity = CalculateCosineSimilarity(queryVector, docVector);
                    d.RelevanceScore = similarity;
                    d.HasRelevanceScore = true;
                    return d;
                }).OrderByDescending(d => d.RelevanceScore);

                return scored.Take(topK);
            }

            public IEnumerable<Document<double>> GetSimilarWithFilters(
                Vector<double> queryVector,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                var filtered = _documents.AsEnumerable();

                if (metadataFilters != null && metadataFilters.Count > 0)
                {
                    foreach (var filter in metadataFilters)
                    {
                        filtered = filtered.Where(d =>
                            d.Metadata != null &&
                            d.Metadata.TryGetValue(filter.Key, out var value) &&
                            Equals(value, filter.Value));
                    }
                }

                var scored = filtered.Select(d =>
                {
                    var docVector = _embeddingModel.Embed(d.Content);
                    var similarity = CalculateCosineSimilarity(queryVector, docVector);
                    d.RelevanceScore = similarity;
                    d.HasRelevanceScore = true;
                    return d;
                }).OrderByDescending(d => d.RelevanceScore);

                return scored.Take(topK);
            }

            private static double CalculateCosineSimilarity(Vector<double> a, Vector<double> b)
            {
                if (a.Length != b.Length) return 0;

                double dotProduct = 0, normA = 0, normB = 0;
                for (int i = 0; i < a.Length; i++)
                {
                    dotProduct += a[i] * b[i];
                    normA += a[i] * a[i];
                    normB += b[i] * b[i];
                }

                var denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
                return denominator > 0 ? dotProduct / denominator : 0;
            }

            public Document<double>? GetById(string documentId)
                => _documents.FirstOrDefault(d => d.Id == documentId);

            public bool Remove(string documentId)
            {
                var doc = _documents.FirstOrDefault(d => d.Id == documentId);
                if (doc != null)
                {
                    _documents.Remove(doc);
                    return true;
                }
                return false;
            }

            public void Clear() => _documents.Clear();
        }

        private MockDocumentStore CreateStoreWithDocuments(params (string id, string content)[] docs)
        {
            var store = new MockDocumentStore();
            foreach (var (id, content) in docs)
            {
                store.AddDocument(new Document<double>(id, content));
            }
            return store;
        }

        private MockDocumentStore CreateStoreWithDocumentsAndMetadata(
            params (string id, string content, Dictionary<string, object> metadata)[] docs)
        {
            var store = new MockDocumentStore();
            foreach (var (id, content, metadata) in docs)
            {
                store.AddDocument(new Document<double>(id, content, metadata));
            }
            return store;
        }

        #endregion

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();

            // Act
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomTopK_SetsCorrectly()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();

            // Act
            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_NullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange
            var embeddingModel = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new DenseRetriever<double>(null!, embeddingModel));
        }

        [Fact]
        public void Constructor_NullEmbeddingModel_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new DenseRetriever<double>(store, null!));
        }

        [Fact]
        public void Constructor_BothParametersNull_ThrowsArgumentNullException()
        {
            // Act & Assert - should throw for the first null parameter (documentStore)
            Assert.Throws<ArgumentNullException>(() =>
                new DenseRetriever<double>(null!, null!));
        }

        #endregion

        #region Retrieve Method Tests

        [Fact]
        public void Retrieve_EmptyStore_ReturnsEmptyResults()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("test query").ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_SingleDocument_ReturnsDocument()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox jumps over the lazy dog"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_MultipleDocuments_ReturnsRankedByRelevance()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"),
                ("doc2", "A lazy dog sleeps"),
                ("doc3", "Climate change impacts"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert
            Assert.True(results.Count >= 1);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_SimilarContent_RankedByVectorSimilarity()
        {
            // Arrange - documents with similar content should have similar embeddings
            var store = CreateStoreWithDocuments(
                ("doc1", "quick brown fox"),
                ("doc2", "fast brown fox"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert - "quick brown fox" should match "quick brown fox" better
            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_WithTopK_ReturnsLimitedResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "content one"),
                ("doc2", "content two"),
                ("doc3", "content three"),
                ("doc4", "content four"),
                ("doc5", "content five"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("content", topK: 3).ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_NullQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        [Fact]
        public void Retrieve_ZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        [Fact]
        public void Retrieve_NegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: -1).ToList());
        }

        #endregion

        #region Vector Similarity Tests

        [Fact]
        public void Retrieve_IdenticalContent_HighestRelevance()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "climate change"),
                ("doc2", "different topic"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("climate change").ToList();

            // Assert - identical content should have highest score
            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_DocumentsHaveRelevanceScores()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_RelevanceScoresAreInRange()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "test content one"),
                ("doc2", "test content two"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("test content").ToList();

            // Assert - cosine similarity should be in [-1, 1], typically [0, 1] for normalized vectors
            foreach (var result in results)
            {
                Assert.True(result.RelevanceScore >= -1 && result.RelevanceScore <= 1);
            }
        }

        [Fact]
        public void Retrieve_ResultsOrderedByRelevance()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "apple"),
                ("doc2", "apple pie"),
                ("doc3", "banana"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("apple").ToList();

            // Assert - results should be ordered by descending relevance
            for (int i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].RelevanceScore >= results[i].RelevanceScore);
            }
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_ReturnsOnlyMatchingDocuments()
        {
            // Arrange
            var store = CreateStoreWithDocumentsAndMetadata(
                ("doc1", "fox in the forest", new Dictionary<string, object> { { "category", "nature" } }),
                ("doc2", "fox in the city", new Dictionary<string, object> { { "category", "urban" } }));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object> { { "category", "nature" } }).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_NoMatchingMetadata_ReturnsEmpty()
        {
            // Arrange
            var store = CreateStoreWithDocumentsAndMetadata(
                ("doc1", "fox document", new Dictionary<string, object> { { "category", "nature" } }));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object> { { "category", "technology" } }).ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_EmptyMetadataFilter_ReturnsAllMatches()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "content one"),
                ("doc2", "content two"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("content", topK: 5,
                new Dictionary<string, object>()).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_NullMetadataFilter_ThrowsArgumentNullException()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "content"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                retriever.Retrieve("content", topK: 5, null!).ToList());
        }

        [Fact]
        public void Retrieve_MultipleMetadataFilters_ReturnsMatchingAll()
        {
            // Arrange
            var store = CreateStoreWithDocumentsAndMetadata(
                ("doc1", "content", new Dictionary<string, object> { { "type", "A" }, { "status", "active" } }),
                ("doc2", "content", new Dictionary<string, object> { { "type", "A" }, { "status", "inactive" } }),
                ("doc3", "content", new Dictionary<string, object> { { "type", "B" }, { "status", "active" } }));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("content", topK: 5,
                new Dictionary<string, object> { { "type", "A" }, { "status", "active" } }).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_SingleWordQuery_Works()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_SingleWordDocument_Works()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_LongDocument_HandlesCorrectly()
        {
            // Arrange
            var longContent = string.Join(" ", Enumerable.Repeat("The quick brown fox jumps over the lazy dog.", 100));
            var store = CreateStoreWithDocuments(("doc1", longContent));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_SpecialCharactersInQuery_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "Hello, world! How are you?"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("hello world").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"),
                ("doc2", "A lazy dog"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results1 = retriever.Retrieve("fox").ToList();
            var results2 = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(results1.Count, results2.Count);
            for (int i = 0; i < results1.Count; i++)
            {
                Assert.Equal(results1[i].Id, results2[i].Id);
            }
        }

        [Fact]
        public void Retrieve_TopKGreaterThanDocCount_ReturnsAllDocuments()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "content one"),
                ("doc2", "content two"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("content", topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_UnicodeContent_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "日本語テスト"),
                ("doc2", "中文测试"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("日本語").ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_EmojisInContent_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Hello world 😀"),
                ("doc2", "Goodbye world 😢"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Hello 😀").ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        #endregion

        #region Embedding Integration Tests

        [Fact]
        public void Retrieve_UsesEmbeddingModel()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "exact match text"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act - query with identical text should have highest similarity
            var results = retriever.Retrieve("exact match text").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
            // Identical text should have very high similarity (close to 1.0)
            Assert.True(results[0].RelevanceScore > 0.9);
        }

        [Fact]
        public void Retrieve_DifferentQueries_ReturnResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "apple fruit"),
                ("doc2", "banana fruit"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new DenseRetriever<double>(store, embeddingModel);

            // Act
            var appleResults = retriever.Retrieve("apple").ToList();
            var bananaResults = retriever.Retrieve("banana").ToList();

            // Assert - both queries should return results with valid scores
            Assert.Equal(2, appleResults.Count);
            Assert.Equal(2, bananaResults.Count);
            Assert.True(appleResults[0].HasRelevanceScore);
            Assert.True(bananaResults[0].HasRelevanceScore);
        }

        #endregion
    }
}
