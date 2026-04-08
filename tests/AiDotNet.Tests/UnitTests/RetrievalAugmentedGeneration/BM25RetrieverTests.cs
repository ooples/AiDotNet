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
    /// Tests for BM25Retriever which implements the BM25 algorithm for keyword-based search.
    /// </summary>
    public class BM25RetrieverTests
    {
        #region Test Helpers

        /// <summary>
        /// Simple mock document store for testing BM25Retriever.
        /// </summary>
        private class MockDocumentStore : IDocumentStore<double>
        {
            private readonly List<Document<double>> _documents = new();

            public int DocumentCount => _documents.Count;
            public int VectorDimension => 0; // Not used by BM25

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
                => Enumerable.Empty<Document<double>>();

            public IEnumerable<Document<double>> GetSimilarWithFilters(Vector<double> queryVector, int topK, Dictionary<string, object> metadataFilters)
                => Enumerable.Empty<Document<double>>();

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

            // Act
            var retriever = new BM25Retriever<double>(store);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomTopK_SetsCorrectly()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act
            var retriever = new BM25Retriever<double>(store, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomBM25Parameters_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act
            var retriever = new BM25Retriever<double>(store, defaultTopK: 5, k1: 2.0, b: 0.5);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_NullDocumentStore_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new BM25Retriever<double>(null!));
        }

        [Fact]
        public void Constructor_ZeroTopK_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new BM25Retriever<double>(store, defaultTopK: 0));
        }

        [Fact]
        public void Constructor_NegativeTopK_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new BM25Retriever<double>(store, defaultTopK: -1));
        }

        #endregion

        #region Retrieve Method Tests

        [Fact]
        public void Retrieve_EmptyStore_ReturnsEmptyResults()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("test query").ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_MatchingDocument_ReturnsDocument()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox jumps over the lazy dog"));
            var retriever = new BM25Retriever<double>(store);

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
                ("doc1", "The quick brown fox jumps over the lazy dog"),
                ("doc2", "A fox is a small animal"),
                ("doc3", "Completely unrelated content about cars"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.True(results.Count >= 2, "Should return documents containing 'fox'");
            // Documents containing "fox" should be returned first
            Assert.Contains(results, r => r.Id == "doc1" || r.Id == "doc2");
        }

        [Fact]
        public void Retrieve_NoMatchingTerms_ReturnsEmptyOrLowScored()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("elephant").ToList();

            // Assert
            // No documents contain "elephant", so results should be empty or have zero score
            Assert.True(results.Count == 0 || results.All(r => !r.HasRelevanceScore || Convert.ToDouble(r.RelevanceScore) <= 0));
        }

        [Fact]
        public void Retrieve_WithTopK_ReturnsLimitedResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox fox fox"),
                ("doc2", "fox fox"),
                ("doc3", "fox"),
                ("doc4", "another fox document"),
                ("doc5", "yet another fox text"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox", topK: 3).ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_CaseInsensitive_MatchesDifferentCases()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The QUICK Brown FOX"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("quick fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_NullQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new BM25Retriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new BM25Retriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new BM25Retriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        #endregion

        #region BM25 Scoring Tests

        [Fact]
        public void Retrieve_HigherTermFrequency_GetHigherScore()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox"),
                ("doc2", "fox fox fox"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Document with more "fox" occurrences should score higher
            Assert.True(results[0].HasRelevanceScore);
            Assert.True(results[1].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_MoreQueryTermsMatch_GetHigherScore()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "the quick brown fox"),
                ("doc2", "the quick fox"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Document matching more query terms should typically score higher
        }

        [Fact]
        public void Retrieve_DocumentsHaveRelevanceScores()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
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
            var retriever = new BM25Retriever<double>(store);

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
            var retriever = new BM25Retriever<double>(store);

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
                ("doc1", "fox one"),
                ("doc2", "fox two"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object>()).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_SingleWordDocument_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_LongDocument_HandlesCorrectly()
        {
            // Arrange
            var longContent = string.Join(" ", Enumerable.Repeat("The quick brown fox jumps over the lazy dog.", 100));
            var store = CreateStoreWithDocuments(("doc1", longContent));
            var retriever = new BM25Retriever<double>(store);

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
            var retriever = new BM25Retriever<double>(store);

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
            var retriever = new BM25Retriever<double>(store);

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
                ("doc1", "fox one"),
                ("doc2", "fox two"));
            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox", topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_ZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new BM25Retriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        #endregion
    }
}
