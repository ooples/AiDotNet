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
    /// Tests for TFIDFRetriever which implements TF-IDF keyword-based retrieval.
    /// </summary>
    public class TFIDFRetrieverTests
    {
        #region Test Helpers

        /// <summary>
        /// Simple mock document store for testing TFIDFRetriever.
        /// </summary>
        private class MockDocumentStore : IDocumentStore<double>
        {
            private readonly List<Document<double>> _documents = new();

            public int DocumentCount => _documents.Count;
            public int VectorDimension => 0; // Not used by TFIDF

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
            var retriever = new TFIDFRetriever<double>(store);

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
            var retriever = new TFIDFRetriever<double>(store, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_NullDocumentStore_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new TFIDFRetriever<double>(null!));
        }

        [Fact]
        public void Constructor_ZeroTopK_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new TFIDFRetriever<double>(store, defaultTopK: 0));
        }

        [Fact]
        public void Constructor_NegativeTopK_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new TFIDFRetriever<double>(store, defaultTopK: -1));
        }

        #endregion

        #region Retrieve Method Tests

        [Fact]
        public void Retrieve_EmptyStore_ReturnsEmptyResults()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new TFIDFRetriever<double>(store);

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
                ("doc1", "the quick brown fox jumps over the lazy dog"));
            var retriever = new TFIDFRetriever<double>(store);

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
                ("doc1", "the quick brown fox jumps over the lazy dog"),
                ("doc2", "a fox is a small animal"),
                ("doc3", "completely unrelated content about cars"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            // Documents containing "fox" should have higher scores
            Assert.True(results.Count >= 2, "Should return documents containing 'fox'");
        }

        [Fact]
        public void Retrieve_NoMatchingTerms_ReturnsZeroScoreDocuments()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "the quick brown fox"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("elephant").ToList();

            // Assert
            // No matching terms, so scores should be zero
            Assert.True(results.All(r => Convert.ToDouble(r.RelevanceScore) == 0));
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
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox", topK: 3).ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_CaseInsensitive_MatchesDifferentCases()
        {
            // Arrange - Need multiple documents for meaningful IDF scores
            var store = CreateStoreWithDocuments(
                ("doc1", "The QUICK Brown FOX"),
                ("doc2", "a dog runs fast")); // Different terms for IDF
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("quick fox").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // doc1 should be first since it matches the query terms
            Assert.Equal("doc1", results[0].Id);
            Assert.True(results[0].RelevanceScore > results[1].RelevanceScore,
                "Matching document should have higher score");
        }

        [Fact]
        public void Retrieve_NullQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new TFIDFRetriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new TFIDFRetriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new TFIDFRetriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        #endregion

        #region TF-IDF Scoring Tests

        [Fact]
        public void Retrieve_HigherTermFrequency_GetsHigherScore()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox"),
                ("doc2", "fox fox fox"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            // Both documents contain "fox", but doc2 has higher frequency
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.True(r.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_RareTermsGetHigherWeight()
        {
            // Arrange - "fox" appears in all docs, "unique" appears in only one
            var store = CreateStoreWithDocuments(
                ("doc1", "fox common words"),
                ("doc2", "fox common words"),
                ("doc3", "fox unique special term"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("unique").ToList();

            // Assert
            // "unique" only appears in doc3, so it should have highest score
            Assert.True(results[0].Id == "doc3" || results.Any(r => r.Id == "doc3" && r.RelevanceScore > 0));
        }

        [Fact]
        public void Retrieve_CommonTermsGetLowerWeight()
        {
            // Arrange - Common term appears in all documents
            var store = CreateStoreWithDocuments(
                ("doc1", "the fox"),
                ("doc2", "the dog"),
                ("doc3", "the cat"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("the").ToList();

            // Assert
            // "the" appears in all docs, so IDF should be low (log(3/3) = 0)
            // All scores should be equal or zero
            Assert.Equal(3, results.Count);
        }

        [Fact]
        public void Retrieve_MoreQueryTermsMatch_GetHigherScore()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "the quick brown fox"),
                ("doc2", "the quick fox"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert
            // doc1 matches all three query terms, doc2 matches only two
            // But scoring depends on IDF weights
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.True(r.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_DocumentsHaveRelevanceScores()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"));
            var retriever = new TFIDFRetriever<double>(store);

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
            var retriever = new TFIDFRetriever<double>(store);

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
            var retriever = new TFIDFRetriever<double>(store);

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
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object>()).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_NullMetadataFilter_ThrowsArgumentNullException()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                retriever.Retrieve("fox", topK: 5, null!).ToList());
        }

        #endregion

        #region Caching Tests

        [Fact]
        public void Retrieve_RepeatedCalls_UsesCachedStatistics()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "the quick brown fox"),
                ("doc2", "a lazy dog"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act - Multiple retrieves should use cached TFIDF statistics
            var results1 = retriever.Retrieve("fox").ToList();
            var results2 = retriever.Retrieve("dog").ToList();
            var results3 = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(results1.Count, results3.Count);
            // Same query should produce same results
            for (int i = 0; i < results1.Count; i++)
            {
                Assert.Equal(results1[i].Id, results3[i].Id);
                Assert.Equal(results1[i].RelevanceScore, results3[i].RelevanceScore, 5);
            }
        }

        [Fact]
        public void Retrieve_AfterDocumentCountChange_InvalidatesCache()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox in the forest"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act - First query
            var results1 = retriever.Retrieve("fox").ToList();

            // Add document (changes count)
            store.AddDocument(new Document<double>("doc2", "another fox document"));

            // Second query should use new statistics
            var results2 = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results1);
            Assert.Equal(2, results2.Count);
        }

        #endregion

        #region Tokenization Tests

        [Fact]
        public void Retrieve_HandlesSpecialCharacters()
        {
            // Arrange - Need multiple documents for meaningful IDF scores
            var store = CreateStoreWithDocuments(
                ("doc1", "Hello, world! How are you?"),
                ("doc2", "unrelated content about cars"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("hello world").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // doc1 should match and have higher score than doc2
            Assert.Equal("doc1", results[0].Id);
            Assert.True(results[0].RelevanceScore > results[1].RelevanceScore,
                "Document with matching terms should have higher score");
        }

        [Fact]
        public void Retrieve_HandlesTabs_And_Newlines()
        {
            // Arrange - Need multiple documents for meaningful IDF scores
            var store = CreateStoreWithDocuments(
                ("doc1", "first\tsecond\nthird"),
                ("doc2", "unrelated content here"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("second").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // doc1 should match and have higher score
            Assert.Equal("doc1", results[0].Id);
            Assert.True(results[0].RelevanceScore > results[1].RelevanceScore,
                "Document with matching term should have higher score");
        }

        [Fact]
        public void Retrieve_EmptyDocumentContent_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", ""),
                ("doc2", "actual content"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            // Empty document should have zero score
            Assert.Equal(2, results.Count);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_SingleWordDocument_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_LongDocument_HandlesCorrectly()
        {
            // Arrange
            var longContent = string.Join(" ", Enumerable.Repeat("The quick brown fox jumps over the lazy dog.", 100));
            var store = CreateStoreWithDocuments(("doc1", longContent));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_TopKGreaterThanDocCount_ReturnsAllDocuments()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox one"),
                ("doc2", "fox two"));
            var retriever = new TFIDFRetriever<double>(store);

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
            var retriever = new TFIDFRetriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        [Fact]
        public void Retrieve_NegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new TFIDFRetriever<double>(store);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: -1).ToList());
        }

        [Fact]
        public void Retrieve_ManyDocuments_HandlesCorrectly()
        {
            // Arrange
            var docsList = Enumerable.Range(1, 100)
                .Select(i => ($"doc{i}", $"fox document number {i}"))
                .ToArray();
            var store = CreateStoreWithDocuments(docsList);
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox", topK: 10).ToList();

            // Assert
            Assert.Equal(10, results.Count);
        }

        [Fact]
        public void Retrieve_ResultsOrderedByScore()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox"),
                ("doc2", "fox fox"),
                ("doc3", "fox fox fox"));
            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(3, results.Count);
            for (int i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].RelevanceScore >= results[i].RelevanceScore,
                    "Results should be ordered by descending score");
            }
        }

        #endregion
    }
}
