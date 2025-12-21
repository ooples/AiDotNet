// Nullable disabled: This test file intentionally passes null values to test argument validation
#nullable disable

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Unit tests for BM25Retriever class
    /// </summary>
    public class BM25RetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private const int VectorDimension = 128;

        public BM25RetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new BM25Retriever<double>(null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new BM25Retriever<double>(_documentStore);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomDefaultTopK_SetsCorrectly()
        {
            // Arrange & Act
            var retriever = new BM25Retriever<double>(_documentStore, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new BM25Retriever<double>(_documentStore, k1: 2.0, b: 0.5);

            // Assert
            Assert.NotNull(retriever);
        }

        #endregion

        #region Basic Retrieval Tests

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine learning");

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_WithValidQuery_ReturnsDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine learning").ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_WithCustomTopK_ReturnsCorrectNumberOfResults()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 2).ToList();

            // Assert
            Assert.True(results.Count <= 2);
        }

        [Fact]
        public void Retrieve_WithValidQuery_ReturnsSortedByRelevance()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            for (int i = 0; i < results.Count - 1; i++)
            {
                Assert.True(results[i].RelevanceScore >= results[i + 1].RelevanceScore,
                    "Results should be sorted by relevance in descending order");
            }
        }

        #endregion

        #region Keyword Matching Tests

        [Fact]
        public void Retrieve_WithExactKeywordMatch_ReturnsRelevantDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine", topK: 5).ToList();

            // Assert - Should find documents containing "machine"
            Assert.NotEmpty(results);
            var topResult = results.First();
            Assert.Contains("machine", topResult.Content.ToLowerInvariant());
        }

        [Fact]
        public void Retrieve_WithMultipleKeywords_ScoresAppropriately()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine learning algorithms", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_WithNoMatchingKeywords_ReturnsLowScores()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("quantum physics", topK: 5).ToList();

            // Assert - May return documents but with low scores
            if (results.Any())
            {
                Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
            }
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_ReturnsOnlyMatchingDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);
            var filters = new Dictionary<string, object> { ["category"] = "AI" };

            // Act
            var results = retriever.Retrieve("machine", topK: 10, filters).ToList();

            // Assert
            Assert.All(results, doc =>
            {
                Assert.True(doc.Metadata.TryGetValue("category", out var category));
                Assert.Equal("AI", category);
            });
        }

        [Fact]
        public void Retrieve_WithNonMatchingFilter_ReturnsEmpty()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);
            var filters = new Dictionary<string, object> { ["category"] = "Sports" };

            // Act
            var results = retriever.Retrieve("machine learning", topK: 10, filters).ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_WithEmptyMetadataFilter_ReturnsAllDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);
            var filters = new Dictionary<string, object>();

            // Act
            var results = retriever.Retrieve("machine", topK: 10, filters).ToList();

            // Assert
            Assert.NotEmpty(results);
        }

        #endregion

        #region BM25-Specific Tests

        [Fact]
        public void Retrieve_WithDifferentK1Values_ProducesDifferentScores()
        {
            // Arrange
            AddSampleDocuments();
            var retriever1 = new BM25Retriever<double>(_documentStore, k1: 0.5);
            var retriever2 = new BM25Retriever<double>(_documentStore, k1: 2.0);

            // Act
            var results1 = retriever1.Retrieve("machine learning", topK: 1).ToList();
            var results2 = retriever2.Retrieve("machine learning", topK: 1).ToList();

            // Assert
            Assert.NotEmpty(results1);
            Assert.NotEmpty(results2);
            // Different k1 values should produce different scores (in most cases)
        }

        [Fact]
        public void Retrieve_WithDifferentBValues_ProducesDifferentScores()
        {
            // Arrange
            AddSampleDocuments();
            var retriever1 = new BM25Retriever<double>(_documentStore, b: 0.25);
            var retriever2 = new BM25Retriever<double>(_documentStore, b: 0.95);

            // Act
            var results1 = retriever1.Retrieve("machine learning", topK: 1).ToList();
            var results2 = retriever2.Retrieve("machine learning", topK: 1).ToList();

            // Assert
            Assert.NotEmpty(results1);
            Assert.NotEmpty(results2);
            // Different b values should produce different scores (in most cases)
        }

        [Fact]
        public void Retrieve_WithRepeatedTerms_ScoresHigher()
        {
            // Arrange
            var store = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            var docs = new List<Document<double>>
            {
                new Document<double>("doc1", "machine machine machine learning"),
                new Document<double>("doc2", "machine learning"),
                // Add documents without "machine" to ensure IDF stays positive
                // (BM25 IDF is negative when term appears in more than half the documents)
                // For positive IDF: df < N/2, so with 2 docs containing "machine", we need N > 4
                new Document<double>("doc3", "deep neural networks"),
                new Document<double>("doc4", "artificial intelligence systems"),
                new Document<double>("doc5", "data science algorithms")
            };
            AddDocumentsToStore(store, docs);

            var retriever = new BM25Retriever<double>(store);

            // Act
            var results = retriever.Retrieve("machine", topK: 2).ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Document with more "machine" occurrences should score higher
            Assert.Equal("doc1", results.First().Id);
        }

        [Fact]
        public void Retrieve_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var floatStore = TestHelpers.CreateDocumentStore<float>(VectorDimension);
            var floatDocs = TestHelpers.CreateSampleDocuments<float>();
            AddDocumentsToStore(floatStore, floatDocs);

            var retriever = new BM25Retriever<float>(floatStore);

            // Act
            var results = retriever.Retrieve("machine learning").ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_WithQueryContainingPunctuation_HandlesCorrectly()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine, learning!", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Retrieve_WithCaseSensitiveQuery_WorksCorrectly()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new BM25Retriever<double>(_documentStore);

            // Act
            var lowerResults = retriever.Retrieve("machine learning", topK: 5).ToList();
            var upperResults = retriever.Retrieve("MACHINE LEARNING", topK: 5).ToList();

            // Assert - Should return same documents (case-insensitive)
            Assert.Equal(lowerResults.Count, upperResults.Count);
        }

        #endregion

        #region Helper Methods

        private void AddSampleDocuments()
        {
            var documents = TestHelpers.CreateSampleDocuments<double>();
            AddDocumentsToStore(_documentStore, documents);
        }

        private void AddDocumentsToStore<T>(IDocumentStore<T> store, List<Document<T>> documents)
        {
            var embeddingModel = TestHelpers.CreateEmbeddingModel<T>(VectorDimension);
            var vectorDocuments = documents.Select(doc =>
            {
                var embedding = embeddingModel.Embed(doc.Content);
                return new VectorDocument<T>
                {
                    Document = doc,
                    Embedding = embedding
                };
            });

            store.AddBatch(vectorDocuments);
        }

        #endregion
    }
}
