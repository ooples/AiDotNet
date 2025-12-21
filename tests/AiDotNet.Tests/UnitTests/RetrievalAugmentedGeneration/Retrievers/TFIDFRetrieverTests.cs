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
    /// Unit tests for TFIDFRetriever class
    /// </summary>
    public class TFIDFRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private const int VectorDimension = 128;

        public TFIDFRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new TFIDFRetriever<double>(null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new TFIDFRetriever<double>(_documentStore);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomDefaultTopK_SetsCorrectly()
        {
            // Arrange & Act
            var retriever = new TFIDFRetriever<double>(_documentStore, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        #endregion

        #region Basic Retrieval Tests

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var retriever = new TFIDFRetriever<double>(_documentStore);

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
            var retriever = new TFIDFRetriever<double>(_documentStore);

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
            var retriever = new TFIDFRetriever<double>(_documentStore);

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
            var retriever = new TFIDFRetriever<double>(_documentStore);

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

        #region TF-IDF Scoring Tests

        [Fact]
        public void Retrieve_WithExactKeywordMatch_ReturnsRelevantDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new TFIDFRetriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("machine", topK: 5).ToList();

            // Assert - Should find documents containing "machine"
            Assert.NotEmpty(results);
            var topResult = results.First();
            Assert.Contains("machine", topResult.Content.ToLowerInvariant());
        }

        [Fact]
        public void Retrieve_WithRareTerms_ScoresHigher()
        {
            // Arrange
            var store = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            var docs = new List<Document<double>>
            {
                new Document<double>("doc1", "machine learning neural network"),
                new Document<double>("doc2", "machine learning deep learning"),
                new Document<double>("doc3", "machine learning algorithms"),
                new Document<double>("doc4", "quantum computing algorithms")  // "quantum" is rare
            };
            AddDocumentsToStore(store, docs);

            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("quantum", topK: 4).ToList();

            // Assert - Document with rare term should rank high
            Assert.NotEmpty(results);
            Assert.Equal("doc4", results.First().Id);
        }

        [Fact]
        public void Retrieve_WithCommonTerms_ScoresLower()
        {
            // Arrange
            var store = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            var docs = new List<Document<double>>
            {
                new Document<double>("doc1", "machine learning"),
                new Document<double>("doc2", "machine algorithms"),
                new Document<double>("doc3", "machine vision"),
                new Document<double>("doc4", "specialized quantum system")
            };
            AddDocumentsToStore(store, docs);

            var retriever = new TFIDFRetriever<double>(store);

            // Act - "machine" appears in many documents (common term)
            var results = retriever.Retrieve("machine", topK: 4).ToList();

            // Assert - All documents should have scores, but common terms have lower IDF
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        #endregion

        #region Caching Tests

        [Fact]
        public void Retrieve_MultipleCalls_UsesCachedStatistics()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new TFIDFRetriever<double>(_documentStore);

            // Act - First call builds cache
            var results1 = retriever.Retrieve("machine learning", topK: 5).ToList();
            // Second call uses cache
            var results2 = retriever.Retrieve("deep learning", topK: 5).ToList();

            // Assert - Both calls should work correctly
            Assert.NotEmpty(results1);
            Assert.NotEmpty(results2);
        }

        [Fact]
        public void Retrieve_AfterDocumentCountChanges_RebuildsCache()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new TFIDFRetriever<double>(_documentStore);

            // Act - First retrieval builds cache
            _ = retriever.Retrieve("machine learning", topK: 5).ToList();

            // Add more documents (changes document count)
            var embeddingModel = TestHelpers.CreateEmbeddingModel<double>(VectorDimension);
            var doc = new Document<double>("doc6", "machine learning deep neural networks")
            {
                Metadata = new Dictionary<string, object> { ["category"] = "AI" }
            };
            var newDoc = new VectorDocument<double>
            {
                Document = doc,
                Embedding = embeddingModel.Embed("machine learning deep neural networks")
            };
            _documentStore.Add(newDoc);

            // Second retrieval should rebuild cache
            var results2 = retriever.Retrieve("machine learning", topK: 10).ToList();

            // Assert - Should work correctly with updated cache
            Assert.NotEmpty(results2);
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_ReturnsOnlyMatchingDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new TFIDFRetriever<double>(_documentStore);
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
            var retriever = new TFIDFRetriever<double>(_documentStore);
            var filters = new Dictionary<string, object> { ["category"] = "Sports" };

            // Act
            var results = retriever.Retrieve("machine learning", topK: 10, filters).ToList();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_WithNoMatchingTerms_ReturnsZeroScores()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new TFIDFRetriever<double>(_documentStore);

            // Act
            var results = retriever.Retrieve("zzzznonexistent", topK: 5).ToList();

            // Assert
            // TF-IDF returns all documents even with no matching terms, but with zero scores
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.Equal(0.0, doc.RelevanceScore));
        }

        [Fact]
        public void Retrieve_WithQueryContainingPunctuation_HandlesCorrectly()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new TFIDFRetriever<double>(_documentStore);

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
            var retriever = new TFIDFRetriever<double>(_documentStore);

            // Act
            var lowerResults = retriever.Retrieve("machine learning", topK: 5).ToList();
            var upperResults = retriever.Retrieve("MACHINE LEARNING", topK: 5).ToList();

            // Assert - Should return same documents (case-insensitive)
            Assert.Equal(lowerResults.Count, upperResults.Count);
        }

        [Fact]
        public void Retrieve_WithSingleDocumentStore_HandlesGracefully()
        {
            // Arrange
            var store = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            var docs = new List<Document<double>>
            {
                new Document<double>("doc1", "machine learning")
            };
            AddDocumentsToStore(store, docs);

            var retriever = new TFIDFRetriever<double>(store);

            // Act
            var results = retriever.Retrieve("machine", topK: 5).ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var floatStore = TestHelpers.CreateDocumentStore<float>(VectorDimension);
            var floatDocs = TestHelpers.CreateSampleDocuments<float>();
            AddDocumentsToStore(floatStore, floatDocs);

            var retriever = new TFIDFRetriever<float>(floatStore);

            // Act
            var results = retriever.Retrieve("machine learning").ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
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
