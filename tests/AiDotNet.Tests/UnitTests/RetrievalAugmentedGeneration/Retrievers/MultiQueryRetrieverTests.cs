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
    /// Unit tests for MultiQueryRetriever class
    /// </summary>
    public class MultiQueryRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private readonly IEmbeddingModel<double> _embeddingModel;
        private const int VectorDimension = 128;

        public MultiQueryRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            _embeddingModel = TestHelpers.CreateEmbeddingModel<double>(VectorDimension);
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullBaseRetriever_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiQueryRetriever<double>(null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var retriever = new MultiQueryRetriever<double>(baseRetriever);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomNumQueries_CreatesInstance()
        {
            // Arrange
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var retriever = new MultiQueryRetriever<double>(baseRetriever, numQueries: 5);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_WithZeroNumQueries_ThrowsArgumentException()
        {
            // Arrange
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new MultiQueryRetriever<double>(baseRetriever, numQueries: 0));
            Assert.Contains("must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeNumQueries_ThrowsArgumentException()
        {
            // Arrange
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new MultiQueryRetriever<double>(baseRetriever, numQueries: -1));
            Assert.Contains("must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithCustomDefaultTopK_SetsCorrectly()
        {
            // Arrange
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var retriever = new MultiQueryRetriever<double>(baseRetriever, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        #endregion

        #region Basic Retrieval Tests

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever);

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
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever);

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
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever);

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
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever);

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

        #region Multi-Query Expansion Tests

        [Fact]
        public void Retrieve_WithMultipleQueries_AggregatesScores()
        {
            // Arrange
            AddSampleDocuments();
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever, numQueries: 3);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc =>
            {
                Assert.True(doc.HasRelevanceScore);
                // Scores should be aggregated from multiple queries
                Assert.True(doc.RelevanceScore > 0.0);
            });
        }

        [Fact]
        public void Retrieve_WithSingleQuery_WorksCorrectly()
        {
            // Arrange
            AddSampleDocuments();
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever, numQueries: 1);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Retrieve_WithManyQueries_WorksCorrectly()
        {
            // Arrange
            AddSampleDocuments();
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever, numQueries: 10);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_ReturnsOnlyMatchingDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever);
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
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var retriever = new MultiQueryRetriever<double>(baseRetriever);
            var filters = new Dictionary<string, object> { ["category"] = "Sports" };

            // Act
            var results = retriever.Retrieve("machine learning", topK: 10, filters).ToList();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Score Aggregation Tests

        [Fact]
        public void Retrieve_AggregatesScoresAcrossQueries()
        {
            // Arrange
            AddSampleDocuments();
            var baseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var multiRetriever = new MultiQueryRetriever<double>(baseRetriever, numQueries: 3);

            // Act
            var multiResults = multiRetriever.Retrieve("machine learning", topK: 5).ToList();
            var baseResults = baseRetriever.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(multiResults);
            Assert.NotEmpty(baseResults);
            // Multi-query results should have aggregated scores
            Assert.All(multiResults, doc => Assert.True(doc.HasRelevanceScore));
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_WithDifferentBaseRetrievers_WorksCorrectly()
        {
            // Arrange - Test with BM25 as base retriever
            AddSampleDocuments();
            var baseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new MultiQueryRetriever<double>(baseRetriever, numQueries: 3);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var floatStore = TestHelpers.CreateDocumentStore<float>(VectorDimension);
            var floatEmbedding = TestHelpers.CreateEmbeddingModel<float>(VectorDimension);
            var floatDocs = TestHelpers.CreateSampleDocuments<float>();
            AddDocumentsToStore(floatStore, floatDocs, floatEmbedding);

            var baseRetriever = new VectorRetriever<float>(floatStore, floatEmbedding);
            var retriever = new MultiQueryRetriever<float>(baseRetriever, numQueries: 3);

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
            AddDocumentsToStore(_documentStore, documents, _embeddingModel);
        }

        private void AddDocumentsToStore<T>(IDocumentStore<T> store, List<Document<T>> documents, IEmbeddingModel<T> embeddingModel)
        {
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
