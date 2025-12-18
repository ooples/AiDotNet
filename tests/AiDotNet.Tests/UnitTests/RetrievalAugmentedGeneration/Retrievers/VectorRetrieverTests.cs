// Nullable disabled: This test file intentionally passes null values to test argument validation
#nullable disable

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Unit tests for VectorRetriever class (also tests RetrieverBase functionality)
    /// </summary>
    public class VectorRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private readonly IEmbeddingModel<double> _embeddingModel;
        private const int VectorDimension = 128;

        public VectorRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            _embeddingModel = TestHelpers.CreateEmbeddingModel<double>(VectorDimension);
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new VectorRetriever<double>(null, _embeddingModel));
        }

        [Fact]
        public void Constructor_WithNullEmbeddingModel_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new VectorRetriever<double>(_documentStore, null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomDefaultTopK_SetsCorrectly()
        {
            // Arrange & Act
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithZeroDefaultTopK_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new VectorRetriever<double>(_documentStore, _embeddingModel, defaultTopK: 0));
            Assert.Contains("must be greater than zero", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeDefaultTopK_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new VectorRetriever<double>(_documentStore, _embeddingModel, defaultTopK: -1));
            Assert.Contains("must be greater than zero", exception.Message);
        }

        #endregion

        #region Query Validation Tests (RetrieverBase)

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(null));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(""));
        }

        [Fact]
        public void Retrieve_WithWhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve("   "));
        }

        #endregion

        #region TopK Validation Tests (RetrieverBase)

        [Fact]
        public void Retrieve_WithZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => retriever.Retrieve("test query", 0));
        }

        [Fact]
        public void Retrieve_WithNegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => retriever.Retrieve("test query", -1));
        }

        #endregion

        #region Metadata Filter Validation Tests (RetrieverBase)

        [Fact]
        public void Retrieve_WithNullMetadataFilters_ThrowsArgumentNullException()
        {
            // Arrange
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                retriever.Retrieve("test query", 5, null));
        }

        #endregion

        #region Basic Retrieval Tests

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

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
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var results = retriever.Retrieve("machine learning").ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_WithDefaultTopK_ReturnsCorrectNumberOfResults()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("machine learning").ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_WithCustomTopK_ReturnsCorrectNumberOfResults()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 2).ToList();

            // Assert
            Assert.True(results.Count <= 2);
        }

        #endregion

        #region Relevance Score Tests

        [Fact]
        public void Retrieve_WithValidQuery_AssignsRelevanceScores()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var results = retriever.Retrieve("machine learning").ToList();

            // Assert
            Assert.All(results, doc =>
            {
                Assert.True(doc.HasRelevanceScore);
                Assert.True(doc.RelevanceScore >= 0.0);
                Assert.True(doc.RelevanceScore <= 1.0);
            });
        }

        [Fact]
        public void Retrieve_WithValidQuery_ReturnsSortedByRelevance()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

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

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_ReturnsOnlyMatchingDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var filters = new Dictionary<string, object> { ["category"] = "AI" };

            // Act
            var results = retriever.Retrieve("cooking", topK: 10, filters).ToList();

            // Assert
            Assert.All(results, doc =>
            {
                Assert.True(doc.Metadata.TryGetValue("category", out var category));
                Assert.Equal("AI", category);
            });
        }

        [Fact]
        public void Retrieve_WithEmptyMetadataFilter_ReturnsAllDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var filters = new Dictionary<string, object>();

            // Act
            var results = retriever.Retrieve("machine learning", topK: 10, filters).ToList();

            // Assert
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Retrieve_WithNonMatchingFilter_ReturnsEmpty()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var filters = new Dictionary<string, object> { ["category"] = "Sports" };

            // Act
            var results = retriever.Retrieve("machine learning", topK: 10, filters).ToList();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Helper Methods

        private void AddSampleDocuments()
        {
            var documents = TestHelpers.CreateSampleDocuments<double>();
            var vectorDocuments = documents.Select(doc =>
            {
                var embedding = _embeddingModel.Embed(doc.Content);
                return new VectorDocument<double>
                {
                    Document = doc,
                    Embedding = embedding
                };
            });

            _documentStore.AddBatch(vectorDocuments);
        }

        #endregion
    }
}
