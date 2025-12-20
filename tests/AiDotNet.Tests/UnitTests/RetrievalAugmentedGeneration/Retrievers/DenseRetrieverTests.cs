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
    /// Unit tests for DenseRetriever class
    /// </summary>
    public class DenseRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private readonly IEmbeddingModel<double> _embeddingModel;
        private const int VectorDimension = 128;

        public DenseRetrieverTests()
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
                new DenseRetriever<double>(null, _embeddingModel));
        }

        [Fact]
        public void Constructor_WithNullEmbeddingModel_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new DenseRetriever<double>(_documentStore, null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomDefaultTopK_SetsCorrectly()
        {
            // Arrange & Act
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        #endregion

        #region Basic Retrieval Tests

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);

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
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);

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
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);

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
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);

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
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);
            var filters = new Dictionary<string, object> { ["category"] = "Food" };

            // Act
            var results = retriever.Retrieve("pasta", topK: 10, filters).ToList();

            // Assert
            Assert.All(results, doc =>
            {
                Assert.True(doc.Metadata.TryGetValue("category", out var category));
                Assert.Equal("Food", category);
            });
        }

        [Fact]
        public void Retrieve_WithNonMatchingFilter_ReturnsEmpty()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);
            var filters = new Dictionary<string, object> { ["category"] = "Sports" };

            // Act
            var results = retriever.Retrieve("machine learning", topK: 10, filters).ToList();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Semantic Search Tests

        [Fact]
        public void Retrieve_WithSemanticQuery_FindsRelevantDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new DenseRetriever<double>(_documentStore, _embeddingModel);

            // Act - Query for AI-related content
            var results = retriever.Retrieve("artificial intelligence", topK: 10).ToList();

            // Assert - Should find AI-related documents
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

            var vectorDocs = floatDocs.Select(doc =>
            {
                var embedding = floatEmbedding.Embed(doc.Content);
                return new VectorDocument<float>
                {
                    Document = doc,
                    Embedding = embedding
                };
            });

            floatStore.AddBatch(vectorDocs);
            var retriever = new DenseRetriever<float>(floatStore, floatEmbedding);

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
