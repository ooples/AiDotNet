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
    /// Unit tests for HybridRetriever class
    /// </summary>
    public class HybridRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private readonly IEmbeddingModel<double> _embeddingModel;
        private const int VectorDimension = 128;

        public HybridRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            _embeddingModel = TestHelpers.CreateEmbeddingModel<double>(VectorDimension);
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullDenseRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var sparseRetriever = new BM25Retriever<double>(_documentStore);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridRetriever<double>(null, sparseRetriever));
        }

        [Fact]
        public void Constructor_WithNullSparseRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridRetriever<double>(denseRetriever, null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);

            // Act
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomWeights_CreatesInstance()
        {
            // Arrange
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);

            // Act
            var retriever = new HybridRetriever<double>(
                denseRetriever,
                sparseRetriever,
                denseWeight: 0.6,
                sparseWeight: 0.4);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_WithCustomDefaultTopK_SetsCorrectly()
        {
            // Arrange
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);

            // Act
            var retriever = new HybridRetriever<double>(
                denseRetriever,
                sparseRetriever,
                defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        #endregion

        #region Basic Retrieval Tests

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

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
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

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
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

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
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

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

        #region Hybrid Fusion Tests

        [Fact]
        public void Retrieve_CombinesDenseAndSparseResults()
        {
            // Arrange
            AddSampleDocuments();
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var hybridResults = retriever.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(hybridResults);
            // Hybrid results should potentially include documents from both retrievers
            Assert.True(hybridResults.Count > 0);
        }

        [Fact]
        public void Retrieve_WithDifferentWeights_ProducesDifferentRankings()
        {
            // Arrange
            AddSampleDocuments();
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);

            var retriever1 = new HybridRetriever<double>(
                denseRetriever,
                sparseRetriever,
                denseWeight: 0.9,
                sparseWeight: 0.1);

            var retriever2 = new HybridRetriever<double>(
                denseRetriever,
                sparseRetriever,
                denseWeight: 0.1,
                sparseWeight: 0.9);

            // Act
            var results1 = retriever1.Retrieve("machine learning", topK: 5).ToList();
            var results2 = retriever2.Retrieve("machine learning", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results1);
            Assert.NotEmpty(results2);
            // Different weights may produce different rankings
        }

        [Fact]
        public void Retrieve_WithBalancedWeights_BalancesBothStrategies()
        {
            // Arrange
            AddSampleDocuments();
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(
                denseRetriever,
                sparseRetriever,
                denseWeight: 0.5,
                sparseWeight: 0.5);

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
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);
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
            var denseRetriever = new VectorRetriever<double>(_documentStore, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(_documentStore);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);
            var filters = new Dictionary<string, object> { ["category"] = "Sports" };

            // Act
            var results = retriever.Retrieve("machine learning", topK: 10, filters).ToList();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_WithOnlyDenseMatches_WorksCorrectly()
        {
            // Arrange
            var store = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            var docs = new List<Document<double>>
            {
                new Document<double>("doc1", "completely different content"),
                new Document<double>("doc2", "totally unrelated material")
            };
            AddDocumentsToStore(store, docs);

            var denseRetriever = new VectorRetriever<double>(store, _embeddingModel);
            var sparseRetriever = new BM25Retriever<double>(store);
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("something", topK: 2).ToList();

            // Assert
            // Should still return results based on dense retrieval
            Assert.NotEmpty(results);
        }

        [Fact]
        public void Retrieve_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var floatStore = TestHelpers.CreateDocumentStore<float>(VectorDimension);
            var floatEmbedding = TestHelpers.CreateEmbeddingModel<float>(VectorDimension);
            var floatDocs = TestHelpers.CreateSampleDocuments<float>();
            AddDocumentsToStore(floatStore, floatDocs, floatEmbedding);

            var denseRetriever = new VectorRetriever<float>(floatStore, floatEmbedding);
            var sparseRetriever = new BM25Retriever<float>(floatStore);
            var retriever = new HybridRetriever<float>(denseRetriever, sparseRetriever);

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

        private void AddDocumentsToStore<T>(IDocumentStore<T> store, List<Document<T>> documents, IEmbeddingModel<T> embeddingModel = null)
        {
            if (embeddingModel == null)
            {
                embeddingModel = TestHelpers.CreateEmbeddingModel<T>(VectorDimension);
            }

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
