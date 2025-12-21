using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    public class HybridDocumentStoreTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidStores_CreatesHybridStore()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act
            var hybridStore = new HybridDocumentStore<float>(
                vectorStore,
                keywordStore,
                vectorWeight: 0.7f,
                keywordWeight: 0.3f);

            // Assert
            Assert.Equal(0, hybridStore.DocumentCount);
            Assert.Equal(3, hybridStore.VectorDimension);
        }

        [Fact]
        public void Constructor_WithNullVectorStore_ThrowsArgumentNullException()
        {
            // Arrange
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridDocumentStore<float>(null!, keywordStore, 0.7f, 0.3f));
        }

        [Fact]
        public void Constructor_WithNullKeywordStore_ThrowsArgumentNullException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridDocumentStore<float>(vectorStore, null!, 0.7f, 0.3f));
        }

        [Fact]
        public void Constructor_WithDifferentWeights_CreatesStore()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act
            var hybridStore1 = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.5f, 0.5f);
            var hybridStore2 = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.8f, 0.2f);

            // Assert
            Assert.NotNull(hybridStore1);
            Assert.NotNull(hybridStore2);
        }

        #endregion

        #region Add Tests

        [Fact]
        public void Add_WithValidDocument_AddsToBothStores()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);
            var doc = CreateTestDocument("doc1", "Test content", 3);

            // Act
            hybridStore.Add(doc);

            // Assert
            Assert.Equal(1, hybridStore.DocumentCount);
            Assert.Equal(1, vectorStore.DocumentCount);
            Assert.Equal(1, keywordStore.DocumentCount);
        }

        [Fact]
        public void Add_WithNullDocument_ThrowsArgumentNullException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => hybridStore.Add(null!));
        }

        [Fact]
        public void Add_MultipleDocuments_IncreasesCount()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act
            for (int i = 0; i < 5; i++)
            {
                hybridStore.Add(CreateTestDocument($"doc{i}", $"Content {i}", 3));
            }

            // Assert
            Assert.Equal(5, hybridStore.DocumentCount);
            Assert.Equal(5, vectorStore.DocumentCount);
            Assert.Equal(5, keywordStore.DocumentCount);
        }

        #endregion

        #region AddBatch Tests

        [Fact]
        public void AddBatch_WithValidDocuments_AddsToBothStores()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);
            var docs = new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3),
                CreateTestDocument("doc3", "Content 3", 3)
            };

            // Act
            hybridStore.AddBatch(docs);

            // Assert
            Assert.Equal(3, hybridStore.DocumentCount);
            Assert.Equal(3, vectorStore.DocumentCount);
            Assert.Equal(3, keywordStore.DocumentCount);
        }

        [Fact]
        public void AddBatch_WithNullCollection_ThrowsArgumentNullException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => hybridStore.AddBatch(null!));
        }

        [Fact]
        public void AddBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => hybridStore.AddBatch(new List<VectorDocument<float>>()));
        }

        #endregion

        #region GetSimilar Tests

        [Fact]
        public void GetSimilar_WithDocuments_CombinesResultsFromBothStores()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            hybridStore.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3, new float[] { 1, 0, 0 }),
                CreateTestDocument("doc2", "Content 2", 3, new float[] { 0, 1, 0 }),
                CreateTestDocument("doc3", "Content 3", 3, new float[] { 0, 0, 1 })
            });

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = hybridStore.GetSimilar(queryVector, topK: 2).ToList();

            // Assert
            Assert.True(results.Count <= 2);
            Assert.All(results, r => Assert.True(r.HasRelevanceScore));
        }

        [Fact]
        public void GetSimilar_WithEmptyStores_ReturnsEmpty()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = hybridStore.GetSimilar(queryVector, topK: 5);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void GetSimilar_WithNullQueryVector_ThrowsArgumentNullException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => hybridStore.GetSimilar(null!, topK: 5));
        }

        [Fact]
        public void GetSimilar_WithZeroTopK_ThrowsArgumentException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => hybridStore.GetSimilar(queryVector, topK: 0));
        }

        [Fact]
        public void GetSimilar_ReturnsAtMostTopK()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            for (int i = 0; i < 10; i++)
            {
                hybridStore.Add(CreateTestDocument($"doc{i}", $"Content {i}", 3));
            }

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = hybridStore.GetSimilar(queryVector, topK: 5).ToList();

            // Assert
            Assert.True(results.Count <= 5);
        }

        [Fact]
        public void GetSimilar_WithWeightedResults_AppliesWeightsCorrectly()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            hybridStore.Add(CreateTestDocument("doc1", "Test", 3));
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = hybridStore.GetSimilar(queryVector, topK: 1).ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        #endregion

        #region GetSimilarWithFilters Tests

        [Fact]
        public void GetSimilarWithFilters_WithMatchingMetadata_ReturnsFilteredResults()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            var doc1 = CreateTestDocument("doc1", "Content 1", 3, new float[] { 1, 0, 0 });
            doc1.Document.Metadata["category"] = "science";

            var doc2 = CreateTestDocument("doc2", "Content 2", 3, new float[] { 1, 0, 0 });
            doc2.Document.Metadata["category"] = "history";

            hybridStore.Add(doc1);
            hybridStore.Add(doc2);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "category", "science" } };

            // Act
            var results = hybridStore.GetSimilarWithFilters(queryVector, topK: 10, filters).ToList();

            // Assert
            Assert.All(results, r => Assert.Equal("science", r.Metadata["category"]));
        }

        [Fact]
        public void GetSimilarWithFilters_WithNullFilters_ThrowsArgumentNullException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                hybridStore.GetSimilarWithFilters(queryVector, topK: 5, null!));
        }

        #endregion

        #region GetById Tests

        [Fact]
        public void GetById_WithExistingDocument_ReturnsDocument()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);
            var doc = CreateTestDocument("doc1", "Test content", 3);
            hybridStore.Add(doc);

            // Act
            var result = hybridStore.GetById("doc1");

            // Assert
            Assert.NotNull(result);
            Assert.Equal("doc1", result.Id);
            Assert.Equal("Test content", result.Content);
        }

        [Fact]
        public void GetById_WithNonExistingDocument_ReturnsNull()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act
            var result = hybridStore.GetById("nonexistent");

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void GetById_WithNullId_ThrowsArgumentException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => hybridStore.GetById(null!));
        }

        #endregion

        #region Remove Tests

        [Fact]
        public void Remove_WithExistingDocument_RemovesFromBothStores()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);
            var doc = CreateTestDocument("doc1", "Test", 3);
            hybridStore.Add(doc);

            // Act
            var result = hybridStore.Remove("doc1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, hybridStore.DocumentCount);
            Assert.Equal(0, vectorStore.DocumentCount);
            Assert.Equal(0, keywordStore.DocumentCount);
        }

        [Fact]
        public void Remove_WithNonExistingDocument_ReturnsFalse()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act
            var result = hybridStore.Remove("nonexistent");

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Remove_WithNullId_ThrowsArgumentException()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => hybridStore.Remove(null!));
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllDocumentsFromBothStores()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            hybridStore.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3),
                CreateTestDocument("doc3", "Content 3", 3)
            });

            // Act
            hybridStore.Clear();

            // Assert
            Assert.Equal(0, hybridStore.DocumentCount);
            Assert.Equal(0, vectorStore.DocumentCount);
            Assert.Equal(0, keywordStore.DocumentCount);
        }

        [Fact]
        public void Clear_OnEmptyStore_DoesNotThrow()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act & Assert
            hybridStore.Clear();
            Assert.Equal(0, hybridStore.DocumentCount);
        }

        #endregion

        #region GetAll Tests

        [Fact]
        public void GetAll_WithDocuments_ReturnsAllDocuments()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            hybridStore.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3),
                CreateTestDocument("doc3", "Content 3", 3)
            });

            // Act
            var results = hybridStore.GetAll().ToList();

            // Assert
            Assert.Equal(3, results.Count);
            Assert.Contains(results, d => d.Id == "doc1");
            Assert.Contains(results, d => d.Id == "doc2");
            Assert.Contains(results, d => d.Id == "doc3");
        }

        [Fact]
        public void GetAll_WithEmptyStore_ReturnsEmpty()
        {
            // Arrange
            var vectorStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var keywordStore = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var hybridStore = new HybridDocumentStore<float>(vectorStore, keywordStore, 0.7f, 0.3f);

            // Act
            var results = hybridStore.GetAll();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Helper Methods

        private VectorDocument<float> CreateTestDocument(
            string id,
            string content,
            int dimension,
            float[]? values = null)
        {
            var vector = values ?? Enumerable.Range(0, dimension).Select(i => (float)i).ToArray();
            return new VectorDocument<float>
            {
                Document = new Document<float>(id, content),
                Embedding = new Vector<float>(vector)
            };
        }

        #endregion
    }
}
