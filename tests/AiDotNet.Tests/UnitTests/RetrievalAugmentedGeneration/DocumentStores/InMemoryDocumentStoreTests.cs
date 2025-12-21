using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    public class InMemoryDocumentStoreTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidDimension_CreatesStore()
        {
            // Arrange & Act
            var store = new InMemoryDocumentStore<float>(vectorDimension: 384);

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(384, store.VectorDimension);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new InMemoryDocumentStore<float>(vectorDimension: 0));
        }

        [Fact]
        public void Constructor_WithNegativeDimension_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new InMemoryDocumentStore<float>(vectorDimension: -1));
        }

        #endregion

        #region Add Tests

        [Fact]
        public void Add_WithValidDocument_IncreasesCount()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc = CreateTestDocument("doc1", "Test content", 3);

            // Act
            store.Add(doc);

            // Assert
            Assert.Equal(1, store.DocumentCount);
        }

        [Fact]
        public void Add_WithMismatchedDimension_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc1 = CreateTestDocument("doc1", "Test 1", 3);
            store.Add(doc1);

            var doc2 = CreateTestDocument("doc2", "Test 2", 5);

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => store.Add(doc2));
            Assert.Contains("dimension mismatch", exception.Message.ToLower());
        }

        #endregion

        #region AddBatch Tests

        [Fact]
        public void AddBatch_WithValidDocuments_IncreasesCount()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var docs = new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3),
                CreateTestDocument("doc3", "Content 3", 3)
            };

            // Act
            store.AddBatch(docs);

            // Assert
            Assert.Equal(3, store.DocumentCount);
        }

        #endregion

        #region GetSimilar Tests

        [Fact]
        public void GetSimilar_WithMatchingDocuments_ReturnsTopK()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            store.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3, new float[] { 1, 0, 0 }),
                CreateTestDocument("doc2", "Content 2", 3, new float[] { 0, 1, 0 }),
                CreateTestDocument("doc3", "Content 3", 3, new float[] { 0, 0, 1 })
            });

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 2).ToList();

            // Assert
            Assert.Equal(2, results.Count);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void GetSimilar_WithEmptyStore_ReturnsEmpty()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 5);

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region GetById Tests

        [Fact]
        public void GetById_WithExistingDocument_ReturnsDocument()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc = CreateTestDocument("doc1", "Test content", 3);
            store.Add(doc);

            // Act
            var result = store.GetById("doc1");

            // Assert
            Assert.NotNull(result);
            Assert.Equal("doc1", result.Id);
            Assert.Equal("Test content", result.Content);
        }

        [Fact]
        public void GetById_WithNonExistingDocument_ReturnsNull()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act
            var result = store.GetById("nonexistent");

            // Assert
            Assert.Null(result);
        }

        #endregion

        #region Remove Tests

        [Fact]
        public void Remove_WithExistingDocument_ReturnsTrue()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc = CreateTestDocument("doc1", "Test", 3);
            store.Add(doc);

            // Act
            var result = store.Remove("doc1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, store.DocumentCount);
        }

        [Fact]
        public void Remove_WithNonExistingDocument_ReturnsFalse()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act
            var result = store.Remove("nonexistent");

            // Assert
            Assert.False(result);
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllDocuments()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            store.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3),
                CreateTestDocument("doc3", "Content 3", 3)
            });

            // Act
            store.Clear();

            // Assert
            Assert.Equal(0, store.DocumentCount);
        }

        #endregion

        #region GetAll Tests

        [Fact]
        public void GetAll_WithDocuments_ReturnsAllDocuments()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            store.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3),
                CreateTestDocument("doc3", "Content 3", 3)
            });

            // Act
            var results = store.GetAll().ToList();

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
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act
            var results = store.GetAll();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Thread Safety Tests

        [Fact]
        public async Task ConcurrentAdd_WithMultipleThreads_AllDocumentsAdded()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var tasks = new List<Task>();

            // Act
            for (int i = 0; i < 100; i++)
            {
                var index = i;
                tasks.Add(Task.Run(() =>
                {
                    var doc = CreateTestDocument($"doc{index}", $"Content {index}", 3);
                    store.Add(doc);
                }));
            }

            await Task.WhenAll(tasks);

            // Assert
            Assert.Equal(100, store.DocumentCount);
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
