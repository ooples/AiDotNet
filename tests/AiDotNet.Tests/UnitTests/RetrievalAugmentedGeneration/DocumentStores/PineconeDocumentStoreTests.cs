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
    public class PineconeDocumentStoreTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidIndexName_CreatesStore()
        {
            // Arrange & Act
            var store = new PineconeDocumentStore<float>("TestIndex");

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(0, store.VectorDimension);
        }

        [Fact]
        public void Constructor_WithCustomCapacity_CreatesStore()
        {
            // Arrange & Act
            var store = new PineconeDocumentStore<float>("TestIndex", initialCapacity: 5000);

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(0, store.VectorDimension);
        }

        [Fact]
        public void Constructor_WithEmptyIndexName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new PineconeDocumentStore<float>(""));
        }

        [Fact]
        public void Constructor_WithNullIndexName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new PineconeDocumentStore<float>(null!));
        }

        [Fact]
        public void Constructor_WithWhitespaceIndexName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new PineconeDocumentStore<float>("   "));
        }

        [Fact]
        public void Constructor_WithZeroCapacity_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new PineconeDocumentStore<float>("TestIndex", initialCapacity: 0));
        }

        [Fact]
        public void Constructor_WithNegativeCapacity_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new PineconeDocumentStore<float>("TestIndex", initialCapacity: -1));
        }

        #endregion

        #region Add Tests

        [Fact]
        public void Add_FirstDocument_SetsDimension()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var doc = CreateTestDocument("doc1", "Test content", 384);

            // Act
            store.Add(doc);

            // Assert
            Assert.Equal(1, store.DocumentCount);
            Assert.Equal(384, store.VectorDimension);
        }

        [Fact]
        public void Add_WithValidDocument_IncreasesCount()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var doc = CreateTestDocument("doc1", "Test content", 3);

            // Act
            store.Add(doc);

            // Assert
            Assert.Equal(1, store.DocumentCount);
        }

        [Fact]
        public void Add_WithNullDocument_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.Add(null!));
        }

        [Fact]
        public void Add_WithMismatchedDimension_ThrowsArgumentException()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var doc1 = CreateTestDocument("doc1", "Test 1", 3);
            store.Add(doc1);

            var doc2 = CreateTestDocument("doc2", "Test 2", 5); // Wrong dimension

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => store.Add(doc2));
            Assert.Contains("dimension mismatch", exception.Message.ToLower());
        }

        [Fact]
        public void Add_WithDuplicateId_UpdatesDocument()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var doc1 = CreateTestDocument("doc1", "Original content", 3);
            var doc2 = CreateTestDocument("doc1", "Updated content", 3);

            // Act
            store.Add(doc1);
            store.Add(doc2);

            // Assert
            Assert.Equal(1, store.DocumentCount);
            var retrieved = store.GetById("doc1");
            Assert.NotNull(retrieved);
            Assert.Equal("Updated content", retrieved.Content);
        }

        #endregion

        #region AddBatch Tests

        [Fact]
        public void AddBatch_WithValidDocuments_IncreasesCount()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
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

        [Fact]
        public void AddBatch_FirstBatch_SetsDimension()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var docs = new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 384),
                CreateTestDocument("doc2", "Content 2", 384)
            };

            // Act
            store.AddBatch(docs);

            // Assert
            Assert.Equal(2, store.DocumentCount);
            Assert.Equal(384, store.VectorDimension);
        }

        [Fact]
        public void AddBatch_WithNullCollection_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.AddBatch(null!));
        }

        [Fact]
        public void AddBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.AddBatch(new List<VectorDocument<float>>()));
        }

        [Fact]
        public void AddBatch_WithMismatchedDimensionsInBatch_ThrowsArgumentException()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var docs = new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 5) // Different dimension
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.AddBatch(docs));
        }

        [Fact]
        public void AddBatch_WithLargeNumberOfDocuments_Succeeds()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex", initialCapacity: 10000);
            var docs = Enumerable.Range(0, 1000)
                .Select(i => CreateTestDocument($"doc{i}", $"Content {i}", 384))
                .ToList();

            // Act
            store.AddBatch(docs);

            // Assert
            Assert.Equal(1000, store.DocumentCount);
        }

        #endregion

        #region GetSimilar Tests

        [Fact]
        public void GetSimilar_WithMatchingDocuments_ReturnsTopK()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
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
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void GetSimilar_WithEmptyStore_ReturnsEmpty()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 5);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void GetSimilar_WithNullQueryVector_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.GetSimilar(null!, topK: 5));
        }

        [Fact]
        public void GetSimilar_OrdersByRelevanceDescending()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            store.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3, new float[] { 1, 0, 0 }),
                CreateTestDocument("doc2", "Content 2", 3, new float[] { 0.9f, 0.1f, 0 }),
                CreateTestDocument("doc3", "Content 3", 3, new float[] { 0, 1, 0 })
            });

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 3).ToList();

            // Assert
            Assert.Equal(3, results.Count);
            // Verify descending order
            for (int i = 0; i < results.Count - 1; i++)
            {
                Assert.True(Convert.ToDouble(results[i].RelevanceScore) >=
                           Convert.ToDouble(results[i + 1].RelevanceScore));
            }
        }

        #endregion

        #region GetSimilarWithFilters Tests

        [Fact]
        public void GetSimilarWithFilters_WithMatchingMetadata_ReturnsFilteredResults()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var doc1 = CreateTestDocument("doc1", "Content 1", 3, new float[] { 1, 0, 0 });
            doc1.Document.Metadata["category"] = "science";

            var doc2 = CreateTestDocument("doc2", "Content 2", 3, new float[] { 1, 0, 0 });
            doc2.Document.Metadata["category"] = "history";

            store.Add(doc1);
            store.Add(doc2);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "category", "science" } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void GetSimilarWithFilters_WithNoMatchingMetadata_ReturnsEmpty()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            var doc = CreateTestDocument("doc1", "Content 1", 3);
            doc.Document.Metadata["category"] = "science";
            store.Add(doc);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "category", "history" } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters);

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region GetById Tests

        [Fact]
        public void GetById_WithExistingDocument_ReturnsDocument()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
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
            var store = new PineconeDocumentStore<float>("TestIndex");

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
            var store = new PineconeDocumentStore<float>("TestIndex");
            var doc = CreateTestDocument("doc1", "Test", 3);
            store.Add(doc);

            // Act
            var result = store.Remove("doc1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, store.DocumentCount);
        }

        [Fact]
        public void Remove_LastDocument_ResetsDimension()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            store.Add(CreateTestDocument("doc1", "Test", 3));

            // Act
            store.Remove("doc1");

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(0, store.VectorDimension);

            // Should be able to add documents with different dimension
            store.Add(CreateTestDocument("doc2", "Test", 5));
            Assert.Equal(5, store.VectorDimension);
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllDocuments()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
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
            Assert.Equal(0, store.VectorDimension);
        }

        [Fact]
        public void Clear_AllowsNewDimensionAfterClear()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
            store.Add(CreateTestDocument("doc1", "Test", 3));
            store.Clear();

            // Act
            store.Add(CreateTestDocument("doc2", "Test", 5));

            // Assert
            Assert.Equal(1, store.DocumentCount);
            Assert.Equal(5, store.VectorDimension);
        }

        #endregion

        #region GetAll Tests

        [Fact]
        public void GetAll_WithDocuments_ReturnsAllDocuments()
        {
            // Arrange
            var store = new PineconeDocumentStore<float>("TestIndex");
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
            var store = new PineconeDocumentStore<float>("TestIndex");

            // Act
            var results = store.GetAll();

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
