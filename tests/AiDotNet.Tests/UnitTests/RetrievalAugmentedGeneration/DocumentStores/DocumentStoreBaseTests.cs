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
    /// <summary>
    /// Base tests for DocumentStoreBase functionality that applies to all derived stores.
    /// This tests the common validation and functionality provided by the base class.
    /// </summary>
    public class DocumentStoreBaseTests
    {
        #region Validation Tests

        [Fact]
        public void Add_WithNullVectorDocument_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.Add(null!));
        }

        [Fact]
        public void Add_WithNullDocument_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var vectorDoc = new VectorDocument<float>
            {
                Document = null!,
                Embedding = new Vector<float>(new float[] { 1, 2, 3 })
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Add(vectorDoc));
        }

        [Fact]
        public void Add_WithNullEmbedding_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var vectorDoc = new VectorDocument<float>
            {
                Document = new Document<float>("doc1", "Test"),
                Embedding = null!
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Add(vectorDoc));
        }

        [Fact]
        public void Add_WithEmptyDocumentId_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var vectorDoc = new VectorDocument<float>
            {
                Document = new Document<float>("", "Test"),
                Embedding = new Vector<float>(new float[] { 1, 2, 3 })
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Add(vectorDoc));
        }

        [Fact]
        public void Add_WithWhitespaceDocumentId_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var vectorDoc = new VectorDocument<float>
            {
                Document = new Document<float>("   ", "Test"),
                Embedding = new Vector<float>(new float[] { 1, 2, 3 })
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Add(vectorDoc));
        }

        [Fact]
        public void GetSimilar_WithNullQueryVector_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.GetSimilar(null!, topK: 5));
        }

        [Fact]
        public void GetSimilar_WithZeroTopK_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetSimilar(queryVector, topK: 0));
        }

        [Fact]
        public void GetSimilar_WithNegativeTopK_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetSimilar(queryVector, topK: -1));
        }

        [Fact]
        public void GetSimilarWithFilters_WithNullFilters_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                store.GetSimilarWithFilters(queryVector, topK: 5, null!));
        }

        [Fact]
        public void GetById_WithNullId_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetById(null!));
        }

        [Fact]
        public void GetById_WithEmptyId_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetById(""));
        }

        [Fact]
        public void GetById_WithWhitespaceId_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetById("   "));
        }

        [Fact]
        public void Remove_WithNullId_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Remove(null!));
        }

        [Fact]
        public void Remove_WithEmptyId_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Remove(""));
        }

        [Fact]
        public void AddBatch_WithNullCollection_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.AddBatch(null!));
        }

        [Fact]
        public void AddBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.AddBatch(new List<VectorDocument<float>>()));
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void GetSimilarWithFilters_WithStringMetadata_FiltersCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc1 = CreateTestDocument("doc1", "Content 1", 3);
            doc1.Document.Metadata["category"] = "science";

            var doc2 = CreateTestDocument("doc2", "Content 2", 3);
            doc2.Document.Metadata["category"] = "history";

            store.Add(doc1);
            store.Add(doc2);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "category", "science" } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters).ToList();

            // Assert
            Assert.All(results, r => Assert.Equal("science", r.Metadata["category"]));
        }

        [Fact]
        public void GetSimilarWithFilters_WithBooleanMetadata_FiltersCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc1 = CreateTestDocument("doc1", "Content 1", 3);
            doc1.Document.Metadata["is_published"] = true;

            var doc2 = CreateTestDocument("doc2", "Content 2", 3);
            doc2.Document.Metadata["is_published"] = false;

            store.Add(doc1);
            store.Add(doc2);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "is_published", true } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters).ToList();

            // Assert
            Assert.All(results, r => Assert.Equal(true, r.Metadata["is_published"]));
        }

        [Fact]
        public void GetSimilarWithFilters_WithNumericComparison_FiltersCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            var doc1 = CreateTestDocument("doc1", "Content 1", 3);
            doc1.Document.Metadata["year"] = 2020;

            var doc2 = CreateTestDocument("doc2", "Content 2", 3);
            doc2.Document.Metadata["year"] = 2023;

            store.Add(doc1);
            store.Add(doc2);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "year", 2022 } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc2", results[0].Id); // Only doc2 has year >= 2022
        }

        [Fact]
        public void GetSimilarWithFilters_WithMultipleFilters_AppliesAllFilters()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);

            var doc1 = CreateTestDocument("doc1", "Content 1", 3);
            doc1.Document.Metadata["category"] = "science";
            doc1.Document.Metadata["year"] = 2023;

            var doc2 = CreateTestDocument("doc2", "Content 2", 3);
            doc2.Document.Metadata["category"] = "science";
            doc2.Document.Metadata["year"] = 2020;

            var doc3 = CreateTestDocument("doc3", "Content 3", 3);
            doc3.Document.Metadata["category"] = "history";
            doc3.Document.Metadata["year"] = 2023;

            store.Add(doc1);
            store.Add(doc2);
            store.Add(doc3);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object>
            {
                { "category", "science" },
                { "year", 2022 }
            };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void GetSimilarWithFilters_WithMissingMetadataKey_ReturnsEmpty()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc = CreateTestDocument("doc1", "Content 1", 3);
            doc.Document.Metadata["category"] = "science";
            store.Add(doc);

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "nonexistent", "value" } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void GetSimilarWithFilters_WithEmptyFilters_ReturnsSameAsGetSimilar()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            store.Add(CreateTestDocument("doc1", "Content 1", 3));
            store.Add(CreateTestDocument("doc2", "Content 2", 3));

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var resultsWithEmptyFilters = store.GetSimilarWithFilters(
                queryVector, topK: 10, new Dictionary<string, object>()).ToList();
            var resultsWithoutFilters = store.GetSimilar(queryVector, topK: 10).ToList();

            // Assert
            Assert.Equal(resultsWithoutFilters.Count, resultsWithEmptyFilters.Count);
        }

        #endregion

        #region GetAll Tests

        [Fact]
        public void GetAll_ReturnsDocumentsWithoutEmbeddings()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            store.Add(CreateTestDocument("doc1", "Content 1", 3));

            // Act
            var results = store.GetAll().ToList();

            // Assert
            Assert.Single(results);
            Assert.Null(results[0].Embedding); // GetAll should not return embeddings
        }

        [Fact]
        public void GetAll_WithMultipleDocuments_ReturnsAll()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            for (int i = 0; i < 10; i++)
            {
                store.Add(CreateTestDocument($"doc{i}", $"Content {i}", 3));
            }

            // Act
            var results = store.GetAll().ToList();

            // Assert
            Assert.Equal(10, results.Count);
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
