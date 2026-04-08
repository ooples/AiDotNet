using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    public class FAISSDocumentStoreTests
    {
        #region Constructor Tests

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultCapacity_CreatesStore()
        {
            // Arrange & Act
            var store = new FAISSDocumentStore<float>();

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(0, store.VectorDimension);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithCustomCapacity_CreatesStore()
        {
            // Arrange & Act
            var store = new FAISSDocumentStore<float>(initialCapacity: 5000);

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(0, store.VectorDimension);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroCapacity_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new FAISSDocumentStore<float>(initialCapacity: 0));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeCapacity_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new FAISSDocumentStore<float>(initialCapacity: -1));
        }

        #endregion

        #region Add Tests

        [Fact(Timeout = 60000)]
        public async Task Add_FirstDocument_SetsDimension()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var doc = CreateTestDocument("doc1", "Test content", 384);

            // Act
            store.Add(doc);

            // Assert
            Assert.Equal(1, store.DocumentCount);
            Assert.Equal(384, store.VectorDimension);
        }

        [Fact(Timeout = 60000)]
        public async Task Add_WithValidDocument_IncreasesCount()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var doc = CreateTestDocument("doc1", "Test content", 3);

            // Act
            store.Add(doc);

            // Assert
            Assert.Equal(1, store.DocumentCount);
        }

        [Fact(Timeout = 60000)]
        public async Task Add_WithNullDocument_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.Add(null!));
        }

        [Fact(Timeout = 60000)]
        public async Task Add_WithNullEmbedding_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var doc = new VectorDocument<float>
            {
                Document = new Document<float>("doc1", "Test"),
                Embedding = null!
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Add(doc));
        }

        [Fact(Timeout = 60000)]
        public async Task Add_WithEmptyDocumentId_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var doc = new VectorDocument<float>
            {
                Document = new Document<float>("", "Test"),
                Embedding = new Vector<float>(new float[] { 1, 2, 3 })
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Add(doc));
        }

        [Fact(Timeout = 60000)]
        public async Task Add_WithMismatchedDimension_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var doc1 = CreateTestDocument("doc1", "Test 1", 3);
            store.Add(doc1);

            var doc2 = CreateTestDocument("doc2", "Test 2", 5); // Wrong dimension

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => store.Add(doc2));
            Assert.Contains("dimension mismatch", exception.Message.ToLower());
        }

        [Fact(Timeout = 60000)]
        public async Task Add_WithDuplicateId_UpdatesDocument()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task Add_MultipleDocuments_MaintainsIndexing()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act
            for (int i = 0; i < 10; i++)
            {
                store.Add(CreateTestDocument($"doc{i}", $"Content {i}", 3));
            }

            // Assert
            Assert.Equal(10, store.DocumentCount);
            for (int i = 0; i < 10; i++)
            {
                var doc = store.GetById($"doc{i}");
                Assert.NotNull(doc);
                Assert.Equal($"Content {i}", doc.Content);
            }
        }

        #endregion

        #region AddBatch Tests

        [Fact(Timeout = 60000)]
        public async Task AddBatch_WithValidDocuments_IncreasesCount()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task AddBatch_FirstBatch_SetsDimension()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task AddBatch_WithNullCollection_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.AddBatch(null!));
        }

        [Fact(Timeout = 60000)]
        public async Task AddBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.AddBatch(new List<VectorDocument<float>>()));
        }

        [Fact(Timeout = 60000)]
        public async Task AddBatch_WithMismatchedDimensionsInBatch_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var docs = new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 5) // Different dimension
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.AddBatch(docs));
        }

        [Fact(Timeout = 60000)]
        public async Task AddBatch_WithMismatchedDimensionFromExisting_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            store.Add(CreateTestDocument("doc1", "Content 1", 3));

            var docs = new List<VectorDocument<float>>
            {
                CreateTestDocument("doc2", "Content 2", 5)
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.AddBatch(docs));
        }

        [Fact(Timeout = 60000)]
        public async Task AddBatch_WithLargeNumberOfDocuments_Succeeds()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>(initialCapacity: 10000);
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

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_WithMatchingDocuments_ReturnsTopK()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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
            Assert.Equal("doc1", results[0].Id); // Should match most similar
        }

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_WithEmptyStore_ReturnsEmpty()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 5);

            // Assert
            Assert.Empty(results);
        }

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_WithNullQueryVector_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.GetSimilar(null!, topK: 5));
        }

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_WithZeroTopK_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetSimilar(queryVector, topK: 0));
        }

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_OrdersByRelevanceDescending()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_ReturnsAtMostTopK()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            for (int i = 0; i < 10; i++)
            {
                store.Add(CreateTestDocument($"doc{i}", $"Content {i}", 3));
            }

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 5).ToList();

            // Assert
            Assert.Equal(5, results.Count);
        }

        #endregion

        #region GetSimilarWithFilters Tests

        [Fact(Timeout = 60000)]
        public async Task GetSimilarWithFilters_WithMatchingMetadata_ReturnsFilteredResults()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task GetSimilarWithFilters_WithNoMatchingMetadata_ReturnsEmpty()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task GetSimilarWithFilters_WithMultipleFilters_AppliesAll()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            var doc1 = CreateTestDocument("doc1", "Content 1", 3);
            doc1.Document.Metadata["category"] = "science";
            doc1.Document.Metadata["year"] = 2023;

            var doc2 = CreateTestDocument("doc2", "Content 2", 3);
            doc2.Document.Metadata["category"] = "science";
            doc2.Document.Metadata["year"] = 2020;

            store.Add(doc1);
            store.Add(doc2);

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

        [Fact(Timeout = 60000)]
        public async Task GetSimilarWithFilters_WithNullFilters_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                store.GetSimilarWithFilters(queryVector, topK: 5, null!));
        }

        #endregion

        #region GetById Tests

        [Fact(Timeout = 60000)]
        public async Task GetById_WithExistingDocument_ReturnsDocument()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var doc = CreateTestDocument("doc1", "Test content", 3);
            store.Add(doc);

            // Act
            var result = store.GetById("doc1");

            // Assert
            Assert.NotNull(result);
            Assert.Equal("doc1", result.Id);
            Assert.Equal("Test content", result.Content);
        }

        [Fact(Timeout = 60000)]
        public async Task GetById_WithNonExistingDocument_ReturnsNull()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act
            var result = store.GetById("nonexistent");

            // Assert
            Assert.Null(result);
        }

        [Fact(Timeout = 60000)]
        public async Task GetById_WithNullId_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetById(null!));
        }

        [Fact(Timeout = 60000)]
        public async Task GetById_WithEmptyId_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.GetById(""));
        }

        #endregion

        #region Remove Tests

        [Fact(Timeout = 60000)]
        public async Task Remove_WithExistingDocument_ReturnsTrue()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            var doc = CreateTestDocument("doc1", "Test", 3);
            store.Add(doc);

            // Act
            var result = store.Remove("doc1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, store.DocumentCount);
        }

        [Fact(Timeout = 60000)]
        public async Task Remove_WithNonExistingDocument_ReturnsFalse()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act
            var result = store.Remove("nonexistent");

            // Assert
            Assert.False(result);
        }

        [Fact(Timeout = 60000)]
        public async Task Remove_LastDocument_ResetsDimensionAndIndex()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task Remove_MiddleDocument_RebuildsIndex()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            store.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3),
                CreateTestDocument("doc3", "Content 3", 3)
            });

            // Act
            store.Remove("doc2");

            // Assert
            Assert.Equal(2, store.DocumentCount);
            Assert.NotNull(store.GetById("doc1"));
            Assert.Null(store.GetById("doc2"));
            Assert.NotNull(store.GetById("doc3"));
        }

        [Fact(Timeout = 60000)]
        public async Task Remove_WithNullId_ThrowsArgumentException()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Remove(null!));
        }

        #endregion

        #region Clear Tests

        [Fact(Timeout = 60000)]
        public async Task Clear_RemovesAllDocuments()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task Clear_ResetsVectorDimensionAndIndex()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            store.Add(CreateTestDocument("doc1", "Test", 3));

            // Act
            store.Clear();

            // Assert
            Assert.Equal(0, store.VectorDimension);

            // Should be able to add documents with different dimension
            store.Add(CreateTestDocument("doc2", "Test", 5));
            Assert.Equal(1, store.DocumentCount);
            Assert.Equal(5, store.VectorDimension);
        }

        [Fact(Timeout = 60000)]
        public async Task Clear_OnEmptyStore_DoesNotThrow()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act & Assert
            store.Clear();
            Assert.Equal(0, store.DocumentCount);
        }

        #endregion

        #region GetAll Tests

        [Fact(Timeout = 60000)]
        public async Task GetAll_WithDocuments_ReturnsAllDocuments()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
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

        [Fact(Timeout = 60000)]
        public async Task GetAll_WithEmptyStore_ReturnsEmpty()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();

            // Act
            var results = store.GetAll();

            // Assert
            Assert.Empty(results);
        }

        [Fact(Timeout = 60000)]
        public async Task GetAll_DoesNotIncludeEmbeddings()
        {
            // Arrange
            var store = new FAISSDocumentStore<float>();
            store.Add(CreateTestDocument("doc1", "Content", 3));

            // Act
            var results = store.GetAll().ToList();

            // Assert
            Assert.Single(results);
            // Document objects returned by GetAll should not have embeddings
            Assert.Null(results[0].Embedding);
        }

        #endregion

        #region Double Type Tests

        [Fact(Timeout = 60000)]
        public async Task Store_WithDoubleType_WorksCorrectly()
        {
            // Arrange
            var store = new FAISSDocumentStore<double>();
            var doc = new VectorDocument<double>
            {
                Document = new Document<double>("doc1", "Test"),
                Embedding = new Vector<double>(new double[] { 1.0, 2.0, 3.0 })
            };

            // Act
            store.Add(doc);

            // Assert
            Assert.Equal(1, store.DocumentCount);
            var retrieved = store.GetById("doc1");
            Assert.NotNull(retrieved);
        }

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_WithDoubleType_CalculatesCorrectly()
        {
            // Arrange
            var store = new FAISSDocumentStore<double>();
            store.Add(new VectorDocument<double>
            {
                Document = new Document<double>("doc1", "Test"),
                Embedding = new Vector<double>(new double[] { 1.0, 0.0, 0.0 })
            });

            var queryVector = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 1).ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
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
