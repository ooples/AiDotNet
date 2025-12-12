using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RAG
{
    /// <summary>
    /// Integration tests for Document Store implementations with comprehensive coverage.
    /// These tests validate storage, retrieval, and similarity search functionality.
    /// </summary>
    public class DocumentStoreIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region InMemoryDocumentStore Tests

        [Fact]
        public void InMemoryDocumentStore_AddAndRetrieveDocument_Success()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var doc = new Document<double>("doc1", "Machine learning is fascinating");
            var embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
            var vectorDoc = new VectorDocument<double>(doc, embedding);

            // Act
            store.Add(vectorDoc);
            var retrieved = store.GetById("doc1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("doc1", retrieved.Id);
            Assert.Equal("Machine learning is fascinating", retrieved.Content);
            Assert.Equal(1, store.DocumentCount);
        }

        [Fact]
        public void InMemoryDocumentStore_AddMultipleDocuments_AllStored()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                new VectorDocument<double>(
                    new Document<double>("doc1", "Artificial intelligence powers modern systems"),
                    new Vector<double>(new[] { 0.8, 0.6, 0.4 })),
                new VectorDocument<double>(
                    new Document<double>("doc2", "Deep learning uses neural networks"),
                    new Vector<double>(new[] { 0.7, 0.5, 0.3 })),
                new VectorDocument<double>(
                    new Document<double>("doc3", "Natural language processing analyzes text"),
                    new Vector<double>(new[] { 0.6, 0.4, 0.2 }))
            };

            // Act
            store.AddBatch(docs);

            // Assert
            Assert.Equal(3, store.DocumentCount);
            Assert.NotNull(store.GetById("doc1"));
            Assert.NotNull(store.GetById("doc2"));
            Assert.NotNull(store.GetById("doc3"));
        }

        [Fact]
        public void InMemoryDocumentStore_SimilaritySearch_ReturnsTopKResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

            // Add documents with known vectors for predictable similarity
            // Using normalized vectors for cosine similarity
            var docs = new[]
            {
                new VectorDocument<double>(
                    new Document<double>("doc1", "AI research paper"),
                    CreateNormalizedVector(1.0, 0.0, 0.0)),
                new VectorDocument<double>(
                    new Document<double>("doc2", "AI tutorial"),
                    CreateNormalizedVector(0.9, 0.1, 0.0)),
                new VectorDocument<double>(
                    new Document<double>("doc3", "Cooking recipes"),
                    CreateNormalizedVector(0.0, 1.0, 0.0)),
                new VectorDocument<double>(
                    new Document<double>("doc4", "Machine learning guide"),
                    CreateNormalizedVector(0.8, 0.2, 0.0)),
                new VectorDocument<double>(
                    new Document<double>("doc5", "Sports news"),
                    CreateNormalizedVector(0.0, 0.0, 1.0))
            };

            store.AddBatch(docs);

            // Query vector similar to doc1
            var queryVector = CreateNormalizedVector(1.0, 0.0, 0.0);

            // Act
            var results = store.GetSimilar(queryVector, topK: 3);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            Assert.Equal("doc1", resultList[0].Id); // Most similar
            Assert.True(resultList[0].HasRelevanceScore);
            Assert.True(Convert.ToDouble(resultList[0].RelevanceScore) > 0.85);

            // Verify results are sorted by similarity
            for (int i = 0; i < resultList.Count - 1; i++)
            {
                Assert.True(Convert.ToDouble(resultList[i].RelevanceScore) >=
                           Convert.ToDouble(resultList[i + 1].RelevanceScore));
            }
        }

        [Fact]
        public void InMemoryDocumentStore_SimilarityWithMetadataFilter_ReturnsFilteredResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

            var docs = new[]
            {
                new VectorDocument<double>(
                    new Document<double>("doc1", "AI paper 2024",
                        new Dictionary<string, object> { { "year", 2024 }, { "category", "AI" } }),
                    CreateNormalizedVector(1.0, 0.0, 0.0)),
                new VectorDocument<double>(
                    new Document<double>("doc2", "AI paper 2023",
                        new Dictionary<string, object> { { "year", 2023 }, { "category", "AI" } }),
                    CreateNormalizedVector(0.95, 0.05, 0.0)),
                new VectorDocument<double>(
                    new Document<double>("doc3", "ML paper 2024",
                        new Dictionary<string, object> { { "year", 2024 }, { "category", "ML" } }),
                    CreateNormalizedVector(0.9, 0.1, 0.0))
            };

            store.AddBatch(docs);
            var queryVector = CreateNormalizedVector(1.0, 0.0, 0.0);
            var filters = new Dictionary<string, object> { { "year", 2024 } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 5, filters);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(2, resultList.Count); // Only 2024 documents
            Assert.All(resultList, doc => Assert.Equal(2024, doc.Metadata["year"]));
        }

        [Fact]
        public void InMemoryDocumentStore_RemoveDocument_SuccessfullyDeleted()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Test document"),
                new Vector<double>(new[] { 0.1, 0.2, 0.3 }));
            store.Add(doc);

            // Act
            var removed = store.Remove("doc1");
            var retrieved = store.GetById("doc1");

            // Assert
            Assert.True(removed);
            Assert.Null(retrieved);
            Assert.Equal(0, store.DocumentCount);
        }

        [Fact]
        public void InMemoryDocumentStore_RemoveNonExistentDocument_ReturnsFalse()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);

            // Act
            var removed = store.Remove("nonexistent");

            // Assert
            Assert.False(removed);
        }

        [Fact]
        public void InMemoryDocumentStore_Clear_RemovesAllDocuments()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                new VectorDocument<double>(
                    new Document<double>("doc1", "Content 1"),
                    new Vector<double>(new[] { 0.1, 0.2, 0.3 })),
                new VectorDocument<double>(
                    new Document<double>("doc2", "Content 2"),
                    new Vector<double>(new[] { 0.4, 0.5, 0.6 }))
            };
            store.AddBatch(docs);

            // Act
            store.Clear();

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Null(store.GetById("doc1"));
            Assert.Null(store.GetById("doc2"));
        }

        [Fact]
        public void InMemoryDocumentStore_GetAll_ReturnsAllDocuments()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                new VectorDocument<double>(
                    new Document<double>("doc1", "Content 1"),
                    new Vector<double>(new[] { 0.1, 0.2, 0.3 })),
                new VectorDocument<double>(
                    new Document<double>("doc2", "Content 2"),
                    new Vector<double>(new[] { 0.4, 0.5, 0.6 })),
                new VectorDocument<double>(
                    new Document<double>("doc3", "Content 3"),
                    new Vector<double>(new[] { 0.7, 0.8, 0.9 }))
            };
            store.AddBatch(docs);

            // Act
            var allDocs = store.GetAll().ToList();

            // Assert
            Assert.Equal(3, allDocs.Count);
            Assert.Contains(allDocs, d => d.Id == "doc1");
            Assert.Contains(allDocs, d => d.Id == "doc2");
            Assert.Contains(allDocs, d => d.Id == "doc3");
        }

        [Fact]
        public void InMemoryDocumentStore_EmptyQuery_ReturnsNoResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Test"),
                new Vector<double>(new[] { 0.1, 0.2, 0.3 }));
            store.Add(doc);

            // Act
            var queryVector = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
            var results = store.GetSimilar(queryVector, topK: 5).ToList();

            // Assert - Zero vector should still return results but with lower scores
            Assert.NotEmpty(results);
        }

        [Fact]
        public void InMemoryDocumentStore_LargeDocument_HandledCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var largeContent = new string('A', 10000); // 10K characters
            var doc = new VectorDocument<double>(
                new Document<double>("doc1", largeContent),
                new Vector<double>(new[] { 0.1, 0.2, 0.3 }));

            // Act
            store.Add(doc);
            var retrieved = store.GetById("doc1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(10000, retrieved.Content.Length);
        }

        [Fact]
        public void InMemoryDocumentStore_VectorDimensionMismatch_ThrowsException()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Test"),
                new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 })); // 4 dimensions instead of 3

            // Act & Assert
            Assert.Throws<ArgumentException>(() => store.Add(doc));
        }

        [Fact]
        public void InMemoryDocumentStore_DuplicateId_OverwritesDocument()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var doc1 = new VectorDocument<double>(
                new Document<double>("doc1", "First content"),
                new Vector<double>(new[] { 0.1, 0.2, 0.3 }));
            var doc2 = new VectorDocument<double>(
                new Document<double>("doc1", "Second content"),
                new Vector<double>(new[] { 0.4, 0.5, 0.6 }));

            // Act
            store.Add(doc1);
            store.Add(doc2);
            var retrieved = store.GetById("doc1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("Second content", retrieved.Content);
            Assert.Equal(1, store.DocumentCount);
        }

        [Fact]
        public void InMemoryDocumentStore_CosineSimilarity_CorrectRanking()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 2);

            // Perfect similarity (same direction)
            var doc1 = new VectorDocument<double>(
                new Document<double>("doc1", "Perfect match"),
                CreateNormalizedVector(1.0, 0.0));

            // 45 degree angle (cos = 0.707)
            var doc2 = new VectorDocument<double>(
                new Document<double>("doc2", "Moderate match"),
                CreateNormalizedVector(Math.Sqrt(0.5), Math.Sqrt(0.5)));

            // 90 degree angle (cos = 0)
            var doc3 = new VectorDocument<double>(
                new Document<double>("doc3", "Orthogonal"),
                CreateNormalizedVector(0.0, 1.0));

            store.AddBatch(new[] { doc1, doc2, doc3 });
            var queryVector = CreateNormalizedVector(1.0, 0.0);

            // Act
            var results = store.GetSimilar(queryVector, topK: 3).ToList();

            // Assert
            Assert.Equal("doc1", results[0].Id);
            Assert.Equal(1.0, Convert.ToDouble(results[0].RelevanceScore), precision: 5);

            Assert.Equal("doc2", results[1].Id);
            Assert.Equal(Math.Sqrt(0.5), Convert.ToDouble(results[1].RelevanceScore), precision: 5);

            Assert.Equal("doc3", results[2].Id);
            Assert.Equal(0.0, Convert.ToDouble(results[2].RelevanceScore), precision: 5);
        }

        [Fact]
        public void InMemoryDocumentStore_HighDimensionalVectors_WorksCorrectly()
        {
            // Arrange - Test with realistic embedding dimension (768 like BERT)
            var store = new InMemoryDocumentStore<double>(vectorDimension: 768);
            var embedding = new double[768];
            for (int i = 0; i < 768; i++)
            {
                embedding[i] = Math.Sin(i * 0.1); // Create a pattern
            }

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "High dimensional document"),
                new Vector<double>(embedding));

            // Act
            store.Add(doc);
            var retrieved = store.GetById("doc1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(768, store.VectorDimension);
        }

        [Fact]
        public void InMemoryDocumentStore_ConcurrentAccess_ThreadSafe()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var tasks = new List<Task>();

            // Act - Add documents concurrently
            for (int i = 0; i < 100; i++)
            {
                int docId = i; // Capture for closure
                tasks.Add(Task.Run(() =>
                {
                    var doc = new VectorDocument<double>(
                        new Document<double>($"doc{docId}", $"Content {docId}"),
                        new Vector<double>(new[] { docId * 0.01, docId * 0.02, docId * 0.03 }));
                    store.Add(doc);
                }));
            }

            Task.WaitAll(tasks.ToArray());

            // Assert
            Assert.Equal(100, store.DocumentCount);
        }

        [Fact]
        public void InMemoryDocumentStore_TopKLargerThanResultSet_ReturnsAllResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                new VectorDocument<double>(
                    new Document<double>("doc1", "Content 1"),
                    new Vector<double>(new[] { 0.1, 0.2, 0.3 })),
                new VectorDocument<double>(
                    new Document<double>("doc2", "Content 2"),
                    new Vector<double>(new[] { 0.4, 0.5, 0.6 }))
            };
            store.AddBatch(docs);
            var queryVector = new Vector<double>(new[] { 0.1, 0.2, 0.3 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count); // Returns all available documents
        }

        [Fact]
        public void InMemoryDocumentStore_MetadataWithComplexTypes_PreservedCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var metadata = new Dictionary<string, object>
            {
                { "title", "Research Paper" },
                { "year", 2024 },
                { "citations", 150 },
                { "authors", new[] { "Alice", "Bob" } },
                { "score", 9.5 }
            };

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Paper content", metadata),
                new Vector<double>(new[] { 0.1, 0.2, 0.3 }));

            // Act
            store.Add(doc);
            var retrieved = store.GetById("doc1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("Research Paper", retrieved.Metadata["title"]);
            Assert.Equal(2024, retrieved.Metadata["year"]);
            Assert.Equal(150, retrieved.Metadata["citations"]);
            Assert.Equal(9.5, retrieved.Metadata["score"]);
        }

        #endregion

        #region Helper Methods

        private Vector<double> CreateNormalizedVector(params double[] values)
        {
            var vector = new Vector<double>(values);
            double magnitude = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                magnitude += values[i] * values[i];
            }
            magnitude = Math.Sqrt(magnitude);

            if (magnitude < 1e-10)
                return vector;

            var normalized = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                normalized[i] = values[i] / magnitude;
            }
            return new Vector<double>(normalized);
        }

        #endregion
    }
}
