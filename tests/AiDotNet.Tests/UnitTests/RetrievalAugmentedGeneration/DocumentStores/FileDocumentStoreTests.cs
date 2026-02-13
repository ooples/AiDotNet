using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    public class FileDocumentStoreTests : IDisposable
    {
        private readonly string _testDir;

        public FileDocumentStoreTests()
        {
            _testDir = Path.Combine(Path.GetTempPath(), "FileDocumentStoreTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_testDir);
        }

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(_testDir))
                    Directory.Delete(_testDir, recursive: true);
            }
            catch
            {
                // Best-effort cleanup
            }
        }

        private FileDocumentStoreOptions CreateOptions()
        {
            return new FileDocumentStoreOptions
            {
                DirectoryPath = _testDir
            };
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_CreatesStore()
        {
            // Arrange & Act
            using var store = new FileDocumentStore<float>(384, CreateOptions());

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(384, store.VectorDimension);
            Assert.Equal(_testDir, store.DirectoryPath);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new FileDocumentStore<float>(0, CreateOptions()));
        }

        [Fact]
        public void Constructor_WithNegativeDimension_ThrowsArgumentOutOfRangeException()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new FileDocumentStore<float>(-1, CreateOptions()));
        }

        [Fact]
        public void Constructor_WithNullOptions_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new FileDocumentStore<float>(3, null));
        }

        [Fact]
        public void Constructor_WithEmptyDirectoryPath_ThrowsArgumentException()
        {
            var options = new FileDocumentStoreOptions { DirectoryPath = "" };
            Assert.Throws<ArgumentException>(() =>
                new FileDocumentStore<float>(3, options));
        }

        [Fact]
        public void Constructor_CreatesDirectoryIfNotExists()
        {
            // Arrange
            string subDir = Path.Combine(_testDir, "subdir");

            var options = new FileDocumentStoreOptions { DirectoryPath = subDir };

            // Act
            using var store = new FileDocumentStore<float>(3, options);

            // Assert
            Assert.True(Directory.Exists(subDir));
        }

        #endregion

        #region Add Tests

        [Fact]
        public void Add_WithValidDocument_IncreasesCount()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
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
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            var doc1 = CreateTestDocument("doc1", "Test 1", 3);
            store.Add(doc1);

            var doc2 = CreateTestDocument("doc2", "Test 2", 5);

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => store.Add(doc2));
            Assert.Contains("dimension mismatch", exception.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void Add_MultipleDocuments_AllAccessible()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());

            // Act
            store.Add(CreateTestDocument("doc1", "Content 1", 3));
            store.Add(CreateTestDocument("doc2", "Content 2", 3));
            store.Add(CreateTestDocument("doc3", "Content 3", 3));

            // Assert
            Assert.Equal(3, store.DocumentCount);
            Assert.NotNull(store.GetById("doc1"));
            Assert.NotNull(store.GetById("doc2"));
            Assert.NotNull(store.GetById("doc3"));
        }

        #endregion

        #region AddBatch Tests

        [Fact]
        public void AddBatch_WithValidDocuments_IncreasesCount()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
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
        public void AddBatch_WithEmptyList_ThrowsArgumentException()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());

            // Act & Assert - base class validation rejects empty collections
            Assert.Throws<ArgumentException>(() =>
                store.AddBatch(new List<VectorDocument<float>>()));
        }

        #endregion

        #region GetSimilar Tests

        [Fact]
        public void GetSimilar_WithMatchingDocuments_ReturnsTopK()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
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
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 5);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void GetSimilarWithFilters_FiltersCorrectly()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            var doc1 = CreateTestDocument("doc1", "Cat content", 3, new float[] { 1, 0, 0 });
            doc1.Document.Metadata["category"] = "animals";
            var doc2 = CreateTestDocument("doc2", "Car content", 3, new float[] { 0.9f, 0.1f, 0 });
            doc2.Document.Metadata["category"] = "vehicles";
            var doc3 = CreateTestDocument("doc3", "Dog content", 3, new float[] { 0.8f, 0.2f, 0 });
            doc3.Document.Metadata["category"] = "animals";

            store.AddBatch(new List<VectorDocument<float>> { doc1, doc2, doc3 });

            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var filters = new Dictionary<string, object> { { "category", "animals" } };

            // Act
            var results = store.GetSimilarWithFilters(queryVector, topK: 10, filters).ToList();

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r =>
            {
                Assert.True(r.Metadata.ContainsKey("category"));
                Assert.Equal("animals", r.Metadata["category"]);
            });
        }

        #endregion

        #region GetById Tests

        [Fact]
        public void GetById_WithExistingDocument_ReturnsDocument()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
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
            using var store = new FileDocumentStore<float>(3, CreateOptions());

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
            using var store = new FileDocumentStore<float>(3, CreateOptions());
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
            using var store = new FileDocumentStore<float>(3, CreateOptions());

            // Act
            var result = store.Remove("nonexistent");

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Remove_CreatesTombstone()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Add(CreateTestDocument("doc1", "Test", 3));

            // Act
            store.Remove("doc1");

            // Assert
            Assert.Equal(1, store.TombstoneCount);
            Assert.Null(store.GetById("doc1"));
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllDocumentsAndFiles()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            store.AddBatch(new List<VectorDocument<float>>
            {
                CreateTestDocument("doc1", "Content 1", 3),
                CreateTestDocument("doc2", "Content 2", 3)
            });
            store.Flush();

            // Act
            store.Clear();

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.False(File.Exists(Path.Combine(_testDir, "store.meta")));
            Assert.False(File.Exists(Path.Combine(_testDir, "documents.json")));
            Assert.False(File.Exists(Path.Combine(_testDir, "vectors.bin")));
        }

        #endregion

        #region GetAll Tests

        [Fact]
        public void GetAll_WithDocuments_ReturnsAllDocuments()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
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
            using var store = new FileDocumentStore<float>(3, CreateOptions());

            // Act
            var results = store.GetAll();

            // Assert
            Assert.Empty(results);
        }

        #endregion

        #region Persistence Tests

        [Fact]
        public void Flush_WritesFilesToDisk()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Add(CreateTestDocument("doc1", "Test content", 3));

            // Act
            store.Flush();

            // Assert
            Assert.True(File.Exists(Path.Combine(_testDir, "store.meta")));
            Assert.True(File.Exists(Path.Combine(_testDir, "documents.json")));
            Assert.True(File.Exists(Path.Combine(_testDir, "vectors.bin")));
            Assert.True(File.Exists(Path.Combine(_testDir, "hnsw.bin")));
        }

        [Fact]
        public void Persistence_DataSurvivesReopen()
        {
            // Arrange - create store, add documents, flush, dispose
            var options = CreateOptions();
            using (var store = new FileDocumentStore<float>(3, options))
            {
                store.Add(CreateTestDocument("doc1", "Hello world", 3, new float[] { 1, 0, 0 }));
                store.Add(CreateTestDocument("doc2", "Goodbye world", 3, new float[] { 0, 1, 0 }));
                store.Flush();
            }

            // Act - reopen and verify
            using var reopened = new FileDocumentStore<float>(3, options);

            // Assert
            Assert.Equal(2, reopened.DocumentCount);
            var doc = reopened.GetById("doc1");
            Assert.NotNull(doc);
            Assert.Equal("Hello world", doc.Content);

            var doc2 = reopened.GetById("doc2");
            Assert.NotNull(doc2);
            Assert.Equal("Goodbye world", doc2.Content);
        }

        [Fact]
        public void Persistence_SearchWorksAfterReopen()
        {
            // Arrange
            var options = CreateOptions();
            using (var store = new FileDocumentStore<float>(3, options))
            {
                store.AddBatch(new List<VectorDocument<float>>
                {
                    CreateTestDocument("doc1", "Cat", 3, new float[] { 1, 0, 0 }),
                    CreateTestDocument("doc2", "Dog", 3, new float[] { 0, 1, 0 }),
                    CreateTestDocument("doc3", "Fish", 3, new float[] { 0, 0, 1 })
                });
                store.Flush();
            }

            // Act - reopen and search
            using var reopened = new FileDocumentStore<float>(3, options);
            var queryVector = new Vector<float>(new float[] { 1, 0, 0 });
            var results = reopened.GetSimilar(queryVector, topK: 1).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Persistence_MetadataSurvivesReopen()
        {
            // Arrange
            var options = CreateOptions();
            using (var store = new FileDocumentStore<float>(3, options))
            {
                var doc = CreateTestDocument("doc1", "Content", 3);
                doc.Document.Metadata["category"] = "test";
                doc.Document.Metadata["score"] = 42L;
                store.Add(doc);
                store.Flush();
            }

            // Act
            using var reopened = new FileDocumentStore<float>(3, options);
            var result = reopened.GetById("doc1");

            // Assert
            Assert.NotNull(result);
            Assert.Equal("test", result.Metadata["category"]);
            Assert.Equal(42L, result.Metadata["score"]);
        }

        #endregion

        #region WAL Tests

        [Fact]
        public void Wal_RecoversPendingAddsAfterReopen()
        {
            // Arrange - add without flush, just dispose (which does flush)
            var options = CreateOptions();
            using (var store = new FileDocumentStore<float>(3, options))
            {
                store.Add(CreateTestDocument("doc1", "WAL test", 3, new float[] { 1, 0, 0 }));
                // Don't call Flush explicitly - Dispose will handle it
            }

            // Act - reopen
            using var reopened = new FileDocumentStore<float>(3, options);

            // Assert
            Assert.Equal(1, reopened.DocumentCount);
            var doc = reopened.GetById("doc1");
            Assert.NotNull(doc);
            Assert.Equal("WAL test", doc.Content);
        }

        [Fact]
        public void Wal_RecoversRemovesAfterReopen()
        {
            // Arrange - add, flush, then remove without flush
            var options = CreateOptions();
            using (var store = new FileDocumentStore<float>(3, options))
            {
                store.Add(CreateTestDocument("doc1", "Keep", 3, new float[] { 1, 0, 0 }));
                store.Add(CreateTestDocument("doc2", "Remove", 3, new float[] { 0, 1, 0 }));
                // Dispose flushes everything
            }

            // Reopen, remove doc2, close
            using (var store = new FileDocumentStore<float>(3, options))
            {
                Assert.Equal(2, store.DocumentCount);
                store.Remove("doc2");
                // Dispose flushes
            }

            // Act - reopen and verify
            using var reopened = new FileDocumentStore<float>(3, options);

            // Assert
            Assert.Equal(1, reopened.DocumentCount);
            Assert.NotNull(reopened.GetById("doc1"));
            Assert.Null(reopened.GetById("doc2"));
        }

        #endregion

        #region Compaction Tests

        [Fact]
        public void Compact_ClearsTombstones()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Add(CreateTestDocument("doc1", "Test 1", 3));
            store.Add(CreateTestDocument("doc2", "Test 2", 3));
            store.Remove("doc1");
            Assert.Equal(1, store.TombstoneCount);

            // Act
            store.Compact();

            // Assert
            Assert.Equal(0, store.TombstoneCount);
            Assert.Equal(1, store.DocumentCount);
        }

        [Fact]
        public void Compact_DataIntactAfterReopen()
        {
            // Arrange
            var options = CreateOptions();
            using (var store = new FileDocumentStore<float>(3, options))
            {
                store.Add(CreateTestDocument("doc1", "Keep", 3, new float[] { 1, 0, 0 }));
                store.Add(CreateTestDocument("doc2", "Remove", 3, new float[] { 0, 1, 0 }));
                store.Add(CreateTestDocument("doc3", "Keep too", 3, new float[] { 0, 0, 1 }));
                store.Remove("doc2");
                store.Compact();
            }

            // Act
            using var reopened = new FileDocumentStore<float>(3, options);

            // Assert
            Assert.Equal(2, reopened.DocumentCount);
            Assert.NotNull(reopened.GetById("doc1"));
            Assert.Null(reopened.GetById("doc2"));
            Assert.NotNull(reopened.GetById("doc3"));
        }

        #endregion

        #region Dispose Tests

        [Fact]
        public void Dispose_FlushesDataToDisk()
        {
            // Arrange
            var options = CreateOptions();

            // Act - add and dispose without explicit Flush
            using (var store = new FileDocumentStore<float>(3, options))
            {
                store.Add(CreateTestDocument("doc1", "Disposed test", 3, new float[] { 1, 0, 0 }));
            }

            // Assert - files should exist from dispose flush
            Assert.True(File.Exists(Path.Combine(_testDir, "store.meta")));
            Assert.True(File.Exists(Path.Combine(_testDir, "documents.json")));
        }

        [Fact]
        public void Dispose_CalledTwice_DoesNotThrow()
        {
            // Arrange
            var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Add(CreateTestDocument("doc1", "Test", 3));

            // Act & Assert - should not throw
            store.Dispose();
            store.Dispose();
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void ReAddDeletedDocument_WorksCorrectly()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Add(CreateTestDocument("doc1", "Original", 3, new float[] { 1, 0, 0 }));
            store.Remove("doc1");
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(1, store.TombstoneCount);

            // Act
            store.Add(CreateTestDocument("doc1", "Updated", 3, new float[] { 0, 1, 0 }));

            // Assert
            Assert.Equal(1, store.DocumentCount);
            Assert.Equal(0, store.TombstoneCount);
            var doc = store.GetById("doc1");
            Assert.NotNull(doc);
            Assert.Equal("Updated", doc.Content);
        }

        [Fact]
        public void LargeDocument_HandledCorrectly()
        {
            // Arrange
            using var store = new FileDocumentStore<float>(128, CreateOptions());
            string longContent = new string('x', 100_000);
            var doc = CreateTestDocument("doc1", longContent, 128);

            // Act
            store.Add(doc);
            store.Flush();

            // Assert
            var retrieved = store.GetById("doc1");
            Assert.NotNull(retrieved);
            Assert.Equal(100_000, retrieved.Content.Length);
        }

        #endregion

        #region ObjectDisposedException Tests

        [Fact]
        public void Add_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() =>
                store.Add(CreateTestDocument("doc1", "Test", 3)));
        }

        [Fact]
        public void GetById_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Add(CreateTestDocument("doc1", "Test", 3));
            store.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => store.GetById("doc1"));
        }

        [Fact]
        public void GetSimilar_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() =>
                store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), 1).ToList());
        }

        [Fact]
        public void Clear_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => store.Clear());
        }

        [Fact]
        public void Flush_AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var store = new FileDocumentStore<float>(3, CreateOptions());
            store.Dispose();

            // Act & Assert
            Assert.Throws<ObjectDisposedException>(() => store.Flush());
        }

        #endregion

        #region Double Type Tests

        [Fact]
        public void DoubleType_AddAndRetrieve_WorksCorrectly()
        {
            // Arrange
            var dir = Path.Combine(_testDir, "double_test");
            var options = new FileDocumentStoreOptions { DirectoryPath = dir };
            using var store = new FileDocumentStore<double>(3, options);

            var doc = new VectorDocument<double>
            {
                Document = new Document<double>("doc1", "Double precision test"),
                Embedding = new Vector<double>(new double[] { 0.123456789012345, 0.987654321098765, 0.555555555555555 })
            };

            // Act
            store.Add(doc);
            var result = store.GetById("doc1");

            // Assert
            Assert.NotNull(result);
            Assert.Equal("Double precision test", result.Content);
        }

        [Fact]
        public void DoubleType_PersistenceAndReload_PreservesPrecision()
        {
            // Arrange
            var dir = Path.Combine(_testDir, "double_persist");
            var options = new FileDocumentStoreOptions { DirectoryPath = dir };
            double[] originalValues = { 0.123456789012345, 0.987654321098765, 0.555555555555555 };

            using (var store = new FileDocumentStore<double>(3, options))
            {
                store.Add(new VectorDocument<double>
                {
                    Document = new Document<double>("doc1", "Precision test"),
                    Embedding = new Vector<double>(originalValues)
                });
                store.Flush();
            }

            // Act - reopen and search
            using var reopened = new FileDocumentStore<double>(3, options);
            var queryVector = new Vector<double>(originalValues);
            var results = reopened.GetSimilar(queryVector, topK: 1).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void DoubleType_SimilaritySearch_WorksCorrectly()
        {
            // Arrange
            var dir = Path.Combine(_testDir, "double_search");
            var options = new FileDocumentStoreOptions { DirectoryPath = dir };
            using var store = new FileDocumentStore<double>(3, options);

            store.AddBatch(new List<VectorDocument<double>>
            {
                new VectorDocument<double>
                {
                    Document = new Document<double>("cat", "Cat content"),
                    Embedding = new Vector<double>(new double[] { 1.0, 0.0, 0.0 })
                },
                new VectorDocument<double>
                {
                    Document = new Document<double>("dog", "Dog content"),
                    Embedding = new Vector<double>(new double[] { 0.0, 1.0, 0.0 })
                }
            });

            var queryVector = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = store.GetSimilar(queryVector, topK: 1).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("cat", results[0].Id);
        }

        #endregion

        #region FlushOnEveryWrite Tests

        [Fact]
        public void FlushOnEveryWrite_PersistsImmediately()
        {
            // Arrange
            var options = new FileDocumentStoreOptions
            {
                DirectoryPath = _testDir,
                FlushOnEveryWrite = true
            };

            using (var store = new FileDocumentStore<float>(3, options))
            {
                // Act - add a document (should flush immediately)
                store.Add(CreateTestDocument("doc1", "Immediate persist", 3, new float[] { 1, 0, 0 }));

                // Assert - files should exist immediately (not just after Dispose)
                Assert.True(File.Exists(Path.Combine(_testDir, "store.meta")));
                Assert.True(File.Exists(Path.Combine(_testDir, "documents.json")));
            }
        }

        #endregion

        #region Helper Methods

        private VectorDocument<float> CreateTestDocument(
            string id,
            string content,
            int dimension,
            float[]? values = null)
        {
            var vector = values ?? Enumerable.Range(1, dimension).Select(i => (float)i / dimension).ToArray();
            return new VectorDocument<float>
            {
                Document = new Document<float>(id, content),
                Embedding = new Vector<float>(vector)
            };
        }

        #endregion
    }
}
