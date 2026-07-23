using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Filtering;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    public class InMemoryDocumentStoreTests
    {
        #region Constructor Tests

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidDimension_CreatesStore()
        {
            // Arrange & Act
            var store = new InMemoryDocumentStore<float>(vectorDimension: 384);

            // Assert
            Assert.Equal(0, store.DocumentCount);
            Assert.Equal(384, store.VectorDimension);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroDimension_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new InMemoryDocumentStore<float>(vectorDimension: 0));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeDimension_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new InMemoryDocumentStore<float>(vectorDimension: -1));
        }

        #endregion

        #region Add Tests

        [Fact(Timeout = 60000)]
        public async Task Add_WithValidDocument_IncreasesCount()
        {
            // Arrange
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            var doc = CreateTestDocument("doc1", "Test content", 3);

            // Act
            store.Add(doc);

            // Assert
            Assert.Equal(1, store.DocumentCount);
        }

        [Fact(Timeout = 60000)]
        public async Task Add_WithMismatchedDimension_ThrowsArgumentException()
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

        [Fact(Timeout = 60000)]
        public async Task AddBatch_WithValidDocuments_IncreasesCount()
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

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_WithMatchingDocuments_ReturnsTopK()
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

        [Fact(Timeout = 60000)]
        public async Task GetSimilar_WithEmptyStore_ReturnsEmpty()
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

        [Fact(Timeout = 60000)]
        public async Task GetById_WithExistingDocument_ReturnsDocument()
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

        [Fact(Timeout = 60000)]
        public async Task GetById_WithNonExistingDocument_ReturnsNull()
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

        [Fact(Timeout = 60000)]
        public async Task Remove_WithExistingDocument_ReturnsTrue()
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

        [Fact(Timeout = 60000)]
        public async Task Remove_WithNonExistingDocument_ReturnsFalse()
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

        [Fact(Timeout = 60000)]
        public async Task Clear_RemovesAllDocuments()
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

        [Fact(Timeout = 60000)]
        public async Task GetAll_WithDocuments_ReturnsAllDocuments()
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

        [Fact(Timeout = 60000)]
        public async Task GetAll_WithEmptyStore_ReturnsEmpty()
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

        [Fact(Timeout = 60000)]
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

        #region Rich MetadataFilter (end-to-end)

        private static VectorDocument<float> RichDoc(string id, float[] vector, Dictionary<string, object> metadata)
            => new VectorDocument<float>
            {
                Document = new Document<float>(id, id + "-content", metadata),
                Embedding = new Vector<float>(vector)
            };

        private static InMemoryDocumentStore<float> BuildRichStore()
        {
            var store = new InMemoryDocumentStore<float>(vectorDimension: 3);
            store.AddBatch(new List<VectorDocument<float>>
            {
                RichDoc("d1", new float[] { 1, 0, 0 }, new Dictionary<string, object> { ["category"] = "science", ["year"] = 2022, ["archived"] = false }),
                RichDoc("d2", new float[] { 0.9f, 0.1f, 0 }, new Dictionary<string, object> { ["category"] = "science", ["year"] = 2015, ["author"] = "A", ["archived"] = false }),
                RichDoc("d3", new float[] { 0.8f, 0.2f, 0 }, new Dictionary<string, object> { ["category"] = "science", ["year"] = 2023, ["archived"] = true }),
                RichDoc("d4", new float[] { 0.7f, 0.3f, 0 }, new Dictionary<string, object> { ["category"] = "food", ["year"] = 2024, ["archived"] = false }),
                RichDoc("d5", new float[] { 0.6f, 0.4f, 0 }, new Dictionary<string, object> { ["category"] = "science", ["year"] = 2010, ["author"] = "Z", ["archived"] = false }),
            });
            return store;
        }

        [Fact]
        public void GetSimilarWithFilter_And_Range_ReturnsOnlyMatching()
        {
            var store = BuildRichStore();
            // category == science AND year >= 2020 AND NOT archived
            var filter = MetadataFilter.Eq("category", "science")
                .And(MetadataFilter.Gte("year", 2020))
                .And(MetadataFilter.Eq("archived", true).Not());

            var results = store.GetSimilarWithFilter(new Vector<float>(new float[] { 1, 0, 0 }), filter, topK: 10).ToList();

            var ids = results.Select(r => r.Id).OrderBy(x => x).ToList();
            Assert.Equal(new[] { "d1" }, ids);
        }

        [Fact]
        public void GetSimilarWithFilter_OrAndIn_ReturnsUnion()
        {
            var store = BuildRichStore();
            // category == science AND (year >= 2020 OR author in [A])
            var filter = MetadataFilter.Eq("category", "science")
                .And(MetadataFilter.Gte("year", 2020).Or(MetadataFilter.In("author", new object[] { "A" })));

            var results = store.GetSimilarWithFilter(new Vector<float>(new float[] { 1, 0, 0 }), filter, topK: 10).ToList();

            var ids = results.Select(r => r.Id).OrderBy(x => x).ToList();
            // d1 (2022), d2 (author A), d3 (2023) - all science; d4 is food, d5 is 2010/author Z.
            Assert.Equal(new[] { "d1", "d2", "d3" }, ids);
        }

        [Fact]
        public void GetSimilarWithFilter_Exists_And_TopKHonoured()
        {
            var store = BuildRichStore();
            var filter = MetadataFilter.Exists("author");

            var results = store.GetSimilarWithFilter(new Vector<float>(new float[] { 1, 0, 0 }), filter, topK: 1).ToList();

            Assert.Single(results);
            Assert.Contains(results[0].Id, new[] { "d2", "d5" });
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact(Timeout = 60000)]
        public async Task GetSimilarWithFilterAsync_MatchesSyncResult()
        {
            var store = BuildRichStore();
            var filter = MetadataFilter.Ne("category", "science");

            var results = (await store.GetSimilarWithFilterAsync(new Vector<float>(new float[] { 1, 0, 0 }), filter, topK: 10)).ToList();

            Assert.Equal(new[] { "d4" }, results.Select(r => r.Id).ToList());
        }

        [Fact]
        public void GetSimilarWithFilter_NullFilter_BehavesLikeGetSimilar()
        {
            var store = BuildRichStore();
            var results = store.GetSimilarWithFilter(new Vector<float>(new float[] { 1, 0, 0 }), null!, topK: 3).ToList();
            Assert.Equal(3, results.Count);
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
