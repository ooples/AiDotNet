using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for ParentDocumentRetriever which retrieves precise chunks but returns full parent documents.
    /// </summary>
    public class ParentDocumentRetrieverTests
    {
        #region Test Helpers

        /// <summary>
        /// Mock embedding model for testing.
        /// </summary>
        private class MockEmbeddingModel : IEmbeddingModel<double>
        {
            public int EmbeddingDimension => 128;
            public int MaxTokens => 512;

            public Vector<double> Embed(string text)
            {
                var embedding = new double[EmbeddingDimension];
                if (!string.IsNullOrEmpty(text))
                {
                    var charSum = text.Sum(c => (double)c);
                    for (int i = 0; i < EmbeddingDimension; i++)
                    {
                        embedding[i] = Math.Sin(charSum + i) * 0.5 + 0.5;
                    }
                }
                return new Vector<double>(embedding);
            }

            public Task<Vector<double>> EmbedAsync(string text)
            {
                return Task.FromResult(Embed(text));
            }

            public Matrix<double> EmbedBatch(IEnumerable<string> texts)
            {
                var textList = texts.ToList();
                var matrix = new Matrix<double>(textList.Count, EmbeddingDimension);
                for (int i = 0; i < textList.Count; i++)
                {
                    var embedding = Embed(textList[i]);
                    for (int j = 0; j < EmbeddingDimension; j++)
                    {
                        matrix[i, j] = embedding[j];
                    }
                }
                return matrix;
            }

            public Task<Matrix<double>> EmbedBatchAsync(IEnumerable<string> texts)
            {
                return Task.FromResult(EmbedBatch(texts));
            }
        }

        /// <summary>
        /// Mock document store for testing parent document retrieval.
        /// </summary>
        private class MockDocumentStore : IDocumentStore<double>
        {
            private readonly List<Document<double>> _documents = new();
            private readonly Dictionary<string, Document<double>> _parentDocuments = new();

            public int DocumentCount => _documents.Count;
            public int VectorDimension => 128;

            public void Add(VectorDocument<double> vectorDocument)
            {
                _documents.Add(vectorDocument.Document);
            }

            public void AddBatch(IEnumerable<VectorDocument<double>> vectorDocuments)
            {
                foreach (var vd in vectorDocuments)
                {
                    _documents.Add(vd.Document);
                }
            }

            public void AddDocument(Document<double> document)
            {
                _documents.Add(document);
            }

            public void AddParentDocument(Document<double> parent)
            {
                _parentDocuments[parent.Id] = parent;
            }

            public IEnumerable<Document<double>> GetAll() => _documents;

            public IEnumerable<Document<double>> GetSimilar(Vector<double> queryVector, int topK)
                => _documents
                    .OrderByDescending(d => d.RelevanceScore)
                    .Take(topK);

            public IEnumerable<Document<double>> GetSimilarWithFilters(
                Vector<double> queryVector, int topK, Dictionary<string, object> metadataFilters)
            {
                var filtered = _documents.Where(doc => MatchesFilters(doc, metadataFilters));
                return filtered
                    .OrderByDescending(d => d.RelevanceScore)
                    .Take(topK);
            }

            private bool MatchesFilters(Document<double> doc, Dictionary<string, object> filters)
            {
                foreach (var filter in filters)
                {
                    if (!doc.Metadata.TryGetValue(filter.Key, out var value) ||
                        !Equals(value, filter.Value))
                    {
                        return false;
                    }
                }
                return true;
            }

            public Document<double>? GetById(string documentId)
            {
                // First check parent documents, then regular documents
                if (_parentDocuments.TryGetValue(documentId, out var parent))
                    return parent;
                return _documents.FirstOrDefault(d => d.Id == documentId);
            }

            public bool Remove(string documentId)
            {
                var doc = _documents.FirstOrDefault(d => d.Id == documentId);
                if (doc != null)
                {
                    _documents.Remove(doc);
                    return true;
                }
                return false;
            }

            public void Clear()
            {
                _documents.Clear();
                _parentDocuments.Clear();
            }
        }

        /// <summary>
        /// Creates a store with chunks that reference parent documents.
        /// </summary>
        private (MockDocumentStore store, MockEmbeddingModel model) CreateStoreWithChunks(
            params (string chunkId, string content, double score, string parentId)[] chunks)
        {
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            foreach (var (chunkId, content, score, parentId) in chunks)
            {
                var doc = new Document<double>(chunkId, content,
                    new Dictionary<string, object> { { "parent_id", parentId } });
                doc.RelevanceScore = score;
                doc.HasRelevanceScore = true;
                store.AddDocument(doc);
            }

            return (store, model);
        }

        /// <summary>
        /// Creates a store with chunks and their parent documents.
        /// </summary>
        private (MockDocumentStore store, MockEmbeddingModel model) CreateStoreWithChunksAndParents(
            (string chunkId, string content, double score, string parentId)[] chunks,
            (string parentId, string content)[] parents)
        {
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            foreach (var (chunkId, content, score, parentId) in chunks)
            {
                var doc = new Document<double>(chunkId, content,
                    new Dictionary<string, object>
                    {
                        { "parent_id", parentId },
                        { "chunk_index", 0 },
                        { "chunk_start", 0 }
                    });
                doc.RelevanceScore = score;
                doc.HasRelevanceScore = true;
                store.AddDocument(doc);
            }

            foreach (var (parentId, content) in parents)
            {
                var parent = new Document<double>(parentId, content);
                store.AddParentDocument(parent);
            }

            return (store, model);
        }

        #endregion

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            // Act
            var retriever = new ParentDocumentRetriever<double>(
                store, model, chunkSize: 256, parentSize: 2048, includeNeighboringChunks: true);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_NullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ParentDocumentRetriever<double>(null!, model, 256, 2048, true));
        }

        [Fact]
        public void Constructor_NullEmbeddingModel_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ParentDocumentRetriever<double>(store, null!, 256, 2048, true));
        }

        [Fact]
        public void Constructor_ZeroChunkSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ParentDocumentRetriever<double>(store, model, chunkSize: 0, parentSize: 2048, includeNeighboringChunks: true));
        }

        [Fact]
        public void Constructor_NegativeChunkSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ParentDocumentRetriever<double>(store, model, chunkSize: -1, parentSize: 2048, includeNeighboringChunks: true));
        }

        [Fact]
        public void Constructor_ZeroParentSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ParentDocumentRetriever<double>(store, model, chunkSize: 256, parentSize: 0, includeNeighboringChunks: true));
        }

        [Fact]
        public void Constructor_NegativeParentSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ParentDocumentRetriever<double>(store, model, chunkSize: 256, parentSize: -1, includeNeighboringChunks: true));
        }

        [Fact]
        public void Constructor_ParentSizeLessThanChunkSize_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new ParentDocumentRetriever<double>(store, model, chunkSize: 512, parentSize: 256, includeNeighboringChunks: true));
        }

        [Fact]
        public void Constructor_ParentSizeEqualsChunkSize_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            // Act
            var retriever = new ParentDocumentRetriever<double>(
                store, model, chunkSize: 256, parentSize: 256, includeNeighboringChunks: false);

            // Assert
            Assert.NotNull(retriever);
        }

        #endregion

        #region Retrieve Method Tests

        [Fact]
        public void Retrieve_EmptyStore_ReturnsEmptyResults()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, true);

            // Act
            var results = retriever.Retrieve("test query").ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_SingleChunk_ReturnsParentDocument()
        {
            // Arrange
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "small chunk content", 0.9, "parent1"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
        }

        [Fact]
        public void Retrieve_MultipleChunksSameParent_ReturnsSingleParent()
        {
            // Arrange
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "first chunk", 0.9, "parent1"),
                ("chunk2", "second chunk", 0.8, "parent1"),
                ("chunk3", "third chunk", 0.7, "parent1"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("chunk").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
        }

        [Fact]
        public void Retrieve_ChunksFromDifferentParents_ReturnsMultipleParents()
        {
            // Arrange
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "chunk from parent 1", 0.9, "parent1"),
                ("chunk2", "chunk from parent 2", 0.8, "parent2"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("chunk").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            Assert.Contains(results, r => r.Id == "parent1");
            Assert.Contains(results, r => r.Id == "parent2");
        }

        [Fact]
        public void Retrieve_WithTopK_ReturnsLimitedResults()
        {
            // Arrange
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "content 1", 0.9, "parent1"),
                ("chunk2", "content 2", 0.8, "parent2"),
                ("chunk3", "content 3", 0.7, "parent3"),
                ("chunk4", "content 4", 0.6, "parent4"),
                ("chunk5", "content 5", 0.5, "parent5"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content", topK: 3).ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_NullQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        #endregion

        #region Parent Document Retrieval Tests

        [Fact]
        public void Retrieve_ParentDocumentExists_ReturnsFullParentContent()
        {
            // Arrange
            var chunks = new[]
            {
                ("chunk1", "small chunk", 0.9, "parent1")
            };
            var parents = new[]
            {
                ("parent1", "This is the full parent document content with much more text than the chunk")
            };
            var (store, model) = CreateStoreWithChunksAndParents(chunks, parents);
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
            Assert.Contains("full parent document content", results[0].Content);
        }

        [Fact]
        public void Retrieve_ParentNotInStore_UsesFallbackWithChunkContent()
        {
            // Arrange - No parent documents, just chunks
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "chunk content only", 0.9, "missing_parent"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("missing_parent", results[0].Id);
            // Content should be from the chunk as fallback
            Assert.Equal("chunk content only", results[0].Content);
        }

        [Fact]
        public void Retrieve_ChunkWithoutParentId_IsSkipped()
        {
            // Arrange - Chunk without parent_id metadata
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            var chunkWithParent = new Document<double>("chunk1", "has parent",
                new Dictionary<string, object> { { "parent_id", "parent1" } });
            chunkWithParent.RelevanceScore = 0.9;
            chunkWithParent.HasRelevanceScore = true;
            store.AddDocument(chunkWithParent);

            var chunkWithoutParent = new Document<double>("chunk2", "no parent",
                new Dictionary<string, object>()); // No parent_id
            chunkWithoutParent.RelevanceScore = 0.95;
            chunkWithoutParent.HasRelevanceScore = true;
            store.AddDocument(chunkWithoutParent);

            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
        }

        #endregion

        #region Score Aggregation Tests

        [Fact]
        public void Retrieve_ParentScore_UsesMaxChunkScore()
        {
            // Arrange - Multiple chunks from same parent with different scores
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "low score chunk", 0.5, "parent1"),
                ("chunk2", "high score chunk", 0.9, "parent1"),
                ("chunk3", "medium score chunk", 0.7, "parent1"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("chunk").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
            Assert.Equal(0.9, results[0].RelevanceScore, 3); // Max of 0.5, 0.9, 0.7
        }

        [Fact]
        public void Retrieve_ResultsOrderedByBestChunkScore()
        {
            // Arrange
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "parent 1 chunk", 0.6, "parent1"),
                ("chunk2", "parent 2 chunk", 0.9, "parent2"),
                ("chunk3", "parent 3 chunk", 0.7, "parent3"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("chunk").ToList();

            // Assert
            Assert.Equal(3, results.Count);
            for (int i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].RelevanceScore >= results[i].RelevanceScore,
                    "Results should be ordered by descending score");
            }
        }

        #endregion

        #region Include Neighboring Chunks Tests

        [Fact]
        public void Retrieve_IncludeNeighboringChunks_ConcatenatesContent()
        {
            // Arrange - Multiple chunks from same parent
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "First chunk content.", 0.9, "parent1"),
                ("chunk2", "Second chunk content.", 0.8, "parent1"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, includeNeighboringChunks: true);

            // Act
            var results = retriever.Retrieve("chunk").ToList();

            // Assert
            Assert.Single(results);
            // Content should contain both chunks
            Assert.Contains("First chunk content", results[0].Content);
            Assert.Contains("Second chunk content", results[0].Content);
        }

        [Fact]
        public void Retrieve_NoIncludeNeighboringChunks_UsesParentDocument()
        {
            // Arrange
            var chunks = new[]
            {
                ("chunk1", "First chunk.", 0.9, "parent1"),
                ("chunk2", "Second chunk.", 0.8, "parent1")
            };
            var parents = new[]
            {
                ("parent1", "Full parent content")
            };
            var (store, model) = CreateStoreWithChunksAndParents(chunks, parents);
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, includeNeighboringChunks: false);

            // Act
            var results = retriever.Retrieve("chunk").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("Full parent content", results[0].Content);
        }

        #endregion

        #region Metadata Tests

        [Fact]
        public void Retrieve_ChunkSpecificMetadataRemoved()
        {
            // Arrange - Chunk with specific metadata that should be removed from parent
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            var chunk = new Document<double>("chunk1", "chunk content",
                new Dictionary<string, object>
                {
                    { "parent_id", "parent1" },
                    { "chunk_index", 0 },
                    { "chunk_start", 100 },
                    { "chunk_end", 200 },
                    { "category", "science" } // This should be kept
                });
            chunk.RelevanceScore = 0.9;
            chunk.HasRelevanceScore = true;
            store.AddDocument(chunk);

            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            // Chunk-specific metadata should be removed
            Assert.False(results[0].Metadata.ContainsKey("parent_id"));
            Assert.False(results[0].Metadata.ContainsKey("chunk_index"));
            Assert.False(results[0].Metadata.ContainsKey("chunk_start"));
            Assert.False(results[0].Metadata.ContainsKey("chunk_end"));
            // General metadata should be preserved
            Assert.True(results[0].Metadata.ContainsKey("category"));
        }

        [Fact]
        public void Retrieve_WithMetadataFilter_FiltersChunks()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            var chunk1 = new Document<double>("chunk1", "science content",
                new Dictionary<string, object>
                {
                    { "parent_id", "parent1" },
                    { "category", "science" }
                });
            chunk1.RelevanceScore = 0.9;
            chunk1.HasRelevanceScore = true;
            store.AddDocument(chunk1);

            var chunk2 = new Document<double>("chunk2", "history content",
                new Dictionary<string, object>
                {
                    { "parent_id", "parent2" },
                    { "category", "history" }
                });
            chunk2.RelevanceScore = 0.85;
            chunk2.HasRelevanceScore = true;
            store.AddDocument(chunk2);

            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content", topK: 5,
                new Dictionary<string, object> { { "category", "science" } }).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_ZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        [Fact]
        public void Retrieve_NegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: -1).ToList());
        }

        [Fact]
        public void Retrieve_TopKGreaterThanParents_ReturnsAllAvailable()
        {
            // Arrange
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "content 1", 0.9, "parent1"),
                ("chunk2", "content 2", 0.8, "parent2"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content", topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_DocumentsHaveRelevanceScores()
        {
            // Arrange
            var (store, model) = CreateStoreWithChunks(
                ("chunk1", "content", 0.9, "parent1"));
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.All(results, r => Assert.True(r.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_ChunkWithNullParentId_IsSkipped()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            var chunkWithNullParent = new Document<double>("chunk1", "null parent",
                new Dictionary<string, object> { { "parent_id", null! } });
            chunkWithNullParent.RelevanceScore = 0.9;
            chunkWithNullParent.HasRelevanceScore = true;
            store.AddDocument(chunkWithNullParent);

            var chunkWithValidParent = new Document<double>("chunk2", "valid parent",
                new Dictionary<string, object> { { "parent_id", "parent1" } });
            chunkWithValidParent.RelevanceScore = 0.8;
            chunkWithValidParent.HasRelevanceScore = true;
            store.AddDocument(chunkWithValidParent);

            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
        }

        [Fact]
        public void Retrieve_ChunkWithEmptyParentId_IsSkipped()
        {
            // Arrange
            var store = new MockDocumentStore();
            var model = new MockEmbeddingModel();

            var chunkWithEmptyParent = new Document<double>("chunk1", "empty parent",
                new Dictionary<string, object> { { "parent_id", "" } });
            chunkWithEmptyParent.RelevanceScore = 0.9;
            chunkWithEmptyParent.HasRelevanceScore = true;
            store.AddDocument(chunkWithEmptyParent);

            var chunkWithValidParent = new Document<double>("chunk2", "valid parent",
                new Dictionary<string, object> { { "parent_id", "parent1" } });
            chunkWithValidParent.RelevanceScore = 0.8;
            chunkWithValidParent.HasRelevanceScore = true;
            store.AddDocument(chunkWithValidParent);

            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("parent1", results[0].Id);
        }

        [Fact]
        public void Retrieve_ManyChunksManyParents_HandlesCorrectly()
        {
            // Arrange
            var chunks = Enumerable.Range(1, 30)
                .Select(i => ($"chunk{i}", $"content for chunk {i}", 0.9 - (i * 0.02), $"parent{(i % 10) + 1}"))
                .ToArray();
            var (store, model) = CreateStoreWithChunks(chunks);
            var retriever = new ParentDocumentRetriever<double>(store, model, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("content", topK: 5).ToList();

            // Assert
            Assert.True(results.Count <= 5);
            // Should return unique parents
            Assert.Equal(results.Count, results.Select(r => r.Id).Distinct().Count());
        }

        #endregion
    }
}
