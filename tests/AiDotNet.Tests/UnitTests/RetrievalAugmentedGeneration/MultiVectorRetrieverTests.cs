using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for MultiVectorRetriever which uses multiple embedding vectors per document.
    /// </summary>
    public class MultiVectorRetrieverTests
    {
        #region Test Helpers

        /// <summary>
        /// Mock document store for testing MultiVectorRetriever.
        /// Returns documents with vector IDs in format "docId_vector_N".
        /// </summary>
        private class MockDocumentStore : IDocumentStore<double>
        {
            private readonly List<Document<double>> _documents = new();

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
                => _documents.FirstOrDefault(d => d.Id == documentId);

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

            public void Clear() => _documents.Clear();
        }

        /// <summary>
        /// Creates a store with documents that have multiple vector representations.
        /// Each document gets vectorsPerDocument entries with IDs like "docId_vector_0", "docId_vector_1", etc.
        /// </summary>
        private MockDocumentStore CreateMultiVectorStore(
            int vectorsPerDocument,
            params (string baseId, string content, double[] vectorScores)[] docs)
        {
            var store = new MockDocumentStore();

            foreach (var (baseId, content, vectorScores) in docs)
            {
                for (int i = 0; i < vectorsPerDocument && i < vectorScores.Length; i++)
                {
                    var vectorDocId = $"{baseId}_vector_{i}";
                    var doc = new Document<double>(vectorDocId, content);
                    doc.RelevanceScore = vectorScores[i];
                    doc.HasRelevanceScore = true;
                    store.AddDocument(doc);
                }
            }

            return store;
        }

        /// <summary>
        /// Creates a store with documents that have multiple vectors and metadata.
        /// </summary>
        private MockDocumentStore CreateMultiVectorStoreWithMetadata(
            int vectorsPerDocument,
            params (string baseId, string content, double[] vectorScores, Dictionary<string, object> metadata)[] docs)
        {
            var store = new MockDocumentStore();

            foreach (var (baseId, content, vectorScores, metadata) in docs)
            {
                for (int i = 0; i < vectorsPerDocument && i < vectorScores.Length; i++)
                {
                    var vectorDocId = $"{baseId}_vector_{i}";
                    var doc = new Document<double>(vectorDocId, content, metadata);
                    doc.RelevanceScore = vectorScores[i];
                    doc.HasRelevanceScore = true;
                    store.AddDocument(doc);
                }
            }

            return store;
        }

        #endregion

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act
            var retriever = new MultiVectorRetriever<double>(store, vectorsPerDocument: 3, aggregationMethod: "max");

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_NullDocumentStore_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiVectorRetriever<double>(null!, vectorsPerDocument: 3, aggregationMethod: "max"));
        }

        [Fact]
        public void Constructor_ZeroVectorsPerDocument_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MultiVectorRetriever<double>(store, vectorsPerDocument: 0, aggregationMethod: "max"));
        }

        [Fact]
        public void Constructor_NegativeVectorsPerDocument_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MultiVectorRetriever<double>(store, vectorsPerDocument: -1, aggregationMethod: "max"));
        }

        [Fact]
        public void Constructor_NullAggregationMethod_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiVectorRetriever<double>(store, vectorsPerDocument: 3, aggregationMethod: null!));
        }

        [Fact]
        public void Constructor_DifferentAggregationMethods_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert - All valid aggregation methods should work
            Assert.NotNull(new MultiVectorRetriever<double>(store, 3, "max"));
            Assert.NotNull(new MultiVectorRetriever<double>(store, 3, "mean"));
            Assert.NotNull(new MultiVectorRetriever<double>(store, 3, "average"));
            Assert.NotNull(new MultiVectorRetriever<double>(store, 3, "weighted"));
        }

        #endregion

        #region Retrieve Method Tests

        [Fact]
        public void Retrieve_EmptyStore_ReturnsEmptyResults()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("test query").ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_SingleDocumentMultipleVectors_ReturnsSingleAggregatedResult()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "test content", new[] { 0.9, 0.7, 0.5 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            // The document ID should be extracted from "doc1_vector_N" format
            Assert.StartsWith("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_MultipleDocuments_ReturnsMultipleAggregatedResults()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "first document", new[] { 0.9, 0.8, 0.7 }),
                ("doc2", "second document", new[] { 0.6, 0.5, 0.4 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("document").ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_WithTopK_ReturnsLimitedResults()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.9, 0.8, 0.7 }),
                ("doc2", "content 2", new[] { 0.85, 0.75, 0.65 }),
                ("doc3", "content 3", new[] { 0.8, 0.7, 0.6 }),
                ("doc4", "content 4", new[] { 0.75, 0.65, 0.55 }),
                ("doc5", "content 5", new[] { 0.7, 0.6, 0.5 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

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
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        #endregion

        #region Aggregation Method Tests - Max

        [Fact]
        public void Retrieve_MaxAggregation_UsesHighestScore()
        {
            // Arrange - Document with vectors scoring 0.9, 0.5, 0.3
            var store = CreateMultiVectorStore(3,
                ("doc1", "test content", new[] { 0.9, 0.5, 0.3 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal(0.9, results[0].RelevanceScore, 3);
        }

        [Fact]
        public void Retrieve_MaxAggregation_RanksDocumentsByMaxScore()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.5, 0.9, 0.3 }),  // Max = 0.9
                ("doc2", "content 2", new[] { 0.95, 0.4, 0.3 })); // Max = 0.95
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            Assert.True(results[0].RelevanceScore >= results[1].RelevanceScore);
        }

        #endregion

        #region Aggregation Method Tests - Mean

        [Fact]
        public void Retrieve_MeanAggregation_UsesAverageScore()
        {
            // Arrange - Document with vectors scoring 0.9, 0.6, 0.3 -> mean = 0.6
            var store = CreateMultiVectorStore(3,
                ("doc1", "test content", new[] { 0.9, 0.6, 0.3 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "mean");

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal(0.6, results[0].RelevanceScore, 3);
        }

        [Fact]
        public void Retrieve_AverageAggregation_SameAsMean()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "test content", new[] { 0.9, 0.6, 0.3 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "average");

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal(0.6, results[0].RelevanceScore, 3);
        }

        [Fact]
        public void Retrieve_MeanAggregation_RanksDocumentsByMeanScore()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.9, 0.9, 0.3 }),  // Mean = 0.7
                ("doc2", "content 2", new[] { 0.8, 0.8, 0.8 })); // Mean = 0.8
            var retriever = new MultiVectorRetriever<double>(store, 3, "mean");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Doc2 should rank higher due to higher mean score
            Assert.True(results[0].RelevanceScore >= results[1].RelevanceScore);
        }

        #endregion

        #region Aggregation Method Tests - Weighted

        [Fact]
        public void Retrieve_WeightedAggregation_AppliesDecreasingWeights()
        {
            // Arrange - Weights: 1/1, 1/2, 1/3 = 1, 0.5, 0.333...
            // Score = (0.9*1 + 0.6*0.5 + 0.3*0.333) / (1 + 0.5 + 0.333)
            //       = (0.9 + 0.3 + 0.1) / 1.833 = 1.3 / 1.833 ≈ 0.709
            var store = CreateMultiVectorStore(3,
                ("doc1", "test content", new[] { 0.9, 0.6, 0.3 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "weighted");

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].RelevanceScore > 0.6 && results[0].RelevanceScore < 0.9,
                $"Weighted score {results[0].RelevanceScore} should be between mean and max");
        }

        [Fact]
        public void Retrieve_WeightedAggregation_FirstVectorMoreImportant()
        {
            // Arrange - Same total scores but different distribution
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.9, 0.3, 0.3 }),  // High first, low rest
                ("doc2", "content 2", new[] { 0.3, 0.3, 0.9 })); // Low first, high last
            var retriever = new MultiVectorRetriever<double>(store, 3, "weighted");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Doc1 should rank higher due to weighted preference for first vector
            Assert.True(results[0].RelevanceScore >= results[1].RelevanceScore);
        }

        #endregion

        #region Aggregation Method Tests - Unknown/Default

        [Fact]
        public void Retrieve_UnknownAggregation_DefaultsToMax()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "test content", new[] { 0.9, 0.5, 0.3 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "unknown_method");

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal(0.9, results[0].RelevanceScore, 3); // Max score
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_FiltersResults()
        {
            // Arrange
            var store = CreateMultiVectorStoreWithMetadata(3,
                ("doc1", "nature content", new[] { 0.9, 0.8, 0.7 },
                    new Dictionary<string, object> { { "category", "nature" } }),
                ("doc2", "urban content", new[] { 0.85, 0.75, 0.65 },
                    new Dictionary<string, object> { { "category", "urban" } }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content", topK: 5,
                new Dictionary<string, object> { { "category", "nature" } }).ToList();

            // Assert
            Assert.Single(results);
            Assert.StartsWith("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_NoMatchingMetadata_ReturnsEmpty()
        {
            // Arrange
            var store = CreateMultiVectorStoreWithMetadata(3,
                ("doc1", "content", new[] { 0.9, 0.8, 0.7 },
                    new Dictionary<string, object> { { "category", "nature" } }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content", topK: 5,
                new Dictionary<string, object> { { "category", "technology" } }).ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_EmptyMetadataFilter_ReturnsAllMatches()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.9, 0.8, 0.7 }),
                ("doc2", "content 2", new[] { 0.85, 0.75, 0.65 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content", topK: 5,
                new Dictionary<string, object>()).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        #endregion

        #region Document ID Extraction Tests

        [Fact]
        public void Retrieve_VectorIdFormat_ExtractsBaseDocumentId()
        {
            // Arrange - Vectors with IDs like "doc1_vector_0", "doc1_vector_1", etc.
            var store = new MockDocumentStore();
            store.AddDocument(new Document<double>("doc1_vector_0", "content") { RelevanceScore = 0.9, HasRelevanceScore = true });
            store.AddDocument(new Document<double>("doc1_vector_1", "content") { RelevanceScore = 0.8, HasRelevanceScore = true });
            store.AddDocument(new Document<double>("doc1_vector_2", "content") { RelevanceScore = 0.7, HasRelevanceScore = true });

            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results); // All vectors should be grouped into one document
        }

        [Fact]
        public void Retrieve_RegularDocumentId_PreservesId()
        {
            // Arrange - Document without "_vector_" format
            var store = new MockDocumentStore();
            store.AddDocument(new Document<double>("regular_doc", "content") { RelevanceScore = 0.9, HasRelevanceScore = true });

            var retriever = new MultiVectorRetriever<double>(store, 1, "max");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("regular_doc", results[0].Id);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_SingleVectorPerDocument_HandlesCorrectly()
        {
            // Arrange
            var store = CreateMultiVectorStore(1,
                ("doc1", "content 1", new[] { 0.9 }),
                ("doc2", "content 2", new[] { 0.8 }));
            var retriever = new MultiVectorRetriever<double>(store, 1, "max");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            Assert.Equal(0.9, results[0].RelevanceScore, 3);
            Assert.Equal(0.8, results[1].RelevanceScore, 3);
        }

        [Fact]
        public void Retrieve_ManyVectorsPerDocument_HandlesCorrectly()
        {
            // Arrange
            var store = CreateMultiVectorStore(10,
                ("doc1", "content", new[] { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05 }));
            var retriever = new MultiVectorRetriever<double>(store, 10, "max");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal(0.9, results[0].RelevanceScore, 3); // Max of all 10 vectors
        }

        [Fact]
        public void Retrieve_ZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        [Fact]
        public void Retrieve_NegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: -1).ToList());
        }

        [Fact]
        public void Retrieve_TopKGreaterThanDocuments_ReturnsAllAvailable()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.9, 0.8, 0.7 }),
                ("doc2", "content 2", new[] { 0.85, 0.75, 0.65 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content", topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_CaseInsensitiveAggregationMethod()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "test content", new[] { 0.9, 0.6, 0.3 }));

            // Act & Assert - All case variations should work
            var retrieverLower = new MultiVectorRetriever<double>(store, 3, "max");
            var retrieverUpper = new MultiVectorRetriever<double>(store, 3, "MAX");
            var retrieverMixed = new MultiVectorRetriever<double>(store, 3, "Max");

            Assert.Single(retrieverLower.Retrieve("test").ToList());
            Assert.Single(retrieverUpper.Retrieve("test").ToList());
            Assert.Single(retrieverMixed.Retrieve("test").ToList());
        }

        [Fact]
        public void Retrieve_DocumentsHaveRelevanceScores()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "content", new[] { 0.9, 0.8, 0.7 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.All(results, r => Assert.True(r.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_ResultsOrderedByAggregatedScore()
        {
            // Arrange
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.5, 0.5, 0.5 }),  // Max = 0.5
                ("doc2", "content 2", new[] { 0.9, 0.1, 0.1 }),  // Max = 0.9
                ("doc3", "content 3", new[] { 0.7, 0.7, 0.7 })); // Max = 0.7
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Equal(3, results.Count);
            for (int i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].RelevanceScore >= results[i].RelevanceScore,
                    "Results should be ordered by descending score");
            }
        }

        #endregion

        #region Oversampling Tests

        [Fact]
        public void Retrieve_OversamplesToEnsureCoverage()
        {
            // Arrange - The retriever should request topK * vectorsPerDocument * 2 candidates
            var store = CreateMultiVectorStore(3,
                ("doc1", "content 1", new[] { 0.9, 0.8, 0.7 }),
                ("doc2", "content 2", new[] { 0.85, 0.75, 0.65 }),
                ("doc3", "content 3", new[] { 0.8, 0.7, 0.6 }),
                ("doc4", "content 4", new[] { 0.75, 0.65, 0.55 }),
                ("doc5", "content 5", new[] { 0.7, 0.6, 0.5 }));
            var retriever = new MultiVectorRetriever<double>(store, 3, "max");

            // Act
            var results = retriever.Retrieve("content", topK: 2).ToList();

            // Assert
            Assert.True(results.Count <= 2);
            // Should get top 2 by aggregated score
        }

        #endregion
    }
}
