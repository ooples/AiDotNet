using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for HybridRetriever which combines dense and sparse retrieval strategies.
    /// </summary>
    public class HybridRetrieverTests
    {
        #region Test Helpers

        /// <summary>
        /// Mock retriever for testing HybridRetriever.
        /// </summary>
        private class MockRetriever : IRetriever<double>
        {
            private readonly List<Document<double>> _documents;
            private readonly Func<string, Document<double>, double>? _scoringFunction;

            public int DefaultTopK { get; }

            public MockRetriever(int defaultTopK = 5)
            {
                _documents = new List<Document<double>>();
                DefaultTopK = defaultTopK;
            }

            public MockRetriever(
                IEnumerable<Document<double>> documents,
                Func<string, Document<double>, double>? scoringFunction = null,
                int defaultTopK = 5)
            {
                _documents = documents.ToList();
                _scoringFunction = scoringFunction;
                DefaultTopK = defaultTopK;
            }

            public IEnumerable<Document<double>> Retrieve(string query)
            {
                return Retrieve(query, DefaultTopK);
            }

            public IEnumerable<Document<double>> Retrieve(string query, int topK)
            {
                return Retrieve(query, topK, new Dictionary<string, object>());
            }

            public IEnumerable<Document<double>> Retrieve(string query, int topK, Dictionary<string, object> metadataFilters)
            {
                if (string.IsNullOrWhiteSpace(query))
                    throw new ArgumentException("Query cannot be null or whitespace.", nameof(query));
                if (metadataFilters == null)
                    throw new ArgumentNullException(nameof(metadataFilters));
                if (topK <= 0)
                    throw new ArgumentOutOfRangeException(nameof(topK));

                var filteredDocs = _documents.Where(doc =>
                    MatchesMetadataFilters(doc, metadataFilters)).ToList();

                var scoredDocs = filteredDocs.Select(doc =>
                {
                    var score = _scoringFunction?.Invoke(query, doc) ?? CalculateDefaultScore(query, doc);
                    var result = new Document<double>(doc.Id, doc.Content, doc.Metadata);
                    result.RelevanceScore = score;
                    result.HasRelevanceScore = true;
                    return result;
                })
                .Where(d => d.RelevanceScore > 0)
                .OrderByDescending(d => d.RelevanceScore)
                .Take(topK)
                .ToList();

                return scoredDocs;
            }

            private double CalculateDefaultScore(string query, Document<double> doc)
            {
                var queryTerms = query.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var contentTerms = doc.Content.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var matchCount = queryTerms.Count(q => contentTerms.Contains(q));
                return matchCount > 0 ? matchCount / (double)queryTerms.Length : 0;
            }

            private bool MatchesMetadataFilters(Document<double> doc, Dictionary<string, object> filters)
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
        }

        private MockRetriever CreateMockRetriever(
            params (string id, string content, double score)[] docs)
        {
            var documents = docs.Select(d =>
            {
                var doc = new Document<double>(d.id, d.content);
                doc.RelevanceScore = d.score;
                doc.HasRelevanceScore = true;
                return doc;
            }).ToList();

            return new MockRetriever(documents, (query, doc) =>
            {
                // Return the preset score
                var original = docs.FirstOrDefault(d => d.id == doc.Id);
                return original.score;
            });
        }

        private MockRetriever CreateMockRetrieverWithMetadata(
            params (string id, string content, double score, Dictionary<string, object> metadata)[] docs)
        {
            var documents = docs.Select(d =>
            {
                var doc = new Document<double>(d.id, d.content, d.metadata);
                doc.RelevanceScore = d.score;
                doc.HasRelevanceScore = true;
                return doc;
            }).ToList();

            return new MockRetriever(documents, (query, doc) =>
            {
                var original = docs.FirstOrDefault(d => d.id == doc.Id);
                return original.score;
            });
        }

        #endregion

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();

            // Act
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Assert
            Assert.NotNull(retriever);
            Assert.Equal(5, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomTopK_SetsCorrectly()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();

            // Act
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever, defaultTopK: 10);

            // Assert
            Assert.Equal(10, retriever.DefaultTopK);
        }

        [Fact]
        public void Constructor_WithCustomWeights_CreatesInstance()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();

            // Act
            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 0.5, sparseWeight: 0.5);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_NullDenseRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var sparseRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridRetriever<double>(null!, sparseRetriever));
        }

        [Fact]
        public void Constructor_NullSparseRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridRetriever<double>(denseRetriever, null!));
        }

        [Fact]
        public void Constructor_ZeroTopK_ThrowsArgumentException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new HybridRetriever<double>(denseRetriever, sparseRetriever, defaultTopK: 0));
        }

        [Fact]
        public void Constructor_NegativeTopK_ThrowsArgumentException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new HybridRetriever<double>(denseRetriever, sparseRetriever, defaultTopK: -1));
        }

        #endregion

        #region Retrieve Method Tests

        [Fact]
        public void Retrieve_EmptyRetrievers_ReturnsEmptyResults()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("test query").ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_OnlyDenseResults_ReturnsWeightedDenseResults()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "the quick brown fox", 0.9),
                ("doc2", "jumps over the lazy dog", 0.7));
            var sparseRetriever = CreateMockRetriever(); // Empty

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 0.7, sparseWeight: 0.3);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Dense weight is 0.7, so scores should be 0.9 * 0.7 = 0.63 and 0.7 * 0.7 = 0.49
            Assert.True(results[0].RelevanceScore > results[1].RelevanceScore);
        }

        [Fact]
        public void Retrieve_OnlySparseResults_ReturnsWeightedSparseResults()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(); // Empty
            var sparseRetriever = CreateMockRetriever(
                ("doc1", "the quick brown fox", 0.8),
                ("doc2", "fox in the forest", 0.6));

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 0.7, sparseWeight: 0.3);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Sparse weight is 0.3, so scores should be 0.8 * 0.3 = 0.24 and 0.6 * 0.3 = 0.18
        }

        [Fact]
        public void Retrieve_OverlappingResults_CombinesScores()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "the quick brown fox", 0.9),
                ("doc2", "jumps over the lazy dog", 0.7));
            var sparseRetriever = CreateMockRetriever(
                ("doc1", "the quick brown fox", 0.8), // Same doc
                ("doc3", "a different document", 0.5));

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 0.7, sparseWeight: 0.3);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(3, results.Count);
            // doc1 should have combined score: 0.9 * 0.7 + 0.8 * 0.3 = 0.63 + 0.24 = 0.87
            var doc1 = results.First(r => r.Id == "doc1");
            Assert.True(doc1.HasRelevanceScore);
            Assert.True(doc1.RelevanceScore > 0.8); // Combined should be higher than either alone
        }

        [Fact]
        public void Retrieve_DisjointResults_ReturnsAllDocuments()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "dense only document", 0.9));
            var sparseRetriever = CreateMockRetriever(
                ("doc2", "sparse only document", 0.8));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("document").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            Assert.Contains(results, r => r.Id == "doc1");
            Assert.Contains(results, r => r.Id == "doc2");
        }

        [Fact]
        public void Retrieve_WithTopK_ReturnsLimitedResults()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "fox one", 0.9),
                ("doc2", "fox two", 0.8),
                ("doc3", "fox three", 0.7));
            var sparseRetriever = CreateMockRetriever(
                ("doc4", "fox four", 0.85),
                ("doc5", "fox five", 0.75));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("fox", topK: 3).ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_ResultsOrderedByScore()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "fox", 0.5),
                ("doc2", "fox", 0.9));
            var sparseRetriever = CreateMockRetriever(
                ("doc3", "fox", 0.7));

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 1.0, sparseWeight: 1.0); // Equal weights for easier testing

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.True(results.Count >= 2);
            for (int i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].RelevanceScore >= results[i].RelevanceScore,
                    "Results should be ordered by descending score");
            }
        }

        [Fact]
        public void Retrieve_NullQuery_ThrowsArgumentException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        #endregion

        #region Weight Combination Tests

        [Fact]
        public void Retrieve_EqualWeights_ProducesEqualContribution()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "test document", 1.0));
            var sparseRetriever = CreateMockRetriever(
                ("doc1", "test document", 1.0));

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 0.5, sparseWeight: 0.5);

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            // Score should be 1.0 * 0.5 + 1.0 * 0.5 = 1.0
            Assert.Equal(1.0, results[0].RelevanceScore, 3);
        }

        [Fact]
        public void Retrieve_HigherDenseWeight_FavorsDenseResults()
        {
            // Arrange - Dense has lower raw score but should win with higher weight
            var denseRetriever = CreateMockRetriever(
                ("doc1", "dense doc", 0.6));
            var sparseRetriever = CreateMockRetriever(
                ("doc2", "sparse doc", 0.9));

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 0.9, sparseWeight: 0.1);

            // Act
            var results = retriever.Retrieve("doc").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Dense: 0.6 * 0.9 = 0.54, Sparse: 0.9 * 0.1 = 0.09
            // Dense should be ranked first despite lower raw score
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_ZeroDenseWeight_OnlyUsesSparseScore()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "dense doc", 1.0));
            var sparseRetriever = CreateMockRetriever(
                ("doc1", "dense doc", 0.5));

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 0.0, sparseWeight: 1.0);

            // Act
            var results = retriever.Retrieve("doc").ToList();

            // Assert
            Assert.Single(results);
            // Score should be 1.0 * 0.0 + 0.5 * 1.0 = 0.5
            Assert.Equal(0.5, results[0].RelevanceScore, 3);
        }

        [Fact]
        public void Retrieve_ZeroSparseWeight_OnlyUsesDenseScore()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "dense doc", 0.8));
            var sparseRetriever = CreateMockRetriever(
                ("doc1", "dense doc", 1.0));

            var retriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever,
                denseWeight: 1.0, sparseWeight: 0.0);

            // Act
            var results = retriever.Retrieve("doc").ToList();

            // Assert
            Assert.Single(results);
            // Score should be 0.8 * 1.0 + 1.0 * 0.0 = 0.8
            Assert.Equal(0.8, results[0].RelevanceScore, 3);
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_FiltersResults()
        {
            // Arrange
            var denseRetriever = CreateMockRetrieverWithMetadata(
                ("doc1", "fox in nature", 0.9, new Dictionary<string, object> { { "category", "nature" } }),
                ("doc2", "fox in city", 0.8, new Dictionary<string, object> { { "category", "urban" } }));
            var sparseRetriever = CreateMockRetrieverWithMetadata(
                ("doc3", "forest fox", 0.7, new Dictionary<string, object> { { "category", "nature" } }));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object> { { "category", "nature" } }).ToList();

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.True(
                r.Metadata.TryGetValue("category", out var cat) && Equals(cat, "nature")));
        }

        [Fact]
        public void Retrieve_NoMatchingMetadata_ReturnsEmpty()
        {
            // Arrange
            var denseRetriever = CreateMockRetrieverWithMetadata(
                ("doc1", "fox document", 0.9, new Dictionary<string, object> { { "category", "nature" } }));
            var sparseRetriever = CreateMockRetrieverWithMetadata(
                ("doc2", "another fox", 0.8, new Dictionary<string, object> { { "category", "nature" } }));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object> { { "category", "technology" } }).ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_EmptyMetadataFilter_ReturnsAllMatches()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "fox one", 0.9));
            var sparseRetriever = CreateMockRetriever(
                ("doc2", "fox two", 0.8));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object>()).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_NullMetadataFilter_ThrowsArgumentNullException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(("doc1", "fox", 0.9));
            var sparseRetriever = CreateMockRetriever();
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                retriever.Retrieve("fox", topK: 5, null!).ToList());
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_DocumentWithoutScore_NotIncluded()
        {
            // Arrange - Create a custom mock where some docs don't have scores
            var denseDocuments = new List<Document<double>>
            {
                new Document<double>("doc1", "scored document") { RelevanceScore = 0.9, HasRelevanceScore = true },
                new Document<double>("doc2", "unscored document") { HasRelevanceScore = false }
            };
            var denseRetriever = new MockRetriever(denseDocuments, (q, d) =>
            {
                if (d.Id == "doc1") return 0.9;
                return 0; // Returns 0 for unscored
            });
            var sparseRetriever = CreateMockRetriever();

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("document").ToList();

            // Assert
            // Only scored documents should be included
            Assert.All(results, r => Assert.True(r.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_TopKGreaterThanResults_ReturnsAllAvailable()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "fox", 0.9));
            var sparseRetriever = CreateMockRetriever(
                ("doc2", "fox", 0.8));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("fox", topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_ZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        [Fact]
        public void Retrieve_NegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever();
            var sparseRetriever = CreateMockRetriever();
            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: -1).ToList());
        }

        [Fact]
        public void Retrieve_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "fox document", 0.9),
                ("doc2", "another fox", 0.7));
            var sparseRetriever = CreateMockRetriever(
                ("doc1", "fox document", 0.8),
                ("doc3", "fox related", 0.6));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results1 = retriever.Retrieve("fox").ToList();
            var results2 = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Equal(results1.Count, results2.Count);
            for (int i = 0; i < results1.Count; i++)
            {
                Assert.Equal(results1[i].Id, results2[i].Id);
                Assert.Equal(results1[i].RelevanceScore, results2[i].RelevanceScore, 5);
            }
        }

        [Fact]
        public void Retrieve_LargeNumberOfDocuments_HandlesCorrectly()
        {
            // Arrange
            var denseDocsList = Enumerable.Range(1, 50)
                .Select(i => ($"dense{i}", $"fox document {i}", 0.9 - (i * 0.01)))
                .ToArray();
            var sparseDocsList = Enumerable.Range(1, 50)
                .Select(i => ($"sparse{i}", $"fox text {i}", 0.8 - (i * 0.01)))
                .ToArray();

            var denseRetriever = CreateMockRetriever(denseDocsList);
            var sparseRetriever = CreateMockRetriever(sparseDocsList);

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("fox", topK: 10).ToList();

            // Assert
            Assert.Equal(10, results.Count);
            // Results should be ordered by score
            for (int i = 1; i < results.Count; i++)
            {
                Assert.True(results[i - 1].RelevanceScore >= results[i].RelevanceScore);
            }
        }

        [Fact]
        public void Retrieve_SameDocumentDifferentContent_UsesFirstEncountered()
        {
            // Arrange - Same doc ID with different content in dense vs sparse
            var denseDocuments = new List<Document<double>>
            {
                new Document<double>("doc1", "dense content") { RelevanceScore = 0.9, HasRelevanceScore = true }
            };
            var sparseDocuments = new List<Document<double>>
            {
                new Document<double>("doc1", "sparse content") { RelevanceScore = 0.8, HasRelevanceScore = true }
            };

            var denseRetriever = new MockRetriever(denseDocuments, (q, d) => 0.9);
            var sparseRetriever = new MockRetriever(sparseDocuments, (q, d) => 0.8);

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("content").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
            // The content from the first retriever (dense) should be used
            Assert.Equal("dense content", results[0].Content);
        }

        #endregion

        #region Default Values Tests

        [Fact]
        public void Constructor_DefaultWeights_Uses70Dense30Sparse()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "test", 1.0));
            var sparseRetriever = CreateMockRetriever(
                ("doc1", "test", 1.0));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("test").ToList();

            // Assert
            Assert.Single(results);
            // Default weights: 0.7 dense + 0.3 sparse = 1.0 * 0.7 + 1.0 * 0.3 = 1.0
            Assert.Equal(1.0, results[0].RelevanceScore, 3);
        }

        [Fact]
        public void Retrieve_WithDefaultTopK_UsesFive()
        {
            // Arrange
            var denseRetriever = CreateMockRetriever(
                ("doc1", "fox 1", 0.9),
                ("doc2", "fox 2", 0.85),
                ("doc3", "fox 3", 0.8));
            var sparseRetriever = CreateMockRetriever(
                ("doc4", "fox 4", 0.75),
                ("doc5", "fox 5", 0.7),
                ("doc6", "fox 6", 0.65));

            var retriever = new HybridRetriever<double>(denseRetriever, sparseRetriever);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert - Default topK is 5
            Assert.True(results.Count <= 5);
        }

        #endregion
    }
}
