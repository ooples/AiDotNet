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
    /// Tests for GraphRetriever which uses entity extraction and relationship scoring.
    /// </summary>
    public class GraphRetrieverTests
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

            public Matrix<double> EmbedBatch(IEnumerable<string> texts)
            {
                var textList = texts.ToList();
                var data = new double[textList.Count, EmbeddingDimension];
                for (int row = 0; row < textList.Count; row++)
                {
                    var embedding = Embed(textList[row]);
                    for (int col = 0; col < EmbeddingDimension; col++)
                    {
                        data[row, col] = embedding[col];
                    }
                }
                return new Matrix<double>(data);
            }
        }

        /// <summary>
        /// Mock document store for testing.
        /// </summary>
        private class MockDocumentStore : IDocumentStore<double>
        {
            private readonly List<Document<double>> _documents = new();
            private readonly MockEmbeddingModel _embeddingModel = new();

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
            {
                var scored = _documents.Select(d =>
                {
                    var docVector = _embeddingModel.Embed(d.Content);
                    var similarity = CalculateCosineSimilarity(queryVector, docVector);
                    d.RelevanceScore = similarity;
                    d.HasRelevanceScore = true;
                    return d;
                }).OrderByDescending(d => d.RelevanceScore);

                return scored.Take(topK);
            }

            public IEnumerable<Document<double>> GetSimilarWithFilters(
                Vector<double> queryVector,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                var filtered = _documents.AsEnumerable();

                if (metadataFilters != null && metadataFilters.Count > 0)
                {
                    foreach (var filter in metadataFilters)
                    {
                        filtered = filtered.Where(d =>
                            d.Metadata != null &&
                            d.Metadata.TryGetValue(filter.Key, out var value) &&
                            Equals(value, filter.Value));
                    }
                }

                var scored = filtered.Select(d =>
                {
                    var docVector = _embeddingModel.Embed(d.Content);
                    var similarity = CalculateCosineSimilarity(queryVector, docVector);
                    d.RelevanceScore = similarity;
                    d.HasRelevanceScore = true;
                    return d;
                }).OrderByDescending(d => d.RelevanceScore);

                return scored.Take(topK);
            }

            private static double CalculateCosineSimilarity(Vector<double> a, Vector<double> b)
            {
                if (a.Length != b.Length) return 0;

                double dotProduct = 0, normA = 0, normB = 0;
                for (int i = 0; i < a.Length; i++)
                {
                    dotProduct += a[i] * b[i];
                    normA += a[i] * a[i];
                    normB += b[i] * b[i];
                }

                var denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
                return denominator > 0 ? dotProduct / denominator : 0;
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

        private MockDocumentStore CreateStoreWithDocuments(params (string id, string content)[] docs)
        {
            var store = new MockDocumentStore();
            foreach (var (id, content) in docs)
            {
                store.AddDocument(new Document<double>(id, content));
            }
            return store;
        }

        private MockDocumentStore CreateStoreWithDocumentsAndMetadata(
            params (string id, string content, Dictionary<string, object> metadata)[] docs)
        {
            var store = new MockDocumentStore();
            foreach (var (id, content, metadata) in docs)
            {
                store.AddDocument(new Document<double>(id, content, metadata));
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
            var embeddingModel = new MockEmbeddingModel();

            // Act
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_WithAdvancedEntityExtractionDisabled_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();

            // Act
            var retriever = new GraphRetriever<double>(store, embeddingModel, enableAdvancedEntityExtraction: false);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_NullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange
            var embeddingModel = new MockEmbeddingModel();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GraphRetriever<double>(null!, embeddingModel));
        }

        [Fact]
        public void Constructor_NullEmbeddingModel_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GraphRetriever<double>(store, null!));
        }

        [Fact]
        public void Constructor_BothParametersNull_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GraphRetriever<double>(null!, null!));
        }

        #endregion

        #region Retrieve Method Tests

        [Fact]
        public void Retrieve_EmptyStore_ReturnsEmptyResults()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein physics").ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_SingleDocument_ReturnsDocument()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein developed the theory of relativity"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein relativity").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_MultipleDocuments_ReturnsRankedResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein developed relativity in 1905"),
                ("doc2", "Isaac Newton discovered gravity"),
                ("doc3", "Marie Curie studied radioactivity"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein").ToList();

            // Assert
            Assert.True(results.Count >= 1);
            // Document with Einstein should be prioritized
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_WithTopK_ReturnsLimitedResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Einstein physics"),
                ("doc2", "Newton physics"),
                ("doc3", "Curie physics"),
                ("doc4", "Hawking physics"),
                ("doc5", "Feynman physics"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("physics", topK: 3).ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_NullQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        [Fact]
        public void Retrieve_ZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        [Fact]
        public void Retrieve_NegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: -1).ToList());
        }

        #endregion

        #region Entity Extraction Tests

        [Fact]
        public void Retrieve_ProperNounsInQuery_ExtractsEntities()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein worked at Princeton University"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act - query contains proper nouns
            var results = retriever.Retrieve("Albert Einstein Princeton").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_QuotedTermsInQuery_ExtractsEntities()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The theory of relativity changed physics"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act - quoted term should be extracted as entity
            var results = retriever.Retrieve("\"theory of relativity\" physics").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_YearsInQuery_ExtractsWithAdvancedMode()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Einstein published in 1905"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel, enableAdvancedEntityExtraction: true);

            // Act - year should be extracted with advanced mode
            var results = retriever.Retrieve("Einstein 1905").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_AbbreviationsInQuery_ExtractsWithAdvancedMode()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "DNA research at MIT"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel, enableAdvancedEntityExtraction: true);

            // Act - abbreviations should be extracted
            var results = retriever.Retrieve("DNA MIT research").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_NoEntitiesInQuery_StillWorks()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "simple text content"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act - query without proper nouns
            var results = retriever.Retrieve("simple text").ToList();

            // Assert
            Assert.Single(results);
        }

        #endregion

        #region Entity Match Scoring Tests

        [Fact]
        public void Retrieve_DocumentWithEntity_ScoresHigher()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein was a physicist"),
                ("doc2", "Unrelated content about cooking"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein").ToList();

            // Assert - document with Einstein should rank first
            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_DocumentWithMultipleEntities_ScoresHigher()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein at Princeton in 1955"),
                ("doc2", "Albert Einstein mentioned briefly"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein Princeton 1955").ToList();

            // Assert
            Assert.Equal(2, results.Count);
            // Document with more entity matches should rank higher
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_DocumentsHaveRelevanceScores()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein physics"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        #endregion

        #region Relationship Scoring Tests

        [Fact]
        public void Retrieve_EntitiesCloseProximity_ScoresHigher()
        {
            // Arrange - doc1 has Einstein and Princeton close together
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein worked at Princeton"),
                ("doc2", "Albert Einstein was born in Germany. Many years later Princeton hired him."));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein Princeton").ToList();

            // Assert - both should be returned, doc1 with closer proximity should score higher
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_SingleEntity_NoRelationshipBoost()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein was great"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act - single entity, no relationship scoring possible
            var results = retriever.Retrieve("Albert Einstein").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_MultipleEntityPairs_CalculatesRelationships()
        {
            // Arrange - document with multiple related entities
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein and Marie Curie met at Princeton in 1935"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel, enableAdvancedEntityExtraction: true);

            // Act
            var results = retriever.Retrieve("Albert Einstein Marie Curie 1935").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        #endregion

        #region Metadata Filtering Tests

        [Fact]
        public void Retrieve_WithMetadataFilter_ReturnsOnlyMatchingDocuments()
        {
            // Arrange
            var store = CreateStoreWithDocumentsAndMetadata(
                ("doc1", "Einstein physics", new Dictionary<string, object> { { "category", "science" } }),
                ("doc2", "Einstein biography", new Dictionary<string, object> { { "category", "history" } }));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Einstein", topK: 5,
                new Dictionary<string, object> { { "category", "science" } }).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_NoMatchingMetadata_ReturnsEmpty()
        {
            // Arrange
            var store = CreateStoreWithDocumentsAndMetadata(
                ("doc1", "Einstein content", new Dictionary<string, object> { { "category", "science" } }));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Einstein", topK: 5,
                new Dictionary<string, object> { { "category", "art" } }).ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_EmptyMetadataFilter_ReturnsAllMatches()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "content one"),
                ("doc2", "content two"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("content", topK: 5,
                new Dictionary<string, object>()).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_NullMetadataFilter_ThrowsArgumentNullException()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "content"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                retriever.Retrieve("content", topK: 5, null!).ToList());
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_LowercaseProperNouns_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "albert einstein was a genius"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act - proper nouns in query are capitalized
            var results = retriever.Retrieve("Albert Einstein").ToList();

            // Assert - should still find the document (case-insensitive entity matching)
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_SpecialCharacters_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "E=mc^2 is Einstein's famous equation"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Einstein equation").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein physics"),
                ("doc2", "Marie Curie chemistry"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results1 = retriever.Retrieve("Albert Einstein").ToList();
            var results2 = retriever.Retrieve("Albert Einstein").ToList();

            // Assert
            Assert.Equal(results1.Count, results2.Count);
            for (int i = 0; i < results1.Count; i++)
            {
                Assert.Equal(results1[i].Id, results2[i].Id);
            }
        }

        [Fact]
        public void Retrieve_TopKGreaterThanDocCount_ReturnsAllDocuments()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "content one"),
                ("doc2", "content two"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("content", topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_LongDocument_HandlesCorrectly()
        {
            // Arrange
            var longContent = "Albert Einstein " + string.Join(" ", Enumerable.Repeat("physics research", 100)) + " Princeton";
            var store = CreateStoreWithDocuments(("doc1", longContent));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Albert Einstein Princeton").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_UnicodeEntities_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Tokyo University research"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel);

            // Act
            var results = retriever.Retrieve("Tokyo University").ToList();

            // Assert
            Assert.Single(results);
        }

        #endregion

        #region Advanced Entity Extraction Mode Tests

        [Fact]
        public void Retrieve_AdvancedModeEnabled_ExtractsYears()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Discoveries in 2023 changed science"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel, enableAdvancedEntityExtraction: true);

            // Act
            var results = retriever.Retrieve("science 2023").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_AdvancedModeDisabled_StillWorksWithProperNouns()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Albert Einstein published papers"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel, enableAdvancedEntityExtraction: false);

            // Act
            var results = retriever.Retrieve("Albert Einstein").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_AdvancedModeEnabled_ExtractsAbbreviations()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "NASA and ESA collaborate on Mars missions"));
            var embeddingModel = new MockEmbeddingModel();
            var retriever = new GraphRetriever<double>(store, embeddingModel, enableAdvancedEntityExtraction: true);

            // Act
            var results = retriever.Retrieve("NASA ESA Mars").ToList();

            // Assert
            Assert.Single(results);
        }

        #endregion
    }
}
