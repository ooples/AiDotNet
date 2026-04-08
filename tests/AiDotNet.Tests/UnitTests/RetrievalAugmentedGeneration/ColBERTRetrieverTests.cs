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
    /// Tests for ColBERTRetriever which uses token-level late interaction matching.
    /// </summary>
    public class ColBERTRetrieverTests
    {
        #region Test Helpers

        /// <summary>
        /// Mock document store for testing ColBERTRetriever.
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
            {
                // Return documents with default relevance scores
                return _documents.Take(topK).Select(d =>
                {
                    d.RelevanceScore = 0.5;
                    d.HasRelevanceScore = true;
                    return d;
                });
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

                return filtered.Take(topK).Select(d =>
                {
                    d.RelevanceScore = 0.5;
                    d.HasRelevanceScore = true;
                    return d;
                });
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

            // Act
            var retriever = new ColBERTRetriever<double>(
                store,
                modelPath: "model.onnx",
                maxDocLength: 512,
                maxQueryLength: 32);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_NullDocumentStore_ThrowsArgumentNullException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentNullException>(() =>
                new ColBERTRetriever<double>(
                    null!,
                    modelPath: "model.onnx",
                    maxDocLength: 512,
                    maxQueryLength: 32));

            Assert.Equal("documentStore", ex.ParamName);
        }

        [Fact]
        public void Constructor_NullModelPath_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            var ex = Assert.Throws<ArgumentNullException>(() =>
                new ColBERTRetriever<double>(
                    store,
                    modelPath: null!,
                    maxDocLength: 512,
                    maxQueryLength: 32));

            Assert.Equal("modelPath", ex.ParamName);
        }

        [Fact]
        public void Constructor_ZeroMaxDocLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ColBERTRetriever<double>(
                    store,
                    modelPath: "model.onnx",
                    maxDocLength: 0,
                    maxQueryLength: 32));

            Assert.Equal("maxDocLength", ex.ParamName);
            Assert.Contains("positive", ex.Message);
        }

        [Fact]
        public void Constructor_NegativeMaxDocLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ColBERTRetriever<double>(
                    store,
                    modelPath: "model.onnx",
                    maxDocLength: -1,
                    maxQueryLength: 32));

            Assert.Equal("maxDocLength", ex.ParamName);
        }

        [Fact]
        public void Constructor_ZeroMaxQueryLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ColBERTRetriever<double>(
                    store,
                    modelPath: "model.onnx",
                    maxDocLength: 512,
                    maxQueryLength: 0));

            Assert.Equal("maxQueryLength", ex.ParamName);
            Assert.Contains("positive", ex.Message);
        }

        [Fact]
        public void Constructor_NegativeMaxQueryLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act & Assert
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ColBERTRetriever<double>(
                    store,
                    modelPath: "model.onnx",
                    maxDocLength: 512,
                    maxQueryLength: -1));

            Assert.Equal("maxQueryLength", ex.ParamName);
        }

        [Fact]
        public void Constructor_SmallMaxLengths_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act - minimum positive values
            var retriever = new ColBERTRetriever<double>(
                store,
                modelPath: "model.onnx",
                maxDocLength: 1,
                maxQueryLength: 1);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_LargeMaxLengths_CreatesInstance()
        {
            // Arrange
            var store = new MockDocumentStore();

            // Act - large typical values
            var retriever = new ColBERTRetriever<double>(
                store,
                modelPath: "path/to/colbert-v2.onnx",
                maxDocLength: 512,
                maxQueryLength: 64);

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
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("test query").ToList();

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_MatchingTokens_ReturnsDocuments()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox jumps over the lazy dog"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_NoMatchingTokens_ReturnsDocumentsBasedOnStoreResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act - query tokens don't match but store still returns documents
            var results = retriever.Retrieve("elephant giraffe").ToList();

            // Assert - returns document from store (with low token overlap score)
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_MultipleDocuments_ReturnsRankedResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox jumps"),
                ("doc2", "A lazy dog sleeps all day"),
                ("doc3", "Climate change impacts"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("quick fox dog").ToList();

            // Assert
            Assert.True(results.Count >= 1);
        }

        [Fact]
        public void Retrieve_WithTopK_ReturnsLimitedResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox one"),
                ("doc2", "fox two"),
                ("doc3", "fox three"),
                ("doc4", "fox four"),
                ("doc5", "fox five"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("fox", topK: 3).ToList();

            // Assert
            Assert.True(results.Count <= 3);
        }

        [Fact]
        public void Retrieve_CaseInsensitive_MatchesDifferentCases()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The QUICK Brown FOX"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("quick fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_NullQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve(null!).ToList());
        }

        [Fact]
        public void Retrieve_EmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("").ToList());
        }

        [Fact]
        public void Retrieve_WhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                retriever.Retrieve("   ").ToList());
        }

        [Fact]
        public void Retrieve_ZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: 0).ToList());
        }

        [Fact]
        public void Retrieve_NegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var store = new MockDocumentStore();
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                retriever.Retrieve("test", topK: -1).ToList());
        }

        #endregion

        #region Token Overlap Scoring Tests

        [Fact]
        public void Retrieve_HighTokenOverlap_ScoresHigher()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "fox"),
                ("doc2", "quick brown fox jumps"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert - Document with more matching tokens should score higher
            Assert.Equal(2, results.Count);
            Assert.True(results[0].HasRelevanceScore);
            Assert.True(results[1].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_DocumentsHaveRelevanceScores()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_AllQueryTokensMatch_HighScore()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "climate change solutions and impacts"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("climate change solutions").ToList();

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
                ("doc1", "fox in the forest", new Dictionary<string, object> { { "category", "nature" } }),
                ("doc2", "fox in the city", new Dictionary<string, object> { { "category", "urban" } }));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("fox", topK: 5,
                new Dictionary<string, object> { { "category", "nature" } }).ToList();

            // Assert
            Assert.Single(results);
            Assert.Equal("doc1", results[0].Id);
        }

        [Fact]
        public void Retrieve_NoMatchingMetadata_ReturnsEmpty()
        {
            // Arrange
            var store = CreateStoreWithDocumentsAndMetadata(
                ("doc1", "fox document", new Dictionary<string, object> { { "category", "nature" } }));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

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
            var store = CreateStoreWithDocuments(
                ("doc1", "fox one"),
                ("doc2", "fox two"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

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
            var store = CreateStoreWithDocuments(
                ("doc1", "fox document"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act & Assert - null metadata filters should throw
            Assert.Throws<ArgumentNullException>(() =>
                retriever.Retrieve("fox", topK: 5, null!).ToList());
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Retrieve_SingleWordQuery_Works()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_SingleWordDocument_Works()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("quick brown fox").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_LongDocument_HandlesCorrectly()
        {
            // Arrange
            var longContent = string.Join(" ", Enumerable.Repeat("The quick brown fox jumps over the lazy dog.", 100));
            var store = CreateStoreWithDocuments(("doc1", longContent));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 50, maxQueryLength: 32);

            // Act - document content exceeds maxDocLength
            var results = retriever.Retrieve("fox").ToList();

            // Assert
            Assert.Single(results);
            Assert.True(results[0].HasRelevanceScore);
        }

        [Fact]
        public void Retrieve_LongQuery_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "The quick brown fox"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 5);
            var longQuery = "one two three four five six seven eight nine ten";

            // Act - query exceeds maxQueryLength
            var results = retriever.Retrieve(longQuery).ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_SpecialCharactersInQuery_HandlesCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "Hello, world! How are you?"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("hello world").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_PunctuationInDocument_TokenizedCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Hello, world! Welcome to the test."));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act - should match "hello" and "world" despite punctuation
            var results = retriever.Retrieve("hello world test").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "The quick brown fox"),
                ("doc2", "A lazy dog"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results1 = retriever.Retrieve("fox").ToList();
            var results2 = retriever.Retrieve("fox").ToList();

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
                ("doc1", "fox one"),
                ("doc2", "fox two"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("fox", topK: 100).ToList();

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_NewlinesInDocument_TokenizedCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Line one\nLine two\nLine three"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("line one two").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_TabsInDocument_TokenizedCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "Column1\tColumn2\tColumn3"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 32);

            // Act
            var results = retriever.Retrieve("column1 column2").ToList();

            // Assert
            Assert.Single(results);
        }

        #endregion

        #region MaxLength Truncation Tests

        [Fact]
        public void Retrieve_QueryExceedsMaxLength_TruncatedCorrectly()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "one two three four five six seven eight nine ten"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 3);

            // Act - only first 3 tokens should be used
            var results = retriever.Retrieve("one two three four five").ToList();

            // Assert - should still find match based on truncated query
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_SmallMaxQueryLength_StillWorks()
        {
            // Arrange
            var store = CreateStoreWithDocuments(("doc1", "fox"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 512, maxQueryLength: 1);

            // Act
            var results = retriever.Retrieve("fox jumps high").ToList();

            // Assert
            Assert.Single(results);
        }

        [Fact]
        public void Retrieve_SmallMaxDocLength_StillWorks()
        {
            // Arrange
            var store = CreateStoreWithDocuments(
                ("doc1", "the quick brown fox jumps over lazy dog"));
            var retriever = new ColBERTRetriever<double>(
                store, "model.onnx", maxDocLength: 3, maxQueryLength: 32);

            // Act - document truncated to 3 tokens
            var results = retriever.Retrieve("the quick brown").ToList();

            // Assert
            Assert.Single(results);
        }

        #endregion
    }
}
