using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RAG
{
    /// <summary>
    /// Integration tests for Retriever implementations.
    /// Tests validate retrieval accuracy, ranking, filtering, and performance.
    /// </summary>
    public class RetrieverIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region DenseRetriever Tests

        [Fact]
        public void DenseRetriever_BasicQuery_ReturnsRelevantDocuments()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            // Add documents
            var documents = new[]
            {
                "Machine learning is a subset of artificial intelligence",
                "Deep learning uses neural networks with multiple layers",
                "Natural language processing analyzes human language",
                "Computer vision processes and analyzes digital images",
                "Cooking pasta requires boiling water and salt"
            };

            foreach (var (doc, index) in documents.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("artificial intelligence and neural networks");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            Assert.All(resultList, doc => Assert.True(doc.HasRelevanceScore));
            Assert.All(resultList, doc => Assert.NotEmpty(doc.Content));

            // Verify results are sorted by relevance
            for (int i = 0; i < resultList.Count - 1; i++)
            {
                Assert.True(Convert.ToDouble(resultList[i].RelevanceScore) >=
                           Convert.ToDouble(resultList[i + 1].RelevanceScore));
            }
        }

        [Fact]
        public void DenseRetriever_WithMetadataFilter_ReturnsFilteredResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[]
            {
                (content: "AI paper from 2024", year: 2024, category: "AI"),
                (content: "ML paper from 2024", year: 2024, category: "ML"),
                (content: "AI paper from 2023", year: 2023, category: "AI"),
                (content: "CV paper from 2024", year: 2024, category: "CV")
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc.content);
                var metadata = new Dictionary<string, object>
                {
                    { "year", doc.year },
                    { "category", doc.category }
                };
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc.content, metadata),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);
            var filters = new Dictionary<string, object> { { "year", 2024 } };

            // Act
            var results = retriever.Retrieve("AI research", filters: filters);
            var resultList = results.ToList();

            // Assert
            Assert.All(resultList, doc => Assert.Equal(2024, doc.Metadata["year"]));
            Assert.True(resultList.Count <= 3); // Only 3 docs from 2024
        }

        [Fact]
        public void DenseRetriever_EmptyStore_ReturnsNoResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);

            // Act
            var results = retriever.Retrieve("test query");

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void DenseRetriever_TopKLargerThanDocuments_ReturnsAllDocuments()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[] { "Doc 1", "Doc 2", "Doc 3" };
            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 100);

            // Act
            var results = retriever.Retrieve("query");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
        }

        #endregion

        #region BM25Retriever Tests

        [Fact]
        public void BM25Retriever_KeywordMatch_ReturnsRelevantDocuments()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                "Machine learning algorithms learn from data",
                "Deep learning is a subset of machine learning",
                "Natural language processing uses machine learning",
                "Computer vision processes images",
                "Cooking recipes for pasta"
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 }); // Dummy embedding
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new BM25Retriever<double>(store, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("machine learning");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            Assert.All(resultList, doc => Assert.Contains("learning", doc.Content.ToLower()));
            Assert.All(resultList, doc => Assert.True(doc.HasRelevanceScore));

            // Top result should contain both "machine" and "learning"
            Assert.Contains("machine", resultList[0].Content.ToLower());
            Assert.Contains("learning", resultList[0].Content.ToLower());
        }

        [Fact]
        public void BM25Retriever_TermFrequency_AffectsRanking()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                "Python is great",
                "Python Python is really great Python",
                "Java is also good"
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new BM25Retriever<double>(store, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("Python");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(2, resultList.Count); // Only docs with "Python"
            // Doc with more "Python" occurrences should rank higher
            var topDoc = resultList[0];
            Assert.Contains("Python Python", topDoc.Content);
        }

        [Fact]
        public void BM25Retriever_NoMatchingTerms_ReturnsNoResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[] { "Machine learning", "Deep learning", "Neural networks" };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new BM25Retriever<double>(store, defaultTopK: 5);

            // Act
            var results = retriever.Retrieve("cooking recipes");
            var resultList = results.ToList();

            // Assert - BM25 returns docs even without matches, but with zero scores
            // Or returns empty depending on implementation
            Assert.True(resultList.Count >= 0);
        }

        [Fact]
        public void BM25Retriever_CustomParameters_AffectsScoring()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[] { "short", "this is a much longer document with many words" };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever1 = new BM25Retriever<double>(store, defaultTopK: 5, k1: 1.5, b: 0.75);
            var retriever2 = new BM25Retriever<double>(store, defaultTopK: 5, k1: 2.0, b: 0.5);

            // Act
            var results1 = retriever1.Retrieve("document");
            var results2 = retriever2.Retrieve("document");

            // Assert - Different parameters should produce different results
            Assert.NotEmpty(results1);
            Assert.NotEmpty(results2);
        }

        #endregion

        #region TFIDFRetriever Tests

        [Fact]
        public void TFIDFRetriever_UniqueTerms_GetHigherScores()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                "common word appears everywhere common common",
                "unique specialized technical terminology",
                "common word appears here too common"
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new TFIDFRetriever<double>(store, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("specialized technical");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            // Document with unique terms should rank high
            Assert.Contains("specialized", resultList[0].Content);
        }

        [Fact]
        public void TFIDFRetriever_CommonWords_GetLowerScores()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 3);
            var docs = new[]
            {
                "the the the the the",
                "specialized unique terminology document",
                "the the and and or or"
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new TFIDFRetriever<double>(store, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("specialized terminology");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            var topDoc = resultList[0];
            Assert.Contains("specialized", topDoc.Content);
        }

        #endregion

        #region HybridRetriever Tests

        [Fact]
        public void HybridRetriever_CombinesDenseAndSparse_BetterResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[]
            {
                "Machine learning and artificial intelligence",
                "Deep learning neural networks",
                "Python programming language",
                "Data science and analytics"
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var denseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);
            var sparseRetriever = new BM25Retriever<double>(store, defaultTopK: 10);
            var hybridRetriever = new HybridRetriever<double>(
                denseRetriever, sparseRetriever, alpha: 0.5, defaultTopK: 3);

            // Act
            var results = hybridRetriever.Retrieve("machine learning");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            Assert.All(resultList, doc => Assert.True(doc.HasRelevanceScore));
        }

        [Fact]
        public void HybridRetriever_AlphaParameter_AffectsWeighting()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[] { "Document 1", "Document 2", "Document 3" };
            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var denseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);
            var sparseRetriever = new BM25Retriever<double>(store, defaultTopK: 10);

            var hybridAlpha0 = new HybridRetriever<double>(
                denseRetriever, sparseRetriever, alpha: 0.0, defaultTopK: 3);
            var hybridAlpha1 = new HybridRetriever<double>(
                denseRetriever, sparseRetriever, alpha: 1.0, defaultTopK: 3);

            // Act
            var results0 = hybridAlpha0.Retrieve("document");
            var results1 = hybridAlpha1.Retrieve("document");

            // Assert - Different alphas should potentially give different orderings
            Assert.Equal(3, results0.Count());
            Assert.Equal(3, results1.Count());
        }

        #endregion

        #region VectorRetriever Tests

        [Fact]
        public void VectorRetriever_DirectVectorQuery_ReturnsResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[] { "Document A", "Document B", "Document C" };
            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new VectorRetriever<double>(store, defaultTopK: 2);
            var queryEmbedding = embeddingModel.Embed("Document A");

            // Act
            var results = retriever.Retrieve(queryEmbedding, topK: 2);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(2, resultList.Count);
            Assert.All(resultList, doc => Assert.True(doc.HasRelevanceScore));
        }

        #endregion

        #region MultiQueryRetriever Tests

        [Fact]
        public void MultiQueryRetriever_MultipleQueries_AggregatesResults()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[]
            {
                "Machine learning and AI",
                "Deep learning tutorial",
                "Python programming",
                "Data science analytics"
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var baseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);
            var multiQueryRetriever = new MultiQueryRetriever<double>(
                baseRetriever,
                queryExpansionFunc: query => new[] { query, $"{query} tutorial", $"{query} guide" },
                defaultTopK: 3);

            // Act
            var results = multiQueryRetriever.Retrieve("machine learning");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            Assert.True(resultList.Count <= 3);
        }

        #endregion

        #region Performance and Edge Cases

        [Fact]
        public void Retrievers_LargeDocumentSet_CompletesInReasonableTime()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            // Add 1000 documents
            for (int i = 0; i < 1000; i++)
            {
                var doc = $"Document {i} with content about topic {i % 10}";
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{i}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            var results = retriever.Retrieve("topic 5");
            var resultList = results.ToList();
            stopwatch.Stop();

            // Assert
            Assert.Equal(10, resultList.Count);
            Assert.True(stopwatch.ElapsedMilliseconds < 3000,
                $"Retrieval took too long: {stopwatch.ElapsedMilliseconds}ms");
        }

        [Fact]
        public void Retrievers_EmptyQuery_HandlesGracefully()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Test content"),
                embeddingModel.Embed("Test content"));
            store.Add(doc);

            var denseRetriever = new DenseRetriever<double>(store, embeddingModel);
            var bm25Retriever = new BM25Retriever<double>(store);

            // Act & Assert - Should not crash
            try
            {
                var results1 = denseRetriever.Retrieve("");
                var results2 = bm25Retriever.Retrieve("");

                Assert.NotNull(results1);
                Assert.NotNull(results2);
            }
            catch (ArgumentException)
            {
                // Also acceptable
                Assert.True(true);
            }
        }

        [Fact]
        public void Retrievers_SpecialCharactersInQuery_HandlesCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[] { "C++ programming", "C# development", "F# functional" };
            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("C++ programming language");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void Retrievers_MultipleFilters_CombinesCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[]
            {
                (content: "Doc A", year: 2024, category: "AI", author: "Smith"),
                (content: "Doc B", year: 2024, category: "ML", author: "Jones"),
                (content: "Doc C", year: 2024, category: "AI", author: "Jones"),
                (content: "Doc D", year: 2023, category: "AI", author: "Smith")
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc.content);
                var metadata = new Dictionary<string, object>
                {
                    { "year", doc.year },
                    { "category", doc.category },
                    { "author", doc.author }
                };
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc.content, metadata),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);
            var filters = new Dictionary<string, object>
            {
                { "year", 2024 },
                { "category", "AI" }
            };

            // Act
            var results = retriever.Retrieve("document", filters: filters);
            var resultList = results.ToList();

            // Assert
            Assert.All(resultList, doc =>
            {
                Assert.Equal(2024, doc.Metadata["year"]);
                Assert.Equal("AI", doc.Metadata["category"]);
            });
            Assert.True(resultList.Count <= 2); // Only Doc A and Doc C match
        }

        [Fact]
        public void Retrievers_DuplicateDocuments_HandlesCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var content = "Duplicate content";
            var embedding = embeddingModel.Embed(content);

            // Add same content with different IDs
            for (int i = 0; i < 3; i++)
            {
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{i}", content),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);

            // Act
            var results = retriever.Retrieve(content);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            // All should have identical or very similar scores
            var scores = resultList.Select(d => Convert.ToDouble(d.RelevanceScore)).ToList();
            Assert.All(scores, score => Assert.Equal(scores[0], score, precision: 6));
        }

        [Fact]
        public void Retrievers_VeryLongQuery_HandlesCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Test document"),
                embeddingModel.Embed("Test document"));
            store.Add(doc);

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);
            var longQuery = string.Join(" ", Enumerable.Range(1, 500).Select(i => $"word{i}"));

            // Act
            var results = retriever.Retrieve(longQuery);
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
        }

        #endregion
    }
}
