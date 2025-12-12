using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RAG
{
    /// <summary>
    /// Comprehensive integration tests covering edge cases, real-world scenarios,
    /// and complex interactions between RAG components.
    /// </summary>
    public class ComprehensiveRAGIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region Real-World Scenario Tests

        [Fact]
        public void RealWorldScenario_TechnicalDocumentation_FullPipeline()
        {
            // Arrange - Simulate technical documentation search
            var documentation = @"
                Python Installation Guide

                Step 1: Download Python from python.org
                Step 2: Run the installer
                Step 3: Add Python to PATH
                Step 4: Verify installation with 'python --version'

                Common Issues:
                - Permission denied: Run as administrator
                - Path not found: Check environment variables
                - Version mismatch: Ensure correct version downloaded
            ";

            var chunker = new RecursiveCharacterChunkingStrategy(chunkSize: 150, chunkOverlap: 20);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);

            // Process documentation
            var chunks = chunker.Chunk(documentation);
            int chunkId = 0;
            foreach (var chunk in chunks)
            {
                var embedding = embeddingModel.Embed(chunk);
                var doc = new VectorDocument<double>(
                    new Document<double>($"doc{chunkId++}", chunk),
                    embedding);
                store.Add(doc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("How do I install Python?");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            Assert.True(resultList.Any(d => d.Content.Contains("Download") || d.Content.Contains("installer")));
        }

        [Fact]
        public void RealWorldScenario_CustomerSupport_FAQ_Retrieval()
        {
            // Arrange
            var faqs = new[]
            {
                (question: "How do I reset my password?", answer: "Click 'Forgot Password' on the login page and follow the email instructions."),
                (question: "What payment methods do you accept?", answer: "We accept credit cards, PayPal, and bank transfers."),
                (question: "How long does shipping take?", answer: "Standard shipping takes 5-7 business days. Express shipping is 2-3 days."),
                (question: "Can I return a product?", answer: "Yes, you can return products within 30 days of purchase."),
                (question: "Do you ship internationally?", answer: "Yes, we ship to over 50 countries worldwide.")
            };

            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);

            foreach (var (faq, index) in faqs.Select((f, i) => (f, i)))
            {
                var content = $"Q: {faq.question}\nA: {faq.answer}";
                var embedding = embeddingModel.Embed(content);
                var doc = new VectorDocument<double>(
                    new Document<double>($"faq{index}", content),
                    embedding);
                store.Add(doc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 2);

            // Act
            var results = retriever.Retrieve("I forgot my password");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
            Assert.Contains(resultList, d => d.Content.Contains("password", StringComparison.OrdinalIgnoreCase));
        }

        [Fact]
        public void RealWorldScenario_CodeSearch_FindRelevantSnippets()
        {
            // Arrange
            var codeSnippets = new[]
            {
                @"def calculate_sum(numbers):
                    return sum(numbers)",
                @"def calculate_average(numbers):
                    return sum(numbers) / len(numbers)",
                @"def find_max(numbers):
                    return max(numbers)",
                @"class DataProcessor:
                    def __init__(self):
                        self.data = []",
                @"import pandas as pd
                  df = pd.read_csv('data.csv')"
            };

            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);

            foreach (var (code, index) in codeSnippets.Select((c, i) => (c, i)))
            {
                var embedding = embeddingModel.Embed(code);
                var doc = new VectorDocument<double>(
                    new Document<double>($"code{index}", code),
                    embedding);
                store.Add(doc);
            }

            var retriever = new BM25Retriever<double>(store, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("calculate average of numbers");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void RealWorldScenario_MultilingualSearch_HandlesUnicodeCorrectly()
        {
            // Arrange
            var documents = new[]
            {
                "Hello world - English greeting",
                "Bonjour le monde - French greeting",
                "Hola mundo - Spanish greeting",
                "こんにちは世界 - Japanese greeting",
                "مرحبا بالعالم - Arabic greeting",
                "你好世界 - Chinese greeting"
            };

            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);

            foreach (var (doc, index) in documents.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 6);

            // Act
            var results = retriever.Retrieve("greeting");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(6, resultList.Count);
            Assert.All(resultList, doc => Assert.Contains("greeting", doc.Content));
        }

        #endregion

        #region Edge Cases - Document Content

        [Fact]
        public void EdgeCase_VeryShortDocuments_SingleWords()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var words = new[] { "AI", "ML", "DL", "NLP", "CV" };
            foreach (var (word, index) in words.Select((w, i) => (w, i)))
            {
                var embedding = embeddingModel.Embed(word);
                var doc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", word),
                    embedding);
                store.Add(doc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            // Act
            var results = retriever.Retrieve("AI");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
        }

        [Fact]
        public void EdgeCase_IdenticalDocuments_DifferentIds()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var content = "Identical content in all documents";
            var embedding = embeddingModel.Embed(content);

            for (int i = 0; i < 5; i++)
            {
                var doc = new VectorDocument<double>(
                    new Document<double>($"doc{i}", content),
                    embedding);
                store.Add(doc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);

            // Act
            var results = retriever.Retrieve(content);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(5, resultList.Count);
            var scores = resultList.Select(d => Convert.ToDouble(d.RelevanceScore)).ToList();
            // All scores should be identical
            Assert.All(scores, score => Assert.Equal(scores[0], score, precision: 10));
        }

        [Fact]
        public void EdgeCase_DocumentsWithSpecialCharacters_HandledCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new[]
            {
                "Email: test@example.com",
                "Price: $99.99",
                "Code: def func(): pass",
                "Math: 2 + 2 = 4",
                "URL: https://example.com/path?query=value",
                "Symbols: © ® ™ § ¶",
                "Emoji: 😀 🎉 🚀 ❤️"
            };

            foreach (var (doc, index) in documents.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 7);

            // Act
            var results = retriever.Retrieve("symbols and special characters");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(7, resultList.Count);
        }

        [Fact]
        public void EdgeCase_DocumentsWithControlCharacters_CleanedProperly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new[]
            {
                "Line 1\nLine 2\nLine 3",
                "Tab\tseparated\tvalues",
                "Return\rcarriage",
                "Mixed\n\r\twhitespace"
            };

            foreach (var (doc, index) in documents.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 4);

            // Act
            var results = retriever.Retrieve("whitespace");
            var resultList = results.ToList();

            // Assert
            Assert.Equal(4, resultList.Count);
        }

        #endregion

        #region Edge Cases - Query Variations

        [Fact]
        public void EdgeCase_QueryWithNumbers_HandledCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new[]
            {
                "Python 3.11 released",
                "Java 17 features",
                "C++ 20 standard",
                "JavaScript ES2023"
            };

            foreach (var (doc, index) in documents.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 4);

            // Act
            var results = retriever.Retrieve("Python 3.11");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void EdgeCase_QueryWithPunctuation_ProcessedCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Question answering system"),
                embeddingModel.Embed("Question answering system"));
            store.Add(doc);

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);

            // Act
            var results1 = retriever.Retrieve("question answering system");
            var results2 = retriever.Retrieve("question answering system?");
            var results3 = retriever.Retrieve("question, answering, system!");

            // Assert
            Assert.NotEmpty(results1);
            Assert.NotEmpty(results2);
            Assert.NotEmpty(results3);
        }

        [Fact]
        public void EdgeCase_QueryCaseSensitivity_BM25vsVector()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Machine Learning Algorithms"),
                embeddingModel.Embed("Machine Learning Algorithms"));
            store.Add(doc);

            var denseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);
            var bm25Retriever = new BM25Retriever<double>(store, defaultTopK: 5);

            // Act
            var denseResults1 = denseRetriever.Retrieve("MACHINE LEARNING").ToList();
            var denseResults2 = denseRetriever.Retrieve("machine learning").ToList();
            var bm25Results1 = bm25Retriever.Retrieve("MACHINE LEARNING").ToList();
            var bm25Results2 = bm25Retriever.Retrieve("machine learning").ToList();

            // Assert - All should return results
            Assert.NotEmpty(denseResults1);
            Assert.NotEmpty(denseResults2);
            Assert.NotEmpty(bm25Results1);
            Assert.NotEmpty(bm25Results2);
        }

        #endregion

        #region Edge Cases - Vector Operations

        [Fact]
        public void EdgeCase_ZeroVector_HandledGracefully()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var zeroVector = new Vector<double>(new double[384]); // All zeros

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Normal document"),
                new Vector<double>(Enumerable.Range(0, 384).Select(i => i * 0.01).ToArray()));
            store.Add(doc);

            // Act
            var results = store.GetSimilar(zeroVector, topK: 5);
            var resultList = results.ToList();

            // Assert - Should not crash, may return results with specific scores
            Assert.NotNull(resultList);
        }

        [Fact]
        public void EdgeCase_NormalizedVsUnnormalizedVectors_SimilarityDiffers()
        {
            // Arrange
            var vec1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var vec2 = new Vector<double>(new[] { 2.0, 4.0, 6.0 }); // Same direction, different magnitude

            // Act
            var cosineSim = StatisticsHelper<double>.CosineSimilarity(vec1, vec2);
            var dotProduct = StatisticsHelper<double>.DotProduct(vec1, vec2);

            // Assert
            Assert.Equal(1.0, cosineSim, precision: 10); // Cosine similarity should be 1 (same direction)
            Assert.True(dotProduct > cosineSim); // Dot product affected by magnitude
        }

        [Fact]
        public void EdgeCase_HighDimensionalSpace_MaintainsAccuracy()
        {
            // Arrange
            var dimensions = new[] { 128, 256, 512, 768, 1536, 3072 };

            foreach (var dim in dimensions)
            {
                var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: dim);
                var store = new InMemoryDocumentStore<double>(vectorDimension: dim);

                var doc = new VectorDocument<double>(
                    new Document<double>("doc1", "Test document"),
                    embeddingModel.Embed("Test document"));
                store.Add(doc);

                var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 1);

                // Act
                var results = retriever.Retrieve("Test document");
                var resultList = results.ToList();

                // Assert
                Assert.Single(resultList);
                Assert.True(Convert.ToDouble(resultList[0].RelevanceScore) > 0.8,
                    $"Low similarity for dimension {dim}");
            }
        }

        #endregion

        #region Edge Cases - Metadata Filtering

        [Fact]
        public void EdgeCase_ComplexMetadataFiltering_MultipleConditions()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new[]
            {
                (content: "Doc 1", year: 2024, category: "AI", score: 9.5, published: true),
                (content: "Doc 2", year: 2024, category: "ML", score: 8.5, published: true),
                (content: "Doc 3", year: 2024, category: "AI", score: 7.5, published: false),
                (content: "Doc 4", year: 2023, category: "AI", score: 9.5, published: true)
            };

            foreach (var (doc, index) in documents.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc.content);
                var metadata = new Dictionary<string, object>
                {
                    { "year", doc.year },
                    { "category", doc.category },
                    { "score", doc.score },
                    { "published", doc.published }
                };
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc.content, metadata),
                    embedding);
                store.Add(vectorDoc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);

            // Act - Multiple filters
            var filters1 = new Dictionary<string, object> { { "year", 2024 }, { "category", "AI" } };
            var results1 = retriever.Retrieve("query", filters: filters1).ToList();

            var filters2 = new Dictionary<string, object> { { "published", true } };
            var results2 = retriever.Retrieve("query", filters: filters2).ToList();

            // Assert
            Assert.True(results1.Count <= 2); // Only Doc 1 and Doc 3, but Doc 3 unpublished
            Assert.All(results1, doc =>
            {
                Assert.Equal(2024, doc.Metadata["year"]);
                Assert.Equal("AI", doc.Metadata["category"]);
            });

            Assert.Equal(3, results2.Count); // Doc 1, 2, 4
        }

        [Fact]
        public void EdgeCase_MetadataWithNullValues_HandledCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var metadata = new Dictionary<string, object>
            {
                { "title", "Test" },
                { "author", null! },
                { "year", 2024 }
            };

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Content", metadata),
                embeddingModel.Embed("Content"));

            // Act & Assert - Should not crash
            Assert.NotNull(doc.Document.Metadata);
            store.Add(doc);

            var retrieved = store.GetById("doc1");
            Assert.NotNull(retrieved);
        }

        #endregion

        #region Edge Cases - Chunking Strategies

        [Fact]
        public void EdgeCase_ChunkSizeEqualsTextLength_ReturnsSingleChunk()
        {
            // Arrange
            var text = "Exactly fifty characters in this text string.";
            var strategy = new FixedSizeChunkingStrategy(chunkSize: text.Length, chunkOverlap: 0);

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.Single(chunkList);
            Assert.Equal(text, chunkList[0]);
        }

        [Fact]
        public void EdgeCase_OverlapLargerThanChunk_HandlesGracefully()
        {
            // Arrange
            var text = "This is a test document with some content.";

            try
            {
                var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 30);

                // Act
                var chunks = strategy.Chunk(text);
                var chunkList = chunks.ToList();

                // Assert - If it doesn't throw, verify it produces valid output
                Assert.NotEmpty(chunkList);
                Assert.All(chunkList, chunk => Assert.NotEmpty(chunk));
            }
            catch (ArgumentException)
            {
                // Also acceptable - implementation may validate and throw
                Assert.True(true);
            }
        }

        [Fact]
        public void EdgeCase_UnicodeChunking_PreservesCharacters()
        {
            // Arrange
            var text = "Hello 世界! こんにちは 🌍 مرحبا";
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 15, chunkOverlap: 3);

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            var reconstructed = string.Join("", chunkList.Select(c => c.Trim()));
            // All original characters should be present (though may have different spacing)
            Assert.Contains("世界", reconstructed);
            Assert.Contains("🌍", reconstructed);
        }

        #endregion

        #region Performance Under Stress

        [Fact]
        public void Stress_ConcurrentRetrieval_ThreadSafe()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            // Add 100 documents
            for (int i = 0; i < 100; i++)
            {
                var doc = new VectorDocument<double>(
                    new Document<double>($"doc{i}", $"Document {i}"),
                    embeddingModel.Embed($"Document {i}"));
                store.Add(doc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);

            // Act - Concurrent retrievals
            var tasks = Enumerable.Range(0, 50).Select(i =>
                Task.Run(() => retriever.Retrieve($"query {i}").ToList())
            ).ToArray();

            Task.WaitAll(tasks);

            // Assert - All tasks should complete successfully
            Assert.All(tasks, task =>
            {
                Assert.True(task.IsCompleted);
                Assert.NotEmpty(task.Result);
            });
        }

        [Fact]
        public void Stress_RapidDocumentAddRemove_MaintainsConsistency()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            // Act - Rapid add/remove cycles
            for (int cycle = 0; cycle < 10; cycle++)
            {
                // Add 50 documents
                for (int i = 0; i < 50; i++)
                {
                    var id = $"doc{cycle}_{i}";
                    var doc = new VectorDocument<double>(
                        new Document<double>(id, $"Content {cycle}_{i}"),
                        embeddingModel.Embed($"Content {cycle}_{i}"));
                    store.Add(doc);
                }

                // Remove half
                for (int i = 0; i < 25; i++)
                {
                    store.Remove($"doc{cycle}_{i}");
                }

                // Verify count
                Assert.Equal(25 * (cycle + 1), store.DocumentCount);
            }

            // Assert
            Assert.Equal(250, store.DocumentCount);
        }

        [Fact]
        public void Stress_VeryLongQueryString_HandlesGracefully()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Test"),
                embeddingModel.Embed("Test"));
            store.Add(doc);

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);
            var longQuery = string.Join(" ", Enumerable.Range(1, 5000).Select(i => $"word{i}"));

            // Act
            var results = retriever.Retrieve(longQuery);
            var resultList = results.ToList();

            // Assert - Should not crash
            Assert.NotEmpty(resultList);
        }

        #endregion

        #region Integration - Component Combinations

        [Fact]
        public void Integration_ChunkingWithEmbedding_ProducesCorrectVectors()
        {
            // Arrange
            var text = "Machine learning is great. Deep learning is powerful. Neural networks are interesting.";
            var chunker = new SentenceChunkingStrategy(chunkSize: 100, chunkOverlap: 0);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            // Act
            var chunks = chunker.Chunk(text);
            var embeddings = chunks.Select(c => embeddingModel.Embed(c)).ToList();

            // Assert
            Assert.NotEmpty(embeddings);
            Assert.All(embeddings, emb => Assert.Equal(384, emb.Length));
            Assert.All(embeddings, emb => Assert.All(emb.ToArray(), val => Assert.False(double.IsNaN(val))));
        }

        [Fact]
        public void Integration_MultipleRerankers_CanBeChained()
        {
            // Arrange
            var documents = Enumerable.Range(1, 10)
                .Select(i => new Document<double>($"doc{i}", $"Content {i}")
                {
                    RelevanceScore = i * 0.1,
                    HasRelevanceScore = true
                })
                .ToList();

            var reranker1 = new IdentityReranker<double>();
            var reranker2 = new LostInTheMiddleReranker<double>();

            // Act
            var results1 = reranker1.Rerank("query", documents);
            var results2 = reranker2.Rerank("query", results1);
            var finalResults = results2.ToList();

            // Assert
            Assert.Equal(10, finalResults.Count);
        }

        [Fact]
        public void Integration_FilteringBeforeAndAfterRetrieval_WorksCorrectly()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            for (int i = 0; i < 20; i++)
            {
                var metadata = new Dictionary<string, object>
                {
                    { "category", i % 2 == 0 ? "Even" : "Odd" },
                    { "value", i }
                };
                var doc = new VectorDocument<double>(
                    new Document<double>($"doc{i}", $"Document {i}", metadata),
                    embeddingModel.Embed($"Document {i}"));
                store.Add(doc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 20);

            // Act - Pre-retrieval filtering
            var filters = new Dictionary<string, object> { { "category", "Even" } };
            var results = retriever.Retrieve("document", filters: filters);

            // Post-retrieval filtering
            var finalResults = results.Where(d => (int)d.Metadata["value"] < 10).ToList();

            // Assert
            Assert.True(finalResults.Count <= 5); // Even numbers < 10: 0, 2, 4, 6, 8
            Assert.All(finalResults, doc =>
            {
                Assert.Equal("Even", doc.Metadata["category"]);
                Assert.True((int)doc.Metadata["value"] < 10);
            });
        }

        #endregion
    }
}
