using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.QueryExpansion;
using AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;
using AiDotNet.RetrievalAugmentedGeneration.Evaluation;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RAG
{
    /// <summary>
    /// Integration tests for advanced RAG components including:
    /// - Context Compression
    /// - Query Expansion
    /// - Query Processors
    /// - Evaluation Metrics
    /// - Advanced Patterns (Chain of Thought, GraphRAG, etc.)
    /// </summary>
    public class AdvancedRAGIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region ContextCompression Tests

        [Fact]
        public void LLMContextCompressor_CompressesContext_RetainsRelevance()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Machine learning is a subset of AI. It focuses on algorithms that learn from data. Neural networks are a key component."),
                new Document<double>("doc2", "Deep learning uses multiple layers. CNNs are used for images. RNNs process sequences."),
                new Document<double>("doc3", "Data preprocessing is important. Feature engineering improves results. Model validation prevents overfitting.")
            };

            Func<string, List<Document<double>>, List<Document<double>>> compressionFunc = (query, docs) =>
            {
                // Simple compression: keep only sentences mentioning query terms
                return docs.Select(doc =>
                {
                    var sentences = doc.Content.Split('.');
                    var relevant = sentences.Where(s =>
                        query.Split(' ').Any(term => s.Contains(term, StringComparison.OrdinalIgnoreCase)));
                    return new Document<double>(doc.Id, string.Join(". ", relevant).Trim(), doc.Metadata);
                }).Where(doc => !string.IsNullOrWhiteSpace(doc.Content)).ToList();
            };

            var compressor = new LLMContextCompressor<double>(compressionFunc);

            // Act
            var compressed = compressor.Compress("machine learning", documents);
            var compressedList = compressed.ToList();

            // Assert
            Assert.NotEmpty(compressedList);
            Assert.All(compressedList, doc => Assert.NotEmpty(doc.Content));
            Assert.True(compressedList[0].Content.Length < documents[0].Content.Length);
        }

        [Fact]
        public void SelectiveContextCompressor_FiltersIrrelevant_KeepsRelevant()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Relevant content about AI") { RelevanceScore = 0.9, HasRelevanceScore = true },
                new Document<double>("doc2", "Somewhat relevant content") { RelevanceScore = 0.6, HasRelevanceScore = true },
                new Document<double>("doc3", "Not very relevant content") { RelevanceScore = 0.3, HasRelevanceScore = true }
            };

            var compressor = new SelectiveContextCompressor<double>(
                relevanceThreshold: 0.5,
                maxDocuments: 5);

            // Act
            var compressed = compressor.Compress("AI query", documents);
            var compressedList = compressed.ToList();

            // Assert
            Assert.Equal(2, compressedList.Count); // Only docs above 0.5 threshold
            Assert.All(compressedList, doc =>
                Assert.True(Convert.ToDouble(doc.RelevanceScore) >= 0.5));
        }

        [Fact]
        public void DocumentSummarizer_SummarizesLongDocuments_ReducesLength()
        {
            // Arrange
            var longDocument = new Document<double>("doc1",
                string.Join(" ", Enumerable.Range(1, 500).Select(i => $"Word{i}")));

            var documents = new List<Document<double>> { longDocument };

            Func<string, string> summarizerFunc = text =>
            {
                // Simple summarization: take first 50 words
                var words = text.Split(' ');
                return string.Join(" ", words.Take(50));
            };

            var summarizer = new DocumentSummarizer<double>(summarizerFunc);

            // Act
            var summarized = summarizer.Compress("query", documents);
            var summarizedList = summarized.ToList();

            // Assert
            Assert.Single(summarizedList);
            Assert.True(summarizedList[0].Content.Length < longDocument.Content.Length);
        }

        [Fact]
        public void AutoCompressor_AutomaticallyCompresses_BasedOnContext()
        {
            // Arrange
            var documents = Enumerable.Range(1, 10)
                .Select(i => new Document<double>($"doc{i}", $"Document {i} with content. More text here. Additional information."))
                .ToList();

            Func<List<Document<double>>, List<Document<double>>> autoCompressionFunc = docs =>
            {
                // Keep only first 3 documents and truncate content
                return docs.Take(3).Select(d => new Document<double>(d.Id, d.Content.Substring(0, 30), d.Metadata)).ToList();
            };

            var compressor = new AutoCompressor<double>(autoCompressionFunc);

            // Act
            var compressed = compressor.Compress("query", documents);
            var compressedList = compressed.ToList();

            // Assert
            Assert.True(compressedList.Count <= 3);
            Assert.All(compressedList, doc => Assert.True(doc.Content.Length <= 30));
        }

        #endregion

        #region QueryExpansion Tests

        [Fact]
        public void MultiQueryExpansion_GeneratesMultipleQueries_CapturesDifferentAspects()
        {
            // Arrange
            Func<string, List<string>> expansionFunc = query =>
            {
                return new List<string>
                {
                    query,
                    $"{query} tutorial",
                    $"{query} guide",
                    $"learn {query}"
                };
            };

            var expander = new MultiQueryExpansion<double>(expansionFunc);

            // Act
            var expandedQueries = expander.Expand("machine learning");

            // Assert
            Assert.Equal(4, expandedQueries.Count);
            Assert.Contains("machine learning", expandedQueries);
            Assert.Contains("machine learning tutorial", expandedQueries);
        }

        [Fact]
        public void HyDEQueryExpansion_GeneratesHypotheticalDocuments_ImproveRetrieval()
        {
            // Arrange
            Func<string, string> hypotheticalDocGenerator = query =>
            {
                return $"A comprehensive answer to '{query}' would discuss the key concepts, " +
                       $"provide examples, and explain the practical applications.";
            };

            var expander = new HyDEQueryExpansion<double>(hypotheticalDocGenerator);

            // Act
            var hypotheticalDoc = expander.GenerateHypotheticalDocument("What is machine learning?");

            // Assert
            Assert.NotEmpty(hypotheticalDoc);
            Assert.Contains("machine learning", hypotheticalDoc, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void SubQueryExpansion_BreaksComplexQuery_IntoSubQueries()
        {
            // Arrange
            Func<string, List<string>> subQueryGenerator = query =>
            {
                // Simple splitting by "and"
                return query.Split(new[] { " and ", " AND " }, StringSplitOptions.RemoveEmptyEntries).ToList();
            };

            var expander = new SubQueryExpansion<double>(subQueryGenerator);

            // Act
            var subQueries = expander.Expand("machine learning and deep learning and neural networks");

            // Assert
            Assert.Equal(3, subQueries.Count);
            Assert.Contains("machine learning", subQueries);
            Assert.Contains("deep learning", subQueries);
            Assert.Contains("neural networks", subQueries);
        }

        [Fact]
        public void LLMQueryExpansion_EnhancesQuery_WithSemanticVariations()
        {
            // Arrange
            Func<string, List<string>> llmExpansionFunc = query =>
            {
                // Simulate LLM generating semantic variations
                return new List<string>
                {
                    query,
                    query.Replace("ML", "Machine Learning"),
                    query.Replace("AI", "Artificial Intelligence"),
                    $"{query} applications",
                    $"{query} techniques"
                };
            };

            var expander = new LLMQueryExpansion<double>(llmExpansionFunc);

            // Act
            var expanded = expander.Expand("ML and AI applications");

            // Assert
            Assert.True(expanded.Count >= 3);
            Assert.Contains(expanded, q => q.Contains("Machine Learning"));
        }

        #endregion

        #region QueryProcessors Tests

        [Fact]
        public void IdentityQueryProcessor_NoModification_ReturnsSameQuery()
        {
            // Arrange
            var processor = new IdentityQueryProcessor<double>();
            var query = "machine learning algorithms";

            // Act
            var processed = processor.Process(query);

            // Assert
            Assert.Equal(query, processed);
        }

        [Fact]
        public void StopWordRemovalQueryProcessor_RemovesStopWords_KeepsKeywords()
        {
            // Arrange
            var stopWords = new HashSet<string> { "the", "is", "a", "an", "and", "or", "but" };
            var processor = new StopWordRemovalQueryProcessor<double>(stopWords);
            var query = "what is the best machine learning algorithm";

            // Act
            var processed = processor.Process(query);

            // Assert
            Assert.DoesNotContain("the", processed);
            Assert.DoesNotContain("is", processed);
            Assert.Contains("machine", processed);
            Assert.Contains("learning", processed);
        }

        [Fact]
        public void SpellCheckQueryProcessor_CorrectsTypos_InQuery()
        {
            // Arrange
            var corrections = new Dictionary<string, string>
            {
                { "machne", "machine" },
                { "learing", "learning" },
                { "algorithim", "algorithm" }
            };

            Func<string, string> spellCheckFunc = query =>
            {
                foreach (var (wrong, correct) in corrections)
                {
                    query = query.Replace(wrong, correct);
                }
                return query;
            };

            var processor = new SpellCheckQueryProcessor<double>(spellCheckFunc);
            var query = "machne learing algorithim";

            // Act
            var processed = processor.Process(query);

            // Assert
            Assert.Equal("machine learning algorithm", processed);
        }

        [Fact]
        public void KeywordExtractionQueryProcessor_ExtractsKeywords_FromQuery()
        {
            // Arrange
            Func<string, List<string>> keywordExtractor = query =>
            {
                // Simple: split and filter by length
                return query.Split(' ')
                    .Where(word => word.Length > 3)
                    .ToList();
            };

            var processor = new KeywordExtractionQueryProcessor<double>(keywordExtractor);
            var query = "how to learn machine learning and AI";

            // Act
            var processed = processor.Process(query);

            // Assert
            Assert.Contains("learn", processed);
            Assert.Contains("machine", processed);
            Assert.DoesNotContain("how", processed);
            Assert.DoesNotContain("to", processed);
        }

        [Fact]
        public void QueryRewritingProcessor_RewritesQuery_ForBetterRetrieval()
        {
            // Arrange
            Func<string, string> rewriteFunc = query =>
            {
                // Expand abbreviations
                return query
                    .Replace("ML", "machine learning")
                    .Replace("AI", "artificial intelligence")
                    .Replace("NLP", "natural language processing");
            };

            var processor = new QueryRewritingProcessor<double>(rewriteFunc);
            var query = "ML and AI for NLP";

            // Act
            var processed = processor.Process(query);

            // Assert
            Assert.Equal("machine learning and artificial intelligence for natural language processing", processed);
        }

        [Fact]
        public void QueryExpansionProcessor_AddsRelatedTerms_ToQuery()
        {
            // Arrange
            var synonyms = new Dictionary<string, List<string>>
            {
                { "car", new List<string> { "automobile", "vehicle" } },
                { "fast", new List<string> { "quick", "rapid" } }
            };

            Func<string, string> expansionFunc = query =>
            {
                foreach (var (term, syns) in synonyms)
                {
                    if (query.Contains(term))
                    {
                        query += " " + string.Join(" ", syns);
                    }
                }
                return query;
            };

            var processor = new QueryExpansionProcessor<double>(expansionFunc);
            var query = "fast car";

            // Act
            var processed = processor.Process(query);

            // Assert
            Assert.Contains("automobile", processed);
            Assert.Contains("vehicle", processed);
            Assert.Contains("quick", processed);
        }

        #endregion

        #region Evaluation Tests

        [Fact]
        public void FaithfulnessMetric_MeasuresFaithfulness_ToSourceDocuments()
        {
            // Arrange
            var contexts = new List<string>
            {
                "The capital of France is Paris.",
                "Paris has a population of about 2.2 million."
            };
            var answer = "Paris is the capital of France with about 2.2 million people.";

            var metric = new FaithfulnessMetric<double>(
                (ans, ctx) => ctx.Any(c => ans.Contains("Paris") && ans.Contains("France")) ? 1.0 : 0.0);

            // Act
            var score = metric.Evaluate(answer, contexts, "What is the capital of France?");

            // Assert
            Assert.True(score >= 0.0 && score <= 1.0);
        }

        [Fact]
        public void AnswerCorrectnessMetric_MeasuresAnswerQuality_AgainstGroundTruth()
        {
            // Arrange
            var groundTruth = "The capital of France is Paris.";
            var answer = "Paris is the capital of France.";

            var metric = new AnswerCorrectnessMetric<double>(
                (ans, gt) =>
                {
                    // Simple overlap-based scoring
                    var ansWords = ans.ToLower().Split(' ').ToHashSet();
                    var gtWords = gt.ToLower().Split(' ').ToHashSet();
                    var intersection = ansWords.Intersect(gtWords).Count();
                    var union = ansWords.Union(gtWords).Count();
                    return intersection / (double)union;
                });

            // Act
            var score = metric.Evaluate(answer, groundTruth);

            // Assert
            Assert.True(score > 0.5); // High overlap
        }

        [Fact]
        public void ContextRelevanceMetric_MeasuresRelevance_OfRetrievedContext()
        {
            // Arrange
            var query = "machine learning algorithms";
            var contexts = new List<string>
            {
                "Machine learning algorithms learn from data.",
                "Deep learning is a type of machine learning.",
                "Cooking recipes for pasta dishes."
            };

            var metric = new ContextRelevanceMetric<double>(
                (q, ctx) =>
                {
                    var queryTerms = q.ToLower().Split(' ').ToHashSet();
                    return ctx.Count(c =>
                        queryTerms.Any(term => c.ToLower().Contains(term))) / (double)ctx.Count;
                });

            // Act
            var score = metric.Evaluate(query, contexts);

            // Assert
            Assert.True(score >= 0.5); // 2 out of 3 contexts relevant
        }

        [Fact]
        public void AnswerSimilarityMetric_ComparesAnswers_Semantically()
        {
            // Arrange
            var answer1 = "Machine learning is a subset of AI.";
            var answer2 = "ML is a branch of artificial intelligence.";

            var metric = new AnswerSimilarityMetric<double>(
                (ans1, ans2) =>
                {
                    // Simple word overlap
                    var words1 = ans1.ToLower().Split(' ').ToHashSet();
                    var words2 = ans2.ToLower().Split(' ').ToHashSet();
                    return words1.Intersect(words2).Count() / (double)Math.Max(words1.Count, words2.Count);
                });

            // Act
            var score = metric.Evaluate(answer1, answer2);

            // Assert
            Assert.True(score >= 0.0 && score <= 1.0);
        }

        [Fact]
        public void RAGEvaluator_EvaluatesFullPipeline_MultipleMetrics()
        {
            // Arrange
            var faithfulness = new FaithfulnessMetric<double>((ans, ctx) => 0.9);
            var correctness = new AnswerCorrectnessMetric<double>((ans, gt) => 0.85);
            var relevance = new ContextRelevanceMetric<double>((q, ctx) => 0.8);

            var evaluator = new RAGEvaluator<double>(
                new[] { faithfulness, correctness, relevance });

            var testCase = new
            {
                Query = "What is AI?",
                Context = new List<string> { "AI is artificial intelligence." },
                Answer = "Artificial intelligence is the simulation of human intelligence.",
                GroundTruth = "AI is artificial intelligence."
            };

            // Act
            var results = evaluator.Evaluate(
                testCase.Query,
                testCase.Context,
                testCase.Answer,
                testCase.GroundTruth);

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results.Values, score => Assert.True(score >= 0.0 && score <= 1.0));
        }

        #endregion

        #region AdvancedPatterns Tests

        [Fact]
        public void ChainOfThoughtRetriever_DecomposesQuery_RetrievesForEachStep()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var docs = new[]
            {
                "Python is a programming language",
                "Machine learning uses Python",
                "Data science requires programming"
            };

            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var baseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            Func<string, List<string>> decomposer = query =>
            {
                return new List<string> { "What is Python?", "How is Python used in ML?" };
            };

            var cotRetriever = new ChainOfThoughtRetriever<double>(
                baseRetriever,
                decomposer);

            // Act
            var results = cotRetriever.Retrieve("How to use Python for machine learning?");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void SelfCorrectingRetriever_Improves_WithFeedback()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Machine learning content"),
                embeddingModel.Embed("Machine learning content"));
            store.Add(doc);

            var baseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);

            Func<string, List<Document<double>>, string> improveQuery = (query, docs) =>
            {
                if (docs.Count == 0) return query + " tutorial";
                return query;
            };

            var selfCorrectingRetriever = new SelfCorrectingRetriever<double>(
                baseRetriever,
                improveQuery,
                maxIterations: 3);

            // Act
            var results = selfCorrectingRetriever.Retrieve("ML basics");
            var resultList = results.ToList();

            // Assert
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void MultiStepReasoningRetriever_PerformsStepByStep_Reasoning()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var generator = new StubGenerator<double>();

            var docs = new[] { "Step 1 info", "Step 2 info", "Step 3 info" };
            foreach (var (doc, index) in docs.Select((d, i) => (d, i)))
            {
                var embedding = embeddingModel.Embed(doc);
                var vectorDoc = new VectorDocument<double>(
                    new Document<double>($"doc{index}", doc),
                    embedding);
                store.Add(vectorDoc);
            }

            var baseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            var reasoningRetriever = new MultiStepReasoningRetriever<double>(
                baseRetriever,
                generator,
                maxSteps: 3);

            // Act
            var result = reasoningRetriever.RetrieveAndReason("Complex multi-step question");

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Steps);
        }

        [Fact]
        public void TreeOfThoughtsRetriever_ExploresMultiplePaths_FindsBest()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var generator = new StubGenerator<double>();

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Information"),
                embeddingModel.Embed("Information"));
            store.Add(doc);

            var baseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            var totRetriever = new TreeOfThoughtsRetriever<double>(
                baseRetriever,
                generator,
                branchingFactor: 2,
                maxDepth: 3);

            // Act
            var result = totRetriever.RetrieveWithTreeSearch("Complex query requiring exploration");

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.BestPath);
        }

        [Fact]
        public void FLARERetriever_GeneratesQueries_BasedOnNeed()
        {
            // Arrange
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var generator = new StubGenerator<double>();

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Test content"),
                embeddingModel.Embed("Test content"));
            store.Add(doc);

            var baseRetriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);

            var flareRetriever = new FLARERetriever<double>(
                baseRetriever,
                generator,
                confidenceThreshold: 0.5);

            // Act
            var result = flareRetriever.GenerateWithActiveRetrieval("Initial query");

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.GeneratedText);
        }

        #endregion

        #region Integration and End-to-End Tests

        [Fact]
        public void FullRAGPipeline_Chunking_Embedding_Retrieval_Reranking_Works()
        {
            // Arrange - Full pipeline
            var chunkingStrategy = new AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies.FixedSizeChunkingStrategy(
                chunkSize: 100, chunkOverlap: 10);

            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);

            // Chunk and embed documents
            var longDocument = "Machine learning is fascinating. " +
                              "Deep learning uses neural networks. " +
                              "Natural language processing analyzes text. " +
                              "Computer vision processes images. " +
                              "Data science combines statistics and programming.";

            var chunks = chunkingStrategy.Chunk(longDocument);

            int chunkId = 0;
            foreach (var chunk in chunks)
            {
                var embedding = embeddingModel.Embed(chunk);
                var doc = new VectorDocument<double>(
                    new Document<double>($"chunk{chunkId++}", chunk),
                    embedding);
                store.Add(doc);
            }

            // Retrieval
            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 3);
            var retrievedDocs = retriever.Retrieve("neural networks deep learning").ToList();

            // Reranking
            var reranker = new AiDotNet.RetrievalAugmentedGeneration.Rerankers.CrossEncoderReranker<double>(
                (query, doc) => doc.Contains("neural") ? 0.95 : 0.5);
            var rerankedDocs = reranker.Rerank("neural networks", retrievedDocs).ToList();

            // Assert
            Assert.NotEmpty(retrievedDocs);
            Assert.NotEmpty(rerankedDocs);
            Assert.True(rerankedDocs[0].Content.Contains("neural", StringComparison.OrdinalIgnoreCase));
        }

        [Fact]
        public void RAGWithQueryProcessing_PreprocessesQuery_ImproveResults()
        {
            // Arrange
            var stopWords = new HashSet<string> { "what", "is", "the" };
            var queryProcessor = new StopWordRemovalQueryProcessor<double>(stopWords);

            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);

            var doc = new VectorDocument<double>(
                new Document<double>("doc1", "Machine learning algorithms"),
                embeddingModel.Embed("Machine learning algorithms"));
            store.Add(doc);

            // Act
            var rawQuery = "what is the machine learning";
            var processedQuery = queryProcessor.Process(rawQuery);

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);
            var results = retriever.Retrieve(processedQuery);

            // Assert
            Assert.NotEmpty(results);
            Assert.DoesNotContain("what", processedQuery);
            Assert.DoesNotContain("the", processedQuery);
        }

        [Fact]
        public void RAGWithCompression_CompressesContext_BeforeGeneration()
        {
            // Arrange
            var documents = Enumerable.Range(1, 10)
                .Select(i => new Document<double>($"doc{i}", $"Document {i} content with many words and details.")
                {
                    RelevanceScore = 0.9 - i * 0.05,
                    HasRelevanceScore = true
                })
                .ToList();

            var compressor = new SelectiveContextCompressor<double>(
                relevanceThreshold: 0.7,
                maxDocuments: 3);

            // Act
            var compressed = compressor.Compress("query", documents);
            var compressedList = compressed.ToList();

            // Assert
            Assert.True(compressedList.Count <= 3);
            Assert.All(compressedList, doc =>
                Assert.True(Convert.ToDouble(doc.RelevanceScore) >= 0.7));
        }

        #endregion

        #region Performance and Stress Tests

        [Fact]
        public void ComplexRAGPipeline_LargeDataset_CompletesInReasonableTime()
        {
            // Arrange
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var store = new InMemoryDocumentStore<double>(vectorDimension: 384);

            // Add 500 documents
            for (int i = 0; i < 500; i++)
            {
                var embedding = embeddingModel.Embed($"Document {i} content");
                var doc = new VectorDocument<double>(
                    new Document<double>($"doc{i}", $"Document {i} content"),
                    embedding);
                store.Add(doc);
            }

            var retriever = new DenseRetriever<double>(store, embeddingModel, defaultTopK: 10);
            var reranker = new AiDotNet.RetrievalAugmentedGeneration.Rerankers.IdentityReranker<double>();

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            var retrieved = retriever.Retrieve("content").ToList();
            var reranked = reranker.Rerank("content", retrieved).ToList();
            stopwatch.Stop();

            // Assert
            Assert.Equal(10, reranked.Count);
            Assert.True(stopwatch.ElapsedMilliseconds < 5000,
                $"Pipeline took too long: {stopwatch.ElapsedMilliseconds}ms");
        }

        #endregion
    }
}
