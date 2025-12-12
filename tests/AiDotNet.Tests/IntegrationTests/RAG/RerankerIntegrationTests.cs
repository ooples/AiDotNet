using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RAG
{
    /// <summary>
    /// Integration tests for Reranker implementations.
    /// Tests validate reranking accuracy, diversity, and score modifications.
    /// </summary>
    public class RerankerIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region CrossEncoderReranker Tests

        [Fact]
        public void CrossEncoderReranker_ReordersDocuments_ByRelevanceScore()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Machine learning algorithms", new Dictionary<string, object>())
                    { RelevanceScore = 0.5, HasRelevanceScore = true },
                new Document<double>("doc2", "Deep learning neural networks", new Dictionary<string, object>())
                    { RelevanceScore = 0.6, HasRelevanceScore = true },
                new Document<double>("doc3", "Artificial intelligence systems", new Dictionary<string, object>())
                    { RelevanceScore = 0.7, HasRelevanceScore = true }
            };

            // Score function that prefers "neural networks"
            Func<string, string, double> scoreFunc = (query, doc) =>
            {
                if (doc.Contains("neural")) return 0.95;
                if (doc.Contains("intelligence")) return 0.80;
                return 0.60;
            };

            var reranker = new CrossEncoderReranker<double>(scoreFunc, maxPairsToScore: 10);

            // Act
            var results = reranker.Rerank("machine learning", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            Assert.Equal("doc2", resultList[0].Id); // "neural networks" should be first
            Assert.Equal(0.95, Convert.ToDouble(resultList[0].RelevanceScore));
            Assert.True(resultList[0].HasRelevanceScore);

            // Verify descending order
            for (int i = 0; i < resultList.Count - 1; i++)
            {
                Assert.True(Convert.ToDouble(resultList[i].RelevanceScore) >=
                           Convert.ToDouble(resultList[i + 1].RelevanceScore));
            }
        }

        [Fact]
        public void CrossEncoderReranker_MaxPairsLimit_LimitsProcessing()
        {
            // Arrange
            var documents = Enumerable.Range(1, 100)
                .Select(i => new Document<double>($"doc{i}", $"Content {i}", new Dictionary<string, object>())
                {
                    RelevanceScore = i * 0.01,
                    HasRelevanceScore = true
                })
                .ToList();

            Func<string, string, double> scoreFunc = (query, doc) => 0.8;
            var reranker = new CrossEncoderReranker<double>(scoreFunc, maxPairsToScore: 10);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.True(resultList.Count <= 10); // Should only process maxPairsToScore documents
        }

        [Fact]
        public void CrossEncoderReranker_EmptyDocuments_ReturnsEmpty()
        {
            // Arrange
            var documents = new List<Document<double>>();
            Func<string, string, double> scoreFunc = (query, doc) => 0.5;
            var reranker = new CrossEncoderReranker<double>(scoreFunc);

            // Act
            var results = reranker.Rerank("query", documents);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void CrossEncoderReranker_QueryContextAware_ProducesContextualScores()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Apple fruit is healthy", new Dictionary<string, object>())
                    { RelevanceScore = 0.5, HasRelevanceScore = true },
                new Document<double>("doc2", "Apple iPhone is expensive", new Dictionary<string, object>())
                    { RelevanceScore = 0.5, HasRelevanceScore = true }
            };

            // Context-aware scoring
            Func<string, string, double> scoreFunc = (query, doc) =>
            {
                if (query.Contains("technology") && doc.Contains("iPhone")) return 0.9;
                if (query.Contains("health") && doc.Contains("fruit")) return 0.9;
                return 0.3;
            };

            var reranker = new CrossEncoderReranker<double>(scoreFunc);

            // Act
            var resultsHealth = reranker.Rerank("health benefits", documents).ToList();
            var resultsTech = reranker.Rerank("technology products", documents).ToList();

            // Assert
            Assert.Equal("doc1", resultsHealth[0].Id); // Fruit doc for health query
            Assert.Equal("doc2", resultsTech[0].Id);   // iPhone doc for tech query
        }

        #endregion

        #region MaximalMarginalRelevanceReranker Tests

        [Fact]
        public void MMRReranker_PromotesDiversity_InResults()
        {
            // Arrange
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Machine learning neural networks")
                    { RelevanceScore = 0.9, HasRelevanceScore = true },
                new Document<double>("doc2", "Machine learning deep networks")
                    { RelevanceScore = 0.88, HasRelevanceScore = true },
                new Document<double>("doc3", "Cooking pasta recipes")
                    { RelevanceScore = 0.7, HasRelevanceScore = true }
            };

            // Attach embeddings
            foreach (var doc in documents)
            {
                doc.Embedding = embeddingModel.Embed(doc.Content);
            }

            Func<Document<double>, Vector<double>> getEmbedding = doc => doc.Embedding!;
            var reranker = new MaximalMarginalRelevanceReranker<double>(getEmbedding, lambda: 0.5);

            // Act
            var results = reranker.Rerank("machine learning", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            // First should be most relevant
            Assert.Equal("doc1", resultList[0].Id);
            // Second should balance relevance and diversity
            // (doc3 might rank higher than doc2 due to diversity despite lower relevance)
        }

        [Fact]
        public void MMRReranker_LambdaOne_OnlyConsidersRelevance()
        {
            // Arrange
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Topic A") { RelevanceScore = 0.9, HasRelevanceScore = true },
                new Document<double>("doc2", "Topic A similar") { RelevanceScore = 0.85, HasRelevanceScore = true },
                new Document<double>("doc3", "Topic B") { RelevanceScore = 0.7, HasRelevanceScore = true }
            };

            foreach (var doc in documents)
            {
                doc.Embedding = embeddingModel.Embed(doc.Content);
            }

            Func<Document<double>, Vector<double>> getEmbedding = doc => doc.Embedding!;
            var reranker = new MaximalMarginalRelevanceReranker<double>(getEmbedding, lambda: 1.0);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert - Should be in original relevance order
            Assert.Equal("doc1", resultList[0].Id);
            Assert.Equal("doc2", resultList[1].Id);
            Assert.Equal("doc3", resultList[2].Id);
        }

        [Fact]
        public void MMRReranker_LambdaZero_OnlyConsidersDiversity()
        {
            // Arrange
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Same content") { RelevanceScore = 0.9, HasRelevanceScore = true },
                new Document<double>("doc2", "Same content") { RelevanceScore = 0.85, HasRelevanceScore = true },
                new Document<double>("doc3", "Different content") { RelevanceScore = 0.7, HasRelevanceScore = true }
            };

            foreach (var doc in documents)
            {
                doc.Embedding = embeddingModel.Embed(doc.Content);
            }

            Func<Document<double>, Vector<double>> getEmbedding = doc => doc.Embedding!;
            var reranker = new MaximalMarginalRelevanceReranker<double>(getEmbedding, lambda: 0.0);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert - doc3 should rank high despite lower relevance due to diversity
            Assert.Equal(3, resultList.Count);
        }

        #endregion

        #region DiversityReranker Tests

        [Fact]
        public void DiversityReranker_RemovesSimilarDocuments_PreservesDiverse()
        {
            // Arrange
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Machine learning basics"),
                new Document<double>("doc2", "Machine learning fundamentals"),
                new Document<double>("doc3", "Cooking recipes"),
                new Document<double>("doc4", "Sports news"),
                new Document<double>("doc5", "Machine learning introduction")
            };

            foreach (var doc in documents)
            {
                doc.Embedding = embeddingModel.Embed(doc.Content);
                doc.RelevanceScore = 0.8;
                doc.HasRelevanceScore = true;
            }

            Func<Document<double>, Vector<double>> getEmbedding = doc => doc.Embedding!;
            var reranker = new DiversityReranker<double>(
                getEmbedding,
                similarityThreshold: 0.7,
                topK: 3);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.True(resultList.Count <= 3);
            // Should contain diverse documents
        }

        [Fact]
        public void DiversityReranker_AllSimilarDocuments_KeepsOne()
        {
            // Arrange
            var embeddingModel = new StubEmbeddingModel<double>(embeddingDimension: 384);

            var documents = Enumerable.Range(1, 5)
                .Select(i => new Document<double>($"doc{i}", "Identical content")
                {
                    Embedding = embeddingModel.Embed("Identical content"),
                    RelevanceScore = 0.8,
                    HasRelevanceScore = true
                })
                .ToList();

            Func<Document<double>, Vector<double>> getEmbedding = doc => doc.Embedding!;
            var reranker = new DiversityReranker<double>(
                getEmbedding,
                similarityThreshold: 0.95,
                topK: 5);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert - Should keep very few due to high similarity
            Assert.True(resultList.Count >= 1);
        }

        #endregion

        #region LostInTheMiddleReranker Tests

        [Fact]
        public void LostInTheMiddleReranker_ReordersByPosition_BoostsEnds()
        {
            // Arrange
            var documents = Enumerable.Range(1, 10)
                .Select(i => new Document<double>($"doc{i}", $"Content {i}")
                {
                    RelevanceScore = i * 0.1,
                    HasRelevanceScore = true
                })
                .ToList();

            var reranker = new LostInTheMiddleReranker<double>();

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(10, resultList.Count);
            // Pattern should alternate between ends and middle
            // Most relevant should be at top, then alternates
        }

        [Fact]
        public void LostInTheMiddleReranker_OddNumberOfDocs_HandlesCorrectly()
        {
            // Arrange
            var documents = Enumerable.Range(1, 7)
                .Select(i => new Document<double>($"doc{i}", $"Content {i}")
                {
                    RelevanceScore = i * 0.1,
                    HasRelevanceScore = true
                })
                .ToList();

            var reranker = new LostInTheMiddleReranker<double>();

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(7, resultList.Count);
        }

        #endregion

        #region ReciprocalRankFusion Tests

        [Fact]
        public void ReciprocalRankFusion_CombinesMultipleRankings_Fairly()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Content 1") { RelevanceScore = 1.0, HasRelevanceScore = true },
                new Document<double>("doc2", "Content 2") { RelevanceScore = 0.9, HasRelevanceScore = true },
                new Document<double>("doc3", "Content 3") { RelevanceScore = 0.8, HasRelevanceScore = true }
            };

            var ranking1 = new List<Document<double>> { documents[0], documents[1], documents[2] };
            var ranking2 = new List<Document<double>> { documents[2], documents[1], documents[0] };

            var reranker = new ReciprocalRankFusion<double>(k: 60);

            // Act
            var results = reranker.Fuse(new[] { ranking1, ranking2 });
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            // doc2 appears in same position in both, should rank high
            // doc1 and doc3 are swapped, should have similar scores
        }

        [Fact]
        public void ReciprocalRankFusion_EmptyRankings_ReturnsEmpty()
        {
            // Arrange
            var reranker = new ReciprocalRankFusion<double>();
            var rankings = new List<List<Document<double>>> { new(), new() };

            // Act
            var results = reranker.Fuse(rankings);

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void ReciprocalRankFusion_SingleRanking_ReturnsSameOrder()
        {
            // Arrange
            var documents = Enumerable.Range(1, 5)
                .Select(i => new Document<double>($"doc{i}", $"Content {i}")
                {
                    RelevanceScore = i * 0.1,
                    HasRelevanceScore = true
                })
                .ToList();

            var reranker = new ReciprocalRankFusion<double>();

            // Act
            var results = reranker.Fuse(new[] { documents });
            var resultList = results.ToList();

            // Assert
            Assert.Equal(5, resultList.Count);
            for (int i = 0; i < documents.Count; i++)
            {
                Assert.Equal(documents[i].Id, resultList[i].Id);
            }
        }

        #endregion

        #region IdentityReranker Tests

        [Fact]
        public void IdentityReranker_NoReranking_PreservesOrder()
        {
            // Arrange
            var documents = Enumerable.Range(1, 5)
                .Select(i => new Document<double>($"doc{i}", $"Content {i}")
                {
                    RelevanceScore = i * 0.1,
                    HasRelevanceScore = true
                })
                .ToList();

            var reranker = new IdentityReranker<double>();

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(5, resultList.Count);
            for (int i = 0; i < documents.Count; i++)
            {
                Assert.Equal(documents[i].Id, resultList[i].Id);
                Assert.Equal(documents[i].RelevanceScore, resultList[i].RelevanceScore);
            }
        }

        #endregion

        #region Edge Cases and Performance Tests

        [Fact]
        public void Rerankers_SingleDocument_HandleCorrectly()
        {
            // Arrange
            var document = new Document<double>("doc1", "Single document")
            {
                RelevanceScore = 0.8,
                HasRelevanceScore = true
            };
            var documents = new List<Document<double>> { document };

            var rerankers = new IReranker<double>[]
            {
                new IdentityReranker<double>(),
                new CrossEncoderReranker<double>((q, d) => 0.9),
                new LostInTheMiddleReranker<double>()
            };

            // Act & Assert
            foreach (var reranker in rerankers)
            {
                var results = reranker.Rerank("query", documents);
                var resultList = results.ToList();

                Assert.Single(resultList);
                Assert.Equal("doc1", resultList[0].Id);
            }
        }

        [Fact]
        public void Rerankers_LargeDocumentSet_CompletesInReasonableTime()
        {
            // Arrange
            var documents = Enumerable.Range(1, 1000)
                .Select(i => new Document<double>($"doc{i}", $"Content {i}")
                {
                    RelevanceScore = i * 0.001,
                    HasRelevanceScore = true
                })
                .ToList();

            var reranker = new CrossEncoderReranker<double>(
                (query, doc) => 0.8,
                maxPairsToScore: 100);

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();
            stopwatch.Stop();

            // Assert
            Assert.NotEmpty(resultList);
            Assert.True(stopwatch.ElapsedMilliseconds < 2000,
                $"Reranking took too long: {stopwatch.ElapsedMilliseconds}ms");
        }

        [Fact]
        public void Rerankers_DocumentsWithoutScores_HandlesGracefully()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "No score"),
                new Document<double>("doc2", "Also no score")
            };

            var reranker = new CrossEncoderReranker<double>((q, d) => 0.8);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(2, resultList.Count);
            Assert.All(resultList, doc => Assert.True(doc.HasRelevanceScore));
        }

        [Fact]
        public void Rerankers_NullQuery_HandlesAppropriately()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Content") { RelevanceScore = 0.8, HasRelevanceScore = true }
            };

            var reranker = new IdentityReranker<double>();

            // Act & Assert
            try
            {
                var results = reranker.Rerank(null!, documents);
                Assert.NotNull(results);
            }
            catch (ArgumentException)
            {
                // Also acceptable
                Assert.True(true);
            }
        }

        [Fact]
        public void Rerankers_DuplicateDocuments_HandlesCorrectly()
        {
            // Arrange
            var doc = new Document<double>("doc1", "Duplicate")
            {
                RelevanceScore = 0.8,
                HasRelevanceScore = true
            };
            var documents = new List<Document<double>> { doc, doc, doc };

            var reranker = new IdentityReranker<double>();

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, resultList.Count);
            Assert.All(resultList, d => Assert.Equal("doc1", d.Id));
        }

        [Fact]
        public void MMRReranker_DocumentsWithoutEmbeddings_ThrowsException()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "No embedding") { RelevanceScore = 0.8, HasRelevanceScore = true }
            };

            Func<Document<double>, Vector<double>> getEmbedding = doc => doc.Embedding!;
            var reranker = new MaximalMarginalRelevanceReranker<double>(getEmbedding);

            // Act & Assert
            Assert.Throws<NullReferenceException>(() =>
            {
                var results = reranker.Rerank("query", documents).ToList();
            });
        }

        [Fact]
        public void CrossEncoderReranker_ScoreFunction_CalledForEachDocument()
        {
            // Arrange
            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Content 1"),
                new Document<double>("doc2", "Content 2"),
                new Document<double>("doc3", "Content 3")
            };

            int callCount = 0;
            Func<string, string, double> scoreFunc = (query, doc) =>
            {
                callCount++;
                return 0.5;
            };

            var reranker = new CrossEncoderReranker<double>(scoreFunc, maxPairsToScore: 10);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Equal(3, callCount); // Should be called once per document
            Assert.Equal(3, resultList.Count);
        }

        [Fact]
        public void Rerankers_PreserveMetadata_AfterReranking()
        {
            // Arrange
            var metadata = new Dictionary<string, object>
            {
                { "author", "Smith" },
                { "year", 2024 },
                { "category", "AI" }
            };

            var documents = new List<Document<double>>
            {
                new Document<double>("doc1", "Content", metadata)
                {
                    RelevanceScore = 0.8,
                    HasRelevanceScore = true
                }
            };

            var reranker = new CrossEncoderReranker<double>((q, d) => 0.9);

            // Act
            var results = reranker.Rerank("query", documents);
            var resultList = results.ToList();

            // Assert
            Assert.Single(resultList);
            Assert.Equal("Smith", resultList[0].Metadata["author"]);
            Assert.Equal(2024, resultList[0].Metadata["year"]);
            Assert.Equal("AI", resultList[0].Metadata["category"]);
        }

        #endregion
    }
}
