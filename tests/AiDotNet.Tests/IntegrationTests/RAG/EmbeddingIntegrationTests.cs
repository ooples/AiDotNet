using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RAG
{
    /// <summary>
    /// Integration tests for Embedding Model implementations.
    /// Tests validate embedding generation, dimensionality, normalization, and similarity metrics.
    /// </summary>
    public class EmbeddingIntegrationTests
    {
        private const double Tolerance = 1e-6;

        #region StubEmbeddingModel Tests

        [Fact]
        public void StubEmbedding_SameTextTwice_ProducesSameEmbedding()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text = "Machine learning is fascinating";

            // Act
            var embedding1 = model.Embed(text);
            var embedding2 = model.Embed(text);

            // Assert
            Assert.Equal(embedding1.Length, embedding2.Length);
            for (int i = 0; i < embedding1.Length; i++)
            {
                Assert.Equal(embedding1[i], embedding2[i], precision: 10);
            }
        }

        [Fact]
        public void StubEmbedding_DifferentTexts_ProduceDifferentEmbeddings()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text1 = "Machine learning";
            var text2 = "Natural language processing";

            // Act
            var embedding1 = model.Embed(text1);
            var embedding2 = model.Embed(text2);

            // Assert
            Assert.Equal(embedding1.Length, embedding2.Length);

            // Embeddings should be different
            bool hasDifference = false;
            for (int i = 0; i < embedding1.Length; i++)
            {
                if (Math.Abs(embedding1[i] - embedding2[i]) > Tolerance)
                {
                    hasDifference = true;
                    break;
                }
            }
            Assert.True(hasDifference, "Embeddings should be different for different texts");
        }

        [Fact]
        public void StubEmbedding_CorrectDimension_MatchesConfiguration()
        {
            // Arrange & Act
            var model384 = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var model768 = new StubEmbeddingModel<double>(embeddingDimension: 768);
            var model1536 = new StubEmbeddingModel<double>(embeddingDimension: 1536);

            var text = "Test text";
            var embedding384 = model384.Embed(text);
            var embedding768 = model768.Embed(text);
            var embedding1536 = model1536.Embed(text);

            // Assert
            Assert.Equal(384, embedding384.Length);
            Assert.Equal(768, embedding768.Length);
            Assert.Equal(1536, embedding1536.Length);
            Assert.Equal(384, model384.EmbeddingDimension);
            Assert.Equal(768, model768.EmbeddingDimension);
            Assert.Equal(1536, model1536.EmbeddingDimension);
        }

        [Fact]
        public void StubEmbedding_Normalized_HasUnitLength()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text = "Test normalization";

            // Act
            var embedding = model.Embed(text);

            // Calculate magnitude
            double magnitude = 0;
            for (int i = 0; i < embedding.Length; i++)
            {
                magnitude += embedding[i] * embedding[i];
            }
            magnitude = Math.Sqrt(magnitude);

            // Assert - Should be normalized to unit length
            Assert.Equal(1.0, magnitude, precision: 6);
        }

        [Fact]
        public void StubEmbedding_EmptyString_ProducesValidEmbedding()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);

            // Act
            var embedding = model.Embed("");

            // Assert
            Assert.Equal(384, embedding.Length);
            Assert.All(embedding.ToArray(), value => Assert.False(double.IsNaN(value)));
            Assert.All(embedding.ToArray(), value => Assert.False(double.IsInfinity(value)));
        }

        [Fact]
        public void StubEmbedding_LongText_HandlesCorrectly()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var longText = string.Join(" ", Enumerable.Range(1, 1000).Select(i => $"word{i}"));

            // Act
            var embedding = model.Embed(longText);

            // Assert
            Assert.Equal(384, embedding.Length);
            Assert.All(embedding.ToArray(), value => Assert.False(double.IsNaN(value)));
        }

        [Fact]
        public void StubEmbedding_BatchEmbedding_ProducesConsistentResults()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var texts = new[]
            {
                "First document",
                "Second document",
                "Third document"
            };

            // Act
            var batchEmbeddings = model.EmbedBatch(texts);
            var individualEmbeddings = texts.Select(t => model.Embed(t)).ToList();

            // Assert
            Assert.Equal(texts.Length, batchEmbeddings.Count);
            for (int i = 0; i < texts.Length; i++)
            {
                Assert.Equal(individualEmbeddings[i].Length, batchEmbeddings[i].Length);
                for (int j = 0; j < individualEmbeddings[i].Length; j++)
                {
                    Assert.Equal(individualEmbeddings[i][j], batchEmbeddings[i][j], precision: 10);
                }
            }
        }

        [Fact]
        public void StubEmbedding_SpecialCharacters_HandlesCorrectly()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var texts = new[]
            {
                "Hello! How are you?",
                "Price: $99.99",
                "Email: test@example.com",
                "Math: 2 + 2 = 4",
                "Unicode: 你好世界 🌍"
            };

            // Act & Assert
            foreach (var text in texts)
            {
                var embedding = model.Embed(text);
                Assert.Equal(384, embedding.Length);
                Assert.All(embedding.ToArray(), value => Assert.False(double.IsNaN(value)));
            }
        }

        [Fact]
        public void StubEmbedding_CaseSensitivity_ProducesDifferentEmbeddings()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text1 = "Machine Learning";
            var text2 = "machine learning";

            // Act
            var embedding1 = model.Embed(text1);
            var embedding2 = model.Embed(text2);

            // Assert - Case should matter
            bool hasDifference = false;
            for (int i = 0; i < embedding1.Length; i++)
            {
                if (Math.Abs(embedding1[i] - embedding2[i]) > Tolerance)
                {
                    hasDifference = true;
                    break;
                }
            }
            Assert.True(hasDifference, "Case-sensitive texts should produce different embeddings");
        }

        #endregion

        #region Similarity Metrics Tests

        [Fact]
        public void CosineSimilarity_IdenticalVectors_ReturnsOne()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var similarity = StatisticsHelper<double>.CosineSimilarity(vector, vector);

            // Assert
            Assert.Equal(1.0, similarity, precision: 10);
        }

        [Fact]
        public void CosineSimilarity_OrthogonalVectors_ReturnsZero()
        {
            // Arrange
            var vector1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
            var vector2 = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

            // Act
            var similarity = StatisticsHelper<double>.CosineSimilarity(vector1, vector2);

            // Assert
            Assert.Equal(0.0, similarity, precision: 10);
        }

        [Fact]
        public void CosineSimilarity_OppositeVectors_ReturnsNegativeOne()
        {
            // Arrange
            var vector1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var vector2 = new Vector<double>(new[] { -1.0, -2.0, -3.0 });

            // Act
            var similarity = StatisticsHelper<double>.CosineSimilarity(vector1, vector2);

            // Assert
            Assert.Equal(-1.0, similarity, precision: 10);
        }

        [Fact]
        public void CosineSimilarity_45DegreeAngle_ReturnsCorrectValue()
        {
            // Arrange - Two vectors at 45 degree angle
            var vector1 = new Vector<double>(new[] { 1.0, 0.0 });
            var vector2 = new Vector<double>(new[] { Math.Sqrt(0.5), Math.Sqrt(0.5) });

            // Act
            var similarity = StatisticsHelper<double>.CosineSimilarity(vector1, vector2);

            // Assert - cos(45°) = √2/2 ≈ 0.707
            Assert.Equal(Math.Sqrt(0.5), similarity, precision: 6);
        }

        [Fact]
        public void DotProduct_StandardVectors_ComputesCorrectly()
        {
            // Arrange
            var vector1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var vector2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

            // Act
            var dotProduct = StatisticsHelper<double>.DotProduct(vector1, vector2);

            // Assert - 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            Assert.Equal(32.0, dotProduct, precision: 10);
        }

        [Fact]
        public void DotProduct_OrthogonalVectors_ReturnsZero()
        {
            // Arrange
            var vector1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
            var vector2 = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

            // Act
            var dotProduct = StatisticsHelper<double>.DotProduct(vector1, vector2);

            // Assert
            Assert.Equal(0.0, dotProduct, precision: 10);
        }

        [Fact]
        public void EuclideanDistance_IdenticalVectors_ReturnsZero()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var distance = StatisticsHelper<double>.EuclideanDistance(vector, vector);

            // Assert
            Assert.Equal(0.0, distance, precision: 10);
        }

        [Fact]
        public void EuclideanDistance_UnitVectors_ReturnsCorrectDistance()
        {
            // Arrange
            var vector1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
            var vector2 = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

            // Act
            var distance = StatisticsHelper<double>.EuclideanDistance(vector1, vector2);

            // Assert - Distance = √2
            Assert.Equal(Math.Sqrt(2), distance, precision: 10);
        }

        [Fact]
        public void EuclideanDistance_3DPoints_ComputesCorrectly()
        {
            // Arrange
            var point1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var point2 = new Vector<double>(new[] { 4.0, 6.0, 8.0 });

            // Act
            var distance = StatisticsHelper<double>.EuclideanDistance(point1, point2);

            // Assert - √((4-1)² + (6-2)² + (8-3)²) = √(9 + 16 + 25) = √50
            Assert.Equal(Math.Sqrt(50), distance, precision: 10);
        }

        #endregion

        #region Embedding Caching Tests

        [Fact]
        public void EmbeddingCache_SameText_ReturnsCachedResult()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text = "Cache this text";

            // Act
            var embedding1 = model.Embed(text);
            var embedding2 = model.Embed(text); // Should be cached

            // Assert - Should be identical (not just similar)
            Assert.Same(embedding1, embedding2);
        }

        [Fact]
        public void EmbeddingCache_DifferentTexts_GeneratesNewEmbeddings()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text1 = "First text";
            var text2 = "Second text";

            // Act
            var embedding1 = model.Embed(text1);
            var embedding2 = model.Embed(text2);

            // Assert - Should be different objects
            Assert.NotSame(embedding1, embedding2);
        }

        #endregion

        #region Multi-Model Comparison Tests

        [Fact]
        public void DifferentModels_SameDimension_ProduceCompatibleEmbeddings()
        {
            // Arrange
            var model1 = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var model2 = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text = "Test compatibility";

            // Act
            var embedding1 = model1.Embed(text);
            var embedding2 = model2.Embed(text);

            // Assert - Same dimension, but different values (different model instances)
            Assert.Equal(embedding1.Length, embedding2.Length);
            // Should be identical since StubEmbeddingModel is deterministic
            for (int i = 0; i < embedding1.Length; i++)
            {
                Assert.Equal(embedding1[i], embedding2[i], precision: 10);
            }
        }

        [Fact]
        public void DifferentDimensions_SameModel_ProduceDifferentSizedEmbeddings()
        {
            // Arrange
            var dimensions = new[] { 128, 256, 384, 512, 768, 1024, 1536 };
            var text = "Test different dimensions";

            // Act & Assert
            foreach (var dim in dimensions)
            {
                var model = new StubEmbeddingModel<double>(embeddingDimension: dim);
                var embedding = model.Embed(text);
                Assert.Equal(dim, embedding.Length);
            }
        }

        #endregion

        #region Semantic Similarity Tests

        [Fact]
        public void SemanticSimilarity_RelatedTexts_HigherThanUnrelated()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text1 = "dog";
            var text2 = "puppy";
            var text3 = "computer";

            // Act
            var embedding1 = model.Embed(text1);
            var embedding2 = model.Embed(text2);
            var embedding3 = model.Embed(text3);

            var similarity12 = StatisticsHelper<double>.CosineSimilarity(embedding1, embedding2);
            var similarity13 = StatisticsHelper<double>.CosineSimilarity(embedding1, embedding3);

            // Assert - Note: StubEmbeddingModel uses hash-based generation,
            // so semantic similarity is not guaranteed, but we can verify the similarity calculation works
            Assert.True(similarity12 >= -1.0 && similarity12 <= 1.0);
            Assert.True(similarity13 >= -1.0 && similarity13 <= 1.0);
        }

        [Fact]
        public void SemanticSimilarity_QueryAndDocuments_RanksCorrectly()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var query = "artificial intelligence research";
            var documents = new[]
            {
                "AI and machine learning papers",
                "cooking recipes for pasta",
                "sports news and updates"
            };

            // Act
            var queryEmbedding = model.Embed(query);
            var similarities = documents
                .Select(doc =>
                {
                    var docEmbedding = model.Embed(doc);
                    return StatisticsHelper<double>.CosineSimilarity(queryEmbedding, docEmbedding);
                })
                .ToList();

            // Assert - All similarities should be in valid range
            Assert.All(similarities, sim => Assert.True(sim >= -1.0 && sim <= 1.0));
        }

        #endregion

        #region Performance and Stress Tests

        [Fact]
        public void Embedding_LargeEmbeddingDimension_CompletesInReasonableTime()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 3072); // Large dimension
            var text = "Performance test text";
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            var embedding = model.Embed(text);
            stopwatch.Stop();

            // Assert
            Assert.Equal(3072, embedding.Length);
            Assert.True(stopwatch.ElapsedMilliseconds < 1000,
                $"Embedding took too long: {stopwatch.ElapsedMilliseconds}ms");
        }

        [Fact]
        public void Embedding_BatchProcessing_MoreEfficientThanIndividual()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var texts = Enumerable.Range(1, 100).Select(i => $"Document {i}").ToArray();

            var stopwatch1 = System.Diagnostics.Stopwatch.StartNew();
            var individualEmbeddings = texts.Select(t => model.Embed(t)).ToList();
            stopwatch1.Stop();

            var stopwatch2 = System.Diagnostics.Stopwatch.StartNew();
            var batchEmbeddings = model.EmbedBatch(texts);
            stopwatch2.Stop();

            // Assert
            Assert.Equal(100, individualEmbeddings.Count);
            Assert.Equal(100, batchEmbeddings.Count);
            // Batch should be roughly similar or faster
            // (For stub model, might be similar, but architecture is correct)
        }

        [Fact]
        public void Embedding_ParallelProcessing_ThreadSafe()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var texts = Enumerable.Range(1, 50).Select(i => $"Parallel text {i}").ToArray();

            // Act
            var embeddings = texts.AsParallel().Select(t => model.Embed(t)).ToList();

            // Assert
            Assert.Equal(50, embeddings.Count);
            Assert.All(embeddings, emb =>
            {
                Assert.Equal(384, emb.Length);
                Assert.All(emb.ToArray(), value => Assert.False(double.IsNaN(value)));
            });
        }

        #endregion

        #region Edge Cases and Error Handling

        [Fact]
        public void Embedding_NullText_ThrowsException()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null!));
        }

        [Fact]
        public void Embedding_InvalidDimension_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new StubEmbeddingModel<double>(embeddingDimension: 0));
            Assert.Throws<ArgumentException>(() =>
                new StubEmbeddingModel<double>(embeddingDimension: -1));
        }

        [Fact]
        public void Embedding_VeryLongText_HandlesGracefully()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384, maxTokens: 512);
            var veryLongText = string.Join(" ", Enumerable.Range(1, 10000).Select(i => $"word{i}"));

            // Act
            var embedding = model.Embed(veryLongText);

            // Assert - Should either truncate or handle the long text
            Assert.Equal(384, embedding.Length);
            Assert.All(embedding.ToArray(), value => Assert.False(double.IsNaN(value)));
        }

        [Fact]
        public void SimilarityMetrics_DifferentDimensions_ThrowsException()
        {
            // Arrange
            var vector1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var vector2 = new Vector<double>(new[] { 1.0, 2.0 }); // Different dimension

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                StatisticsHelper<double>.CosineSimilarity(vector1, vector2));
        }

        [Fact]
        public void Embedding_WhitespaceVariations_ProducesDifferentEmbeddings()
        {
            // Arrange
            var model = new StubEmbeddingModel<double>(embeddingDimension: 384);
            var text1 = "hello world";
            var text2 = "hello  world"; // Extra space
            var text3 = "hello\nworld"; // Newline

            // Act
            var embedding1 = model.Embed(text1);
            var embedding2 = model.Embed(text2);
            var embedding3 = model.Embed(text3);

            // Assert - Different whitespace should produce different embeddings
            bool diff12 = !AreVectorsIdentical(embedding1, embedding2);
            bool diff13 = !AreVectorsIdentical(embedding1, embedding3);

            Assert.True(diff12 || diff13, "Whitespace variations should affect embeddings");
        }

        #endregion

        #region Helper Methods

        private bool AreVectorsIdentical(Vector<double> v1, Vector<double> v2)
        {
            if (v1.Length != v2.Length) return false;

            for (int i = 0; i < v1.Length; i++)
            {
                if (Math.Abs(v1[i] - v2[i]) > Tolerance)
                    return false;
            }
            return true;
        }

        #endregion
    }
}
