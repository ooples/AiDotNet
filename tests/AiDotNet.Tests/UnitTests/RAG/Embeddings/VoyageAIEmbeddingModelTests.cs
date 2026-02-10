#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Xunit;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class VoyageAIEmbeddingModelTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(16000, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>(null, "voyage-model-path.onnx", "document", 1024));
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", null, "document", 1024));
        }

        [Fact]
        public void Constructor_WithNullInputType_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", null, 1024));
        }

        [Fact]
        public void Embed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var text = "This is a test sentence.";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);
        }

        [Fact]
        public void Embed_WithSameTextTwice_ReturnsSameEmbedding()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var text = "Hello world";

            // Act
            var embedding1 = model.Embed(text);
            var embedding2 = model.Embed(text);

            // Assert
            for (int i = 0; i < embedding1.Length; i++)
            {
                Assert.Equal(embedding1[i], embedding2[i], 10);
            }
        }

        [Fact]
        public void Embed_WithDifferentTexts_ReturnsDifferentEmbeddings()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var text1 = "Hello world";
            var text2 = "Goodbye world";

            // Act
            var embedding1 = model.Embed(text1);
            var embedding2 = model.Embed(text2);

            // Assert
            var hasDifference = false;
            for (int i = 0; i < embedding1.Length; i++)
            {
                if (Math.Abs(embedding1[i] - embedding2[i]) > 1e-10)
                {
                    hasDifference = true;
                    break;
                }
            }
            Assert.True(hasDifference, "Embeddings for different texts should be different");
        }

        [Fact]
        public void Embed_ReturnsNormalizedVector()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var text = "Test normalization";

            // Act
            var embedding = model.Embed(text);

            // Assert
            var magnitude = 0.0;
            for (int i = 0; i < embedding.Length; i++)
            {
                magnitude += embedding[i] * embedding[i];
            }
            magnitude = Math.Sqrt(magnitude);
            Assert.Equal(1.0, magnitude, 5);
        }

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            // Act
            var embeddings = model.EmbedBatch(texts);

            // Assert
            Assert.NotNull(embeddings);
            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(1024, embeddings.Columns);
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First", "Second", "Third" };

            // Act
            var batchEmbeddings = model.EmbedBatch(texts);
            var individualEmbeddings = texts.Select(t => model.Embed(t)).ToList();

            // Assert
            for (int i = 0; i < texts.Count; i++)
            {
                for (int j = 0; j < model.EmbeddingDimension; j++)
                {
                    Assert.Equal(individualEmbeddings[i][j], batchEmbeddings[i, j], 10);
                }
            }
        }

        [Fact]
        public void Embed_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<float>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var text = "Test with float type";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);

            // Check normalization
            var magnitude = 0.0f;
            for (int i = 0; i < embedding.Length; i++)
            {
                magnitude += embedding[i] * embedding[i];
            }
            magnitude = (float)Math.Sqrt(magnitude);
            Assert.Equal(1.0f, magnitude, 5);
        }

        [Fact]
        public void Embed_WithCustomDimension_ReturnsCorrectSize()
        {
            // Arrange
            var customDimension = 512;
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", customDimension);
            var text = "Testing custom dimension";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.Equal(customDimension, embedding.Length);
        }

        [Fact]
        public void Embed_Deterministic_MultipleInstances()
        {
            // Arrange
            var model1 = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var model2 = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var text = "Determinism test";

            // Act
            var embedding1 = model1.Embed(text);
            var embedding2 = model2.Embed(text);

            // Assert
            for (int i = 0; i < embedding1.Length; i++)
            {
                Assert.Equal(embedding1[i], embedding2[i], 10);
            }
        }

        [Fact]
        public void EmbedBatch_AllRowsAreNormalized()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First", "Second", "Third" };

            // Act
            var embeddings = model.EmbedBatch(texts);

            // Assert
            for (int i = 0; i < embeddings.Rows; i++)
            {
                var magnitude = 0.0;
                for (int j = 0; j < embeddings.Columns; j++)
                {
                    magnitude += embeddings[i, j] * embeddings[i, j];
                }
                magnitude = Math.Sqrt(magnitude);
                Assert.Equal(1.0, magnitude, 5);
            }
        }

        [Fact]
        public void Embed_WithLongText_ReturnsEmbedding()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var longText = string.Join(" ", Enumerable.Repeat("word", 2000));

            // Act
            var embedding = model.Embed(longText);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);
        }

        [Fact]
        public void Constructor_WithDifferentInputTypes_CreatesInstances()
        {
            // Arrange & Act
            var documentModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var queryModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "query", 1024);

            // Assert
            Assert.NotNull(documentModel);
            Assert.NotNull(queryModel);
        }

        [Fact]
        public void MaxTokens_ReturnsCorrectValue()
        {
            // Arrange
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            // Act
            var maxTokens = model.MaxTokens;

            // Assert
            Assert.Equal(16000, maxTokens);
        }
    }
}
