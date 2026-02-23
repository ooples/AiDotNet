#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Xunit;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class ONNXSentenceTransformerTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Assert
            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullModelPath_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>(null, 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithEmptyModelPath_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithWhitespaceModelPath_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("   ", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", 0, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", -1, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithZeroMaxTokens_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 0));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeMaxTokens_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, -1));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact]
        public void Embed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var text = "This is a test sentence.";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(384, embedding.Length);
        }

        [Fact]
        public void Embed_WithSameTextTwice_ReturnsSameEmbedding()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
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
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
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
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
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
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            // Act
            var embeddings = model.EmbedBatch(texts);

            // Assert
            Assert.NotNull(embeddings);
            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(384, embeddings.Columns);
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
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
            var model = new ONNXSentenceTransformer<float>("test-model-path.onnx", 384, 512);
            var text = "Test with float type";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(384, embedding.Length);

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
            var customDimension = 768;
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", customDimension, 512);
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
            var model1 = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var model2 = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
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
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
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
        public void Embed_WithTextContainingMultipleWords_ReturnsEmbedding()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var text = "This is a longer sentence with multiple words";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(384, embedding.Length);
        }

        [Fact]
        public void Embed_WithPunctuationMarks_ReturnsEmbedding()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var text = "Hello, world! How are you?";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(384, embedding.Length);
        }

        [Fact]
        public void Embed_CaseInsensitive_SameEmbedding()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var text1 = "Hello World";
            var text2 = "hello world";

            // Act
            var embedding1 = model.Embed(text1);
            var embedding2 = model.Embed(text2);

            // Assert - Should be same due to ToLowerInvariant in implementation
            for (int i = 0; i < embedding1.Length; i++)
            {
                Assert.Equal(embedding1[i], embedding2[i], 10);
            }
        }

        [Fact]
        public void Embed_WithLongText_ReturnsEmbedding()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var longText = string.Join(" ", Enumerable.Repeat("word", 500));

            // Act
            var embedding = model.Embed(longText);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(384, embedding.Length);
        }
    }
}
