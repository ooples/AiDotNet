#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class OpenAIEmbeddingModelTests
    {
        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(1536, model.EmbeddingDimension);
            Assert.Equal(8191, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new OpenAIEmbeddingModel<double>("test-api-key");

            // Assert
            Assert.NotNull(model);
            Assert.Equal(1536, model.EmbeddingDimension);
            Assert.Equal(8191, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>(null, "text-embedding-ada-002"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("", "text-embedding-ada-002"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithWhitespaceApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("   ", "text-embedding-ada-002"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullModelName_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", null));
            Assert.Contains("Model name cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyModelName_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", ""));
            Assert.Contains("Model name cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 0, 8191));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", -1, 8191));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroMaxTokens_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 0));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeMaxTokens_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, -1));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
            var text = "This is a test sentence.";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(1536, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSameTextTwice_ReturnsSameEmbedding()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
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

        [Fact(Timeout = 60000)]
        public async Task Embed_WithDifferentTexts_ReturnsDifferentEmbeddings()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
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

        [Fact(Timeout = 60000)]
        public async Task Embed_ReturnsNormalizedVector()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
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

        [Fact(Timeout = 60000)]
        public async Task Embed_WithNullText_ThrowsArgumentException()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            // Act
            var embeddings = model.EmbedBatch(texts);

            // Assert
            Assert.NotNull(embeddings);
            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(1536, embeddings.Columns);
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key");
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
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

        [Fact(Timeout = 60000)]
        public async Task Embed_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<float>("test-api-key", "text-embedding-ada-002", 1536, 8191);
            var text = "Test with float type";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(1536, embedding.Length);

            // Check normalization
            var magnitude = 0.0f;
            for (int i = 0; i < embedding.Length; i++)
            {
                magnitude += embedding[i] * embedding[i];
            }
            magnitude = (float)Math.Sqrt(magnitude);
            Assert.Equal(1.0f, magnitude, 5);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithCustomDimension_ReturnsCorrectSize()
        {
            // Arrange
            var customDimension = 512;
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-3-small", customDimension, 8191);
            var text = "Testing custom dimension";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.Equal(customDimension, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_Deterministic_MultipleInstances()
        {
            // Arrange
            var model1 = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
            var model2 = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
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

        [Fact(Timeout = 60000)]
        public async Task Embed_WithLongText_ReturnsEmbedding()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
            var longText = string.Join(" ", Enumerable.Repeat("word", 1000));

            // Act
            var embedding = model.Embed(longText);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(1536, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDifferentModelNames_CreatesInstances()
        {
            // Arrange & Act
            var adaModel = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
            var smallModel = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-3-small", 1536, 8191);
            var largeModel = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-3-large", 3072, 8191);

            // Assert
            Assert.NotNull(adaModel);
            Assert.NotNull(smallModel);
            Assert.NotNull(largeModel);
            Assert.Equal(3072, largeModel.EmbeddingDimension);
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_AllRowsAreNormalized()
        {
            // Arrange
            var model = new OpenAIEmbeddingModel<double>("test-api-key", "text-embedding-ada-002", 1536, 8191);
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
    }
}
