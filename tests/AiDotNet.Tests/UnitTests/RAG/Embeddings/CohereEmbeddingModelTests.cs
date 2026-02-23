#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Xunit;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class CohereEmbeddingModelTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithDefaultDimension_CreatesInstance()
        {
            // Arrange & Act
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Assert
            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>(null, "embed-english-v3.0", "search_document"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithEmptyApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("", "embed-english-v3.0", "search_document"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithWhitespaceApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("   ", "embed-english-v3.0", "search_document"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", null, "search_document"));
            Assert.Contains("Model cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithEmptyModel_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "", "search_document"));
            Assert.Contains("Model cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithNullInputType_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", null));
            Assert.Contains("Input type cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithEmptyInputType_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", ""));
            Assert.Contains("Input type cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 0));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", -1));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Embed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
        public void Embed_WithDifferentInputTypes_ReturnsDifferentEmbeddings()
        {
            // Arrange
            var model1 = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
            var model2 = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_query", 1024);
            var text = "Test text";

            // Act
            var embedding1 = model1.Embed(text);
            var embedding2 = model2.Embed(text);

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
            Assert.True(hasDifference, "Embeddings with different input types should be different");
        }

        [Fact]
        public void Embed_ReturnsNormalizedVector()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
            var model = new CohereEmbeddingModel<float>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", customDimension);
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
            var model1 = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
            var model2 = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
        public void Constructor_WithValidInputTypes_CreatesInstances()
        {
            // Arrange & Act
            var documentModel = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");
            var queryModel = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_query");
            var classificationModel = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "classification");
            var clusteringModel = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "clustering");

            // Assert
            Assert.NotNull(documentModel);
            Assert.NotNull(queryModel);
            Assert.NotNull(classificationModel);
            Assert.NotNull(clusteringModel);
        }

        [Fact]
        public void EmbedBatch_AllRowsAreNormalized()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
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
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);
            var longText = string.Join(" ", Enumerable.Repeat("word", 500));

            // Act
            var embedding = model.Embed(longText);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);
        }
    }
}
