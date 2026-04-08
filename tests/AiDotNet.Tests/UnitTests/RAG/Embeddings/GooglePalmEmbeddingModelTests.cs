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
    public class GooglePalmEmbeddingModelTests
    {
        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(768, model.EmbeddingDimension);
            Assert.Equal(2048, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultDimension_CreatesInstance()
        {
            // Arrange & Act
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");

            // Assert
            Assert.NotNull(model);
            Assert.Equal(768, model.EmbeddingDimension);
            Assert.Equal(2048, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullProjectId_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>(null, "us-central1", "textembedding-gecko@001", "test-api-key"));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullLocation_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>("test-project-id", null, "textembedding-gecko@001", "test-api-key"));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", null, "test-api-key"));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", null));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
            var text = "This is a test sentence.";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(768, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSameTextTwice_ReturnsSameEmbedding()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
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
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
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
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
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
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            // Act
            var embeddings = model.EmbedBatch(texts);

            // Assert
            Assert.NotNull(embeddings);
            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(768, embeddings.Columns);
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
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
            var model = new GooglePalmEmbeddingModel<float>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
            var text = "Test with float type";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(768, embedding.Length);

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
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", customDimension);
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
            var model1 = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
            var model2 = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
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
        public async Task EmbedBatch_AllRowsAreNormalized()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
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

        [Fact(Timeout = 60000)]
        public async Task Embed_WithLongText_ReturnsEmbedding()
        {
            // Arrange
            var model = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key", 768);
            var longText = string.Join(" ", Enumerable.Repeat("word", 1000));

            // Act
            var embedding = model.Embed(longText);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(768, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDifferentLocations_CreatesInstances()
        {
            // Arrange & Act
            var usCentralModel = new GooglePalmEmbeddingModel<double>("test-project-id", "us-central1", "textembedding-gecko@001", "test-api-key");
            var europeModel = new GooglePalmEmbeddingModel<double>("test-project-id", "europe-west1", "textembedding-gecko@001", "test-api-key");
            var asiaModel = new GooglePalmEmbeddingModel<double>("test-project-id", "asia-southeast1", "textembedding-gecko@001", "test-api-key");

            // Assert
            Assert.NotNull(usCentralModel);
            Assert.NotNull(europeModel);
            Assert.NotNull(asiaModel);
        }
    }
}
