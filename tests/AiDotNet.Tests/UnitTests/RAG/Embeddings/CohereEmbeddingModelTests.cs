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
    public class CohereEmbeddingModelTests
    {
        private static readonly string CohereApiKey =
            Environment.GetEnvironmentVariable("COHERE_API_KEY") ?? string.Empty;

        private static bool HasApiKey => !string.IsNullOrEmpty(CohereApiKey);

        private CohereEmbeddingModel<T> CreateModel<T>(string inputType = "search_document", int dimension = 1024)
            => new(HasApiKey ? CohereApiKey : "test-api-key", "embed-english-v3.0", inputType, dimension);

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 1024);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultDimension_CreatesInstance()
        {
            // Arrange & Act
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Assert
            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>(null, "embed-english-v3.0", "search_document"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("", "embed-english-v3.0", "search_document"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithWhitespaceApiKey_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("   ", "embed-english-v3.0", "search_document"));
            Assert.Contains("API key cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullModel_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", null, "search_document"));
            Assert.Contains("Model cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyModel_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "", "search_document"));
            Assert.Contains("Model cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullInputType_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", null));
            Assert.Contains("Input type cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyInputType_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", ""));
            Assert.Contains("Input type cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", 0));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document", -1));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var embedding = model.Embed("This is a test sentence.");
            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSameTextTwice_ReturnsSameEmbedding()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var embedding1 = model.Embed("Hello world");
            var embedding2 = model.Embed("Hello world");
            for (int i = 0; i < embedding1.Length; i++)
                Assert.Equal(embedding1[i], embedding2[i], 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithDifferentTexts_ReturnsDifferentEmbeddings()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var embedding1 = model.Embed("Hello world");
            var embedding2 = model.Embed("Goodbye world");
            var hasDifference = false;
            for (int i = 0; i < embedding1.Length; i++)
            {
                if (Math.Abs(embedding1[i] - embedding2[i]) > 1e-10) { hasDifference = true; break; }
            }
            Assert.True(hasDifference, "Embeddings for different texts should be different");
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithDifferentInputTypes_ReturnsDifferentEmbeddings()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model1 = CreateModel<double>("search_document");
            var model2 = CreateModel<double>("search_query");
            var embedding1 = model1.Embed("Test text");
            var embedding2 = model2.Embed("Test text");
            var hasDifference = false;
            for (int i = 0; i < embedding1.Length; i++)
            {
                if (Math.Abs(embedding1[i] - embedding2[i]) > 1e-10) { hasDifference = true; break; }
            }
            Assert.True(hasDifference, "Embeddings with different input types should be different");
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_ReturnsNormalizedVector()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var embedding = model.Embed("Test normalization");
            var magnitude = 0.0;
            for (int i = 0; i < embedding.Length; i++)
                magnitude += embedding[i] * embedding[i];
            magnitude = Math.Sqrt(magnitude);
            Assert.Equal(1.0, magnitude, 5);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithNullText_ThrowsArgumentException()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var texts = new List<string> { "First text", "Second text", "Third text" };
            var embeddings = model.EmbedBatch(texts);
            Assert.NotNull(embeddings);
            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(1024, embeddings.Columns);
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new CohereEmbeddingModel<double>("test-api-key", "embed-english-v3.0", "search_document");
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var texts = new List<string> { "First", "Second", "Third" };
            var batchEmbeddings = model.EmbedBatch(texts);
            var individualEmbeddings = texts.Select(t => model.Embed(t)).ToList();
            for (int i = 0; i < texts.Count; i++)
                for (int j = 0; j < model.EmbeddingDimension; j++)
                    Assert.Equal(individualEmbeddings[i][j], batchEmbeddings[i, j], 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithFloatType_WorksCorrectly()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<float>();
            var embedding = model.Embed("Test with float type");
            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);
            var magnitude = 0.0f;
            for (int i = 0; i < embedding.Length; i++)
                magnitude += embedding[i] * embedding[i];
            magnitude = (float)Math.Sqrt(magnitude);
            Assert.Equal(1.0f, magnitude, 5);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithCustomDimension_ReturnsCorrectSize()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>(dimension: 512);
            var embedding = model.Embed("Testing custom dimension");
            Assert.Equal(512, embedding.Length);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_Deterministic_MultipleInstances()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model1 = CreateModel<double>();
            var model2 = CreateModel<double>();
            var embedding1 = model1.Embed("Determinism test");
            var embedding2 = model2.Embed("Determinism test");
            for (int i = 0; i < embedding1.Length; i++)
                Assert.Equal(embedding1[i], embedding2[i], 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidInputTypes_CreatesInstances()
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

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_AllRowsAreNormalized()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var texts = new List<string> { "First", "Second", "Third" };
            var embeddings = model.EmbedBatch(texts);
            for (int i = 0; i < embeddings.Rows; i++)
            {
                var magnitude = 0.0;
                for (int j = 0; j < embeddings.Columns; j++)
                    magnitude += embeddings[i, j] * embeddings[i, j];
                magnitude = Math.Sqrt(magnitude);
                Assert.Equal(1.0, magnitude, 5);
            }
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithLongText_ReturnsEmbedding()
        {
            if (!HasApiKey) return; // requires COHERE_API_KEY env var
            var model = CreateModel<double>();
            var longText = string.Join(" ", Enumerable.Repeat("word", 500));
            var embedding = model.Embed(longText);
            Assert.NotNull(embedding);
            Assert.Equal(1024, embedding.Length);
        }
    }
}
