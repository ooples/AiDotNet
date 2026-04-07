#nullable disable
using System;
using System.Collections.Generic;
using System.IO;
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
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.NotNull(model);
            Assert.Equal(1024, model.EmbeddingDimension);
            Assert.Equal(16000, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>(null, "voyage-model-path.onnx", "document", 1024));
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", null, "document", 1024));
        }

        [Fact]
        public void Constructor_WithNullInputType_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", null, 1024));
        }

        [Fact]
        public void Embed_WithValidText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("This is a test sentence."));
        }

        [Fact]
        public void Embed_WithSameTextTwice_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact]
        public void Embed_WithDifferentTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact]
        public void Embed_ReturnsNormalizedVector_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test normalization"));
        }

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string>();

            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void Embed_WithFloatType_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<float>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test with float type"));
        }

        [Fact]
        public void Embed_WithCustomDimension_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Testing custom dimension"));
        }

        [Fact]
        public void Embed_Deterministic_MultipleInstances_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Determinism test"));
        }

        [Fact]
        public void EmbedBatch_AllRowsAreNormalized_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void Embed_WithLongText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var longText = string.Join(" ", Enumerable.Repeat("word", 2000));

            Assert.Throws<FileNotFoundException>(() => model.Embed(longText));
        }

        [Fact]
        public void Constructor_WithDifferentInputTypes_CreatesInstances()
        {
            var documentModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);
            var queryModel = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "query", 1024);

            Assert.NotNull(documentModel);
            Assert.NotNull(queryModel);
        }

        [Fact]
        public void MaxTokens_ReturnsCorrectValue()
        {
            var model = new VoyageAIEmbeddingModel<double>("test-api-key", "voyage-model-path.onnx", "document", 1024);

            Assert.Equal(16000, model.MaxTokens);
        }
    }
}
