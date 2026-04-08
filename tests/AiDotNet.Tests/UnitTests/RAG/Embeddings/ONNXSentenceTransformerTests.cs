#nullable disable
using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class ONNXSentenceTransformerTests
    {
        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Assert
            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullModelPath_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>(null, 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyModelPath_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithWhitespaceModelPath_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("   ", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", 0, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", -1, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroMaxTokens_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 0));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeMaxTokens_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            var exception = Assert.Throws<ArgumentException>(() =>
                new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, -1));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithMissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => model.Embed("This is a test sentence."));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithNullText_ThrowsArgumentException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithMissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", 384, 512);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx");
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithFloatType_MissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange
            var model = new ONNXSentenceTransformer<float>("test-model-path.onnx", 384, 512);

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => model.Embed("Test with float type"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithCustomDimension_MissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange
            var customDimension = 768;
            var model = new ONNXSentenceTransformer<double>("test-model-path.onnx", customDimension, 512);

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => model.Embed("Testing custom dimension"));
        }
    }
}
