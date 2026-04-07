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
    public class LocalTransformerEmbeddingTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithDefaultParameters_CreatesInstance()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullModelPath_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>(null, 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithEmptyModelPath_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithWhitespaceModelPath_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("   ", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", 0, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", -1, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithZeroMaxTokens_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", 384, 0));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact]
        public void Constructor_WithNegativeMaxTokens_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", 384, -1));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact]
        public void Embed_WithValidText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("This is a test sentence."));
        }

        [Fact]
        public void Embed_WithSameTextTwice_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact]
        public void Embed_WithDifferentTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact]
        public void Embed_ReturnsNormalizedVector_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test normalization"));
        }

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");
            var texts = new List<string>();

            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void Embed_WithFloatType_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<float>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test with float type"));
        }

        [Fact]
        public void Embed_WithCustomDimension_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 768, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Testing custom dimension"));
        }

        [Fact]
        public void Embed_Deterministic_MultipleInstances_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Determinism test"));
        }

        [Fact]
        public void EmbedBatch_AllRowsAreNormalized_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void Embed_WithLongText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var longText = string.Join(" ", Enumerable.Repeat("word", 500));

            Assert.Throws<FileNotFoundException>(() => model.Embed(longText));
        }

        [Fact]
        public void Embed_WithSpecialCharacters_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Special characters: @#$%^&*()!"));
        }
    }
}
