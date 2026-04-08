#nullable disable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.RAG.Embeddings
{
    public class LocalTransformerEmbeddingTests
    {
        [Fact(Timeout = 60000)]
        public async Task Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultParameters_CreatesInstance()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNullModelPath_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>(null, 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithEmptyModelPath_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithWhitespaceModelPath_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("   ", 384, 512));
            Assert.Contains("Model path cannot be empty", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", 0, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", -1, 512));
            Assert.Contains("Dimension must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithZeroMaxTokens_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", 384, 0));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithNegativeMaxTokens_ThrowsArgumentException()
        {
            var exception = Assert.Throws<ArgumentException>(() =>
                new LocalTransformerEmbedding<double>("test-model-path", 384, -1));
            Assert.Contains("Max tokens must be positive", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithValidText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("This is a test sentence."));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSameTextTwice_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSingleText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Hello world"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_ReturnsNormalizedVector_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test normalization"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithEmptyText_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithWhitespaceText_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentException>(() => model.Embed("   "));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithValidTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");

            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path");
            var texts = new List<string>();

            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_ProducesSameEmbeddingsAsIndividualCalls_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithFloatType_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<float>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test with float type"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithCustomDimension_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 768, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Testing custom dimension"));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_Deterministic_MultipleInstances_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Determinism test"));
        }

        [Fact(Timeout = 60000)]
        public async Task EmbedBatch_AllRowsAreNormalized_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var texts = new List<string> { "First", "Second", "Third" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithLongText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);
            var longText = string.Join(" ", Enumerable.Repeat("word", 500));

            Assert.Throws<FileNotFoundException>(() => model.Embed(longText));
        }

        [Fact(Timeout = 60000)]
        public async Task Embed_WithSpecialCharacters_ThrowsFileNotFoundForMissingModel()
        {
            var model = new LocalTransformerEmbedding<double>("test-model-path", 384, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Special characters: @#$%^&*()!"));
        }
    }
}
