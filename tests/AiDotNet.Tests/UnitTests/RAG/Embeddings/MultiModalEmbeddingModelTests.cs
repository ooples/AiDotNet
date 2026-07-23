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
    public class MultiModalEmbeddingModelTests
    {
        private string CreateTempImageFile()
        {
            var tempFile = Path.GetTempFileName();
            File.WriteAllText(tempFile, "fake image content");
            return tempFile;
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            Assert.NotNull(model);
            Assert.Equal(512, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullModelPath_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new MultiModalEmbeddingModel<double>(null, true, 512));
        }

        // ────────── Text path: missing ONNX model throws (no fake vector) ──────────

        [Fact]
        public void Embed_WithValidText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            Assert.Throws<FileNotFoundException>(() => model.Embed("This is a test sentence."));
        }

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var texts = new List<string> { "First text", "Second text", "Third text" };
            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void MaxTokens_ReturnsCorrectValue()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            Assert.Equal(512, model.MaxTokens);
        }

        // ────────── Image path: no fake hash-based vector — throws clearly ──────────

        [Fact]
        public void EmbedImage_WithValidImagePath_ThrowsNotSupported()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                var ex = Assert.Throws<NotSupportedException>(() => model.EmbedImage(tempImageFile));
                Assert.Contains("Image embedding is not supported", ex.Message);
            }
            finally
            {
                if (File.Exists(tempImageFile))
                    File.Delete(tempImageFile);
            }
        }

        [Fact]
        public void EmbedImage_WithNullImagePath_ThrowsArgumentException()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var exception = Assert.Throws<ArgumentException>(() => model.EmbedImage(null));
            Assert.Contains("Image path cannot be null or whitespace", exception.Message);
        }

        [Fact]
        public void EmbedImage_WithEmptyImagePath_ThrowsArgumentException()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var exception = Assert.Throws<ArgumentException>(() => model.EmbedImage(string.Empty));
            Assert.Contains("Image path cannot be null or whitespace", exception.Message);
        }

        [Fact]
        public void EmbedImageBatch_WhenEnumerated_ThrowsNotSupported()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                var imagePaths = new List<string> { tempImageFile };
                Assert.Throws<NotSupportedException>(() => model.EmbedImageBatch(imagePaths).ToList());
            }
            finally
            {
                if (File.Exists(tempImageFile))
                    File.Delete(tempImageFile);
            }
        }

        [Fact]
        public void EmbedImageBatch_WithNullImagePaths_ThrowsArgumentNullException()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            Assert.Throws<ArgumentNullException>(() => model.EmbedImageBatch(null));
        }
    }
}
