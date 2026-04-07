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

        [Fact]
        public void Embed_WithValidText_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("This is a test sentence."));
        }

        [Fact]
        public void Embed_WithNormalization_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test normalization"));
        }

        [Fact]
        public void Embed_WithoutNormalization_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", false, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test no normalization"));
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
        public void EmbedImage_WithValidImagePath_ReturnsVectorOfCorrectDimension()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                var embedding = model.EmbedImage(tempImageFile);

                Assert.NotNull(embedding);
                Assert.Equal(512, embedding.Length);
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
        public void EmbedImage_WithNonExistentImagePath_ThrowsFileNotFoundException()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var nonExistentPath = Path.Combine("non", "existent", "image.jpg");

            Assert.Throws<FileNotFoundException>(() => model.EmbedImage(nonExistentPath));
        }

        [Fact]
        public void EmbedImage_WithNormalization_ReturnsNormalizedVector()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                var embedding = model.EmbedImage(tempImageFile);

                var magnitude = 0.0;
                for (int i = 0; i < embedding.Length; i++)
                {
                    magnitude += embedding[i] * embedding[i];
                }
                magnitude = Math.Sqrt(magnitude);
                Assert.Equal(1.0, magnitude, 5);
            }
            finally
            {
                if (File.Exists(tempImageFile))
                    File.Delete(tempImageFile);
            }
        }

        [Fact]
        public void EmbedImage_WithSameImageTwice_ReturnsSameEmbedding()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                var embedding1 = model.EmbedImage(tempImageFile);
                var embedding2 = model.EmbedImage(tempImageFile);

                for (int i = 0; i < embedding1.Length; i++)
                {
                    Assert.Equal(embedding1[i], embedding2[i], 10);
                }
            }
            finally
            {
                if (File.Exists(tempImageFile))
                    File.Delete(tempImageFile);
            }
        }

        [Fact]
        public void EmbedImageBatch_WithValidImagePaths_ReturnsCorrectNumberOfEmbeddings()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile1 = CreateTempImageFile();
            var tempImageFile2 = CreateTempImageFile();
            var tempImageFile3 = CreateTempImageFile();

            try
            {
                var imagePaths = new List<string> { tempImageFile1, tempImageFile2, tempImageFile3 };

                var embeddings = model.EmbedImageBatch(imagePaths).ToList();

                Assert.Equal(3, embeddings.Count);
                foreach (var embedding in embeddings)
                {
                    Assert.Equal(512, embedding.Length);
                }
            }
            finally
            {
                if (File.Exists(tempImageFile1)) File.Delete(tempImageFile1);
                if (File.Exists(tempImageFile2)) File.Delete(tempImageFile2);
                if (File.Exists(tempImageFile3)) File.Delete(tempImageFile3);
            }
        }

        [Fact]
        public void EmbedImageBatch_WithNullImagePaths_ThrowsArgumentNullException()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            Assert.Throws<ArgumentNullException>(() => model.EmbedImageBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void Embed_WithFloatType_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<float>("test-model-path.onnx", true, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Test with float type"));
        }

        [Fact]
        public void Embed_WithCustomDimension_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 768);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Testing custom dimension"));
        }

        [Fact]
        public void EmbedImage_WithCustomDimension_ReturnsCorrectSize()
        {
            var customDimension = 768;
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, customDimension);
            var tempImageFile = CreateTempImageFile();

            try
            {
                var embedding = model.EmbedImage(tempImageFile);

                Assert.Equal(customDimension, embedding.Length);
            }
            finally
            {
                if (File.Exists(tempImageFile))
                    File.Delete(tempImageFile);
            }
        }

        [Fact]
        public void Embed_Deterministic_MultipleInstances_ThrowsFileNotFoundForMissingModel()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            Assert.Throws<FileNotFoundException>(() => model.Embed("Determinism test"));
        }

        [Fact]
        public void MaxTokens_ReturnsCorrectValue()
        {
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            Assert.Equal(512, model.MaxTokens);
        }
    }
}
