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
            // Arrange & Act
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            // Assert
            Assert.NotNull(model);
            Assert.Equal(512, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullModelPath_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiModalEmbeddingModel<double>(null, true, 512));
        }

        [Fact]
        public void Embed_WithValidText_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var text = "This is a test sentence.";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(512, embedding.Length);
        }

        [Fact]
        public void Embed_WithNormalization_ReturnsNormalizedVector()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
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
        public void Embed_WithoutNormalization_ReturnsUnnormalizedVector()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", false, 512);
            var text = "Test no normalization";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            var magnitude = 0.0;
            for (int i = 0; i < embedding.Length; i++)
            {
                magnitude += embedding[i] * embedding[i];
            }
            magnitude = Math.Sqrt(magnitude);
            // Should not be normalized to 1.0 when normalization is false
            // We can't assert exact value, but we verify it's generated
            Assert.True(magnitude > 0);
        }

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void EmbedImage_WithValidImagePath_ReturnsVectorOfCorrectDimension()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                // Act
                var embedding = model.EmbedImage(tempImageFile);

                // Assert
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
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => model.EmbedImage(null));
            Assert.Contains("Image path cannot be null or whitespace", exception.Message);
        }

        [Fact]
        public void EmbedImage_WithEmptyImagePath_ThrowsArgumentException()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => model.EmbedImage(string.Empty));
            Assert.Contains("Image path cannot be null or whitespace", exception.Message);
        }

        [Fact]
        public void EmbedImage_WithNonExistentImagePath_ThrowsFileNotFoundException()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var nonExistentPath = Path.Combine("non", "existent", "image.jpg");

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => model.EmbedImage(nonExistentPath));
        }

        [Fact]
        public void EmbedImage_WithNormalization_ReturnsNormalizedVector()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                // Act
                var embedding = model.EmbedImage(tempImageFile);

                // Assert
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
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile = CreateTempImageFile();

            try
            {
                // Act
                var embedding1 = model.EmbedImage(tempImageFile);
                var embedding2 = model.EmbedImage(tempImageFile);

                // Assert
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
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var tempImageFile1 = CreateTempImageFile();
            var tempImageFile2 = CreateTempImageFile();
            var tempImageFile3 = CreateTempImageFile();

            try
            {
                var imagePaths = new List<string> { tempImageFile1, tempImageFile2, tempImageFile3 };

                // Act
                var embeddings = model.EmbedImageBatch(imagePaths).ToList();

                // Assert
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
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedImageBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithValidTexts_ReturnsMatrixOfCorrectDimensions()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var texts = new List<string> { "First text", "Second text", "Third text" };

            // Act
            var embeddings = model.EmbedBatch(texts);

            // Assert
            Assert.NotNull(embeddings);
            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(512, embeddings.Columns);
        }

        [Fact]
        public void Embed_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<float>("test-model-path.onnx", true, 512);
            var text = "Test with float type";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(512, embedding.Length);

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
            var customDimension = 768;
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, customDimension);
            var text = "Testing custom dimension";

            // Act
            var embedding = model.Embed(text);

            // Assert
            Assert.Equal(customDimension, embedding.Length);
        }

        [Fact]
        public void EmbedImage_WithCustomDimension_ReturnsCorrectSize()
        {
            // Arrange
            var customDimension = 768;
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, customDimension);
            var tempImageFile = CreateTempImageFile();

            try
            {
                // Act
                var embedding = model.EmbedImage(tempImageFile);

                // Assert
                Assert.Equal(customDimension, embedding.Length);
            }
            finally
            {
                if (File.Exists(tempImageFile))
                    File.Delete(tempImageFile);
            }
        }

        [Fact]
        public void Embed_Deterministic_MultipleInstances()
        {
            // Arrange
            var model1 = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
            var model2 = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);
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
        public void MaxTokens_ReturnsCorrectValue()
        {
            // Arrange
            var model = new MultiModalEmbeddingModel<double>("test-model-path.onnx", true, 512);

            // Act
            var maxTokens = model.MaxTokens;

            // Assert
            Assert.Equal(512, maxTokens);
        }
    }
}
