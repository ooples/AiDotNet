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
    public class SentenceTransformersFineTunerTests
    {
        private static string GetMissingModelPath() =>
            Path.Combine(Path.GetTempPath(), $"missing-{Guid.NewGuid():N}.onnx");

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );

            // Assert
            Assert.NotNull(model);
            Assert.Equal(384, model.EmbeddingDimension);
            Assert.Equal(512, model.MaxTokens);
        }

        [Fact]
        public void Constructor_WithNullBaseModelPath_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SentenceTransformersFineTuner<double>(null, GetMissingModelPath(), 10, 0.00002, 384));
        }

        [Fact]
        public void Constructor_WithNullOutputModelPath_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SentenceTransformersFineTuner<double>(GetMissingModelPath(), null, 10, 0.00002, 384));
        }

        [Fact]
        public void Constructor_WithZeroEpochs_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SentenceTransformersFineTuner<double>(GetMissingModelPath(), GetMissingModelPath(), 0, 0.00002, 384));
        }

        [Fact]
        public void Constructor_WithNegativeEpochs_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SentenceTransformersFineTuner<double>(GetMissingModelPath(), GetMissingModelPath(), -1, 0.00002, 384));
        }

        [Fact]
        public void Embed_WithMissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange - model file does not exist on disk
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );

            // Act & Assert - should throw because ONNX model file doesn't exist
            Assert.Throws<FileNotFoundException>(() => model.Embed("Test text"));
        }

        [Fact]
        public void Embed_WithNullText_ThrowsArgumentException()
        {
            // Arrange
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(null));
        }

        [Fact]
        public void Embed_WithEmptyText_ThrowsArgumentException()
        {
            // Arrange
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.Embed(string.Empty));
        }

        [Fact]
        public void FineTune_WithMissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange - using a non-existent model path
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                5,
                0.00002,
                384
            );
            var trainingPairs = new List<(string, string, string)>
            {
                ("anchor1", "positive1", "negative1"),
                ("anchor2", "positive2", "negative2"),
                ("anchor3", "positive3", "negative3")
            };

            // Act & Assert - should throw because model file doesn't exist
            Assert.Throws<FileNotFoundException>(() => model.FineTune(trainingPairs));
        }

        [Fact]
        public void FineTune_WithNullTrainingPairs_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.FineTune(null));
        }

        [Fact]
        public void FineTune_WithEmptyTrainingPairs_ThrowsArgumentException()
        {
            // Arrange
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );
            var trainingPairs = new List<(string, string, string)>();

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => model.FineTune(trainingPairs));
            Assert.Contains("Training pairs cannot be empty", exception.Message);
        }

        [Fact]
        public void EmbedBatch_WithNullTexts_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => model.EmbedBatch(null));
        }

        [Fact]
        public void EmbedBatch_WithEmptyCollection_ThrowsArgumentException()
        {
            // Arrange
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );
            var texts = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void EmbedBatch_WithMissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange - model file does not exist on disk
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );
            var texts = new List<string> { "First text", "Second text", "Third text" };

            // Act & Assert - should throw because ONNX model file doesn't exist
            Assert.Throws<FileNotFoundException>(() => model.EmbedBatch(texts));
        }

        [Fact]
        public void Embed_WithFloatType_MissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange - model file does not exist on disk
            var model = new SentenceTransformersFineTuner<float>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002f,
                384
            );

            // Act & Assert - should throw because ONNX model file doesn't exist
            Assert.Throws<FileNotFoundException>(() => model.Embed("Test with float type"));
        }

        [Fact]
        public void MaxTokens_ReturnsCorrectValue()
        {
            // Arrange
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                10,
                0.00002,
                384
            );

            // Act
            var maxTokens = model.MaxTokens;

            // Assert
            Assert.Equal(512, maxTokens);
        }

        [Fact]
        public void FineTune_WithLargeTrainingSet_ThrowsFileNotFoundException()
        {
            // Arrange - using a non-existent model path
            var model = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                2,
                0.00002,
                384
            );
            var trainingPairs = Enumerable.Range(0, 50)
                .Select(i => ($"anchor{i}", $"positive{i}", $"negative{i}"))
                .ToList();

            // Act & Assert - should throw because model file doesn't exist
            Assert.Throws<FileNotFoundException>(() => model.FineTune(trainingPairs));
        }

        [Fact]
        public void FineTune_WithDifferentLearningRates_MissingModelFile_ThrowsFileNotFoundException()
        {
            // Arrange - model files don't exist, both should throw
            var model1 = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                5,
                0.00001,
                384
            );
            var model2 = new SentenceTransformersFineTuner<double>(
                GetMissingModelPath(),
                GetMissingModelPath(),
                5,
                0.0001,
                384
            );
            var trainingPairs = new List<(string, string, string)>
            {
                ("anchor", "positive", "negative")
            };

            // Act & Assert - both should throw because model file doesn't exist
            Assert.Throws<FileNotFoundException>(() => model1.FineTune(trainingPairs));
            Assert.Throws<FileNotFoundException>(() => model2.FineTune(trainingPairs));
        }
    }
}
