using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class SparsePruningCompressionTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new SparsePruningCompression<double>(
                sparsityTarget: 0.9,
                minMagnitudeThreshold: 0,
                useGlobalThreshold: true);

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithInvalidSparsityTarget_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new SparsePruningCompression<double>(sparsityTarget: -0.1));
            Assert.Throws<ArgumentException>(() =>
                new SparsePruningCompression<double>(sparsityTarget: 1.5));
        }

        [Fact]
        public void Constructor_WithNegativeMagnitudeThreshold_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new SparsePruningCompression<double>(minMagnitudeThreshold: -0.1));
        }

        [Fact]
        public void Compress_WithValidWeights_ReturnsCompressedData()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>(sparsityTarget: 0.5);
            var weights = new Vector<double>(new double[] {
                0.001, 0.5, -0.002, 0.8, 0.003, -0.7, 0.001, 0.9
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);

            // Assert
            Assert.NotNull(compressedWeights);
            Assert.NotNull(metadata);
            Assert.IsType<SparsePruningMetadata<double>>(metadata);
        }

        [Fact]
        public void Compress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => compression.Compress(null!));
        }

        [Fact]
        public void Compress_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compression.Compress(new Vector<double>(Array.Empty<double>())));
        }

        [Fact]
        public void Compress_PrunesSmallMagnitudeWeights()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>(sparsityTarget: 0.5);
            var weights = new Vector<double>(new double[] {
                0.001, 10.0, 0.002, 20.0, 0.003, 30.0, 0.004, 40.0
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var sparseMetadata = (SparsePruningMetadata<double>)metadata;

            // Assert
            // Should keep only the large values
            Assert.True(compressedWeights.Length <= weights.Length);
            Assert.True(sparseMetadata.ActualSparsity >= 0.4); // At least 40% pruned
        }

        [Fact]
        public void Decompress_ReconstructsWeightsWithZeros()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>(sparsityTarget: 0.5);
            var originalWeights = new Vector<double>(new double[] {
                0.001, 10.0, 0.002, 20.0, 0.003, 30.0, 0.004, 40.0
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(originalWeights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(originalWeights.Length, decompressedWeights.Length);

            // Large values should be preserved, small values should be zero
            Assert.Equal(10.0, decompressedWeights[1]);
            Assert.Equal(20.0, decompressedWeights[3]);
            Assert.Equal(30.0, decompressedWeights[5]);
            Assert.Equal(40.0, decompressedWeights[7]);
        }

        [Fact]
        public void Decompress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>();
            var metadata = new SparsePruningMetadata<double>(
                new int[] { 0, 1 }, 10, 0.1, 0.8);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(null!, metadata));
        }

        [Fact]
        public void GetCompressedSize_ReturnsCorrectSize()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>(sparsityTarget: 0.9);
            var weights = new double[100];
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble() * 10.0;
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (compressedWeights, metadata) = compression.Compress(weightsVector);
            var compressedSize = compression.GetCompressedSize(compressedWeights, metadata);
            var sparseMetadata = (SparsePruningMetadata<double>)metadata;

            // Assert
            Assert.True(compressedSize > 0);
            Assert.True(sparseMetadata.ActualSparsity > 0.8); // Close to 90% sparsity target
        }

        [Fact]
        public void Compress_WithExplicitThreshold_UsesThreshold()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>(
                minMagnitudeThreshold: 0.5);
            var weights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var sparseMetadata = (SparsePruningMetadata<double>)metadata;

            // Assert
            // Should keep only weights >= 0.5
            Assert.Equal(0.5, sparseMetadata.Threshold);
            Assert.Equal(4, compressedWeights.Length); // 0.6, 0.7, 0.8, 0.9
        }

        [Fact]
        public void Compress_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var compression = new SparsePruningCompression<float>(sparsityTarget: 0.5);
            var weights = new Vector<float>(new float[] {
                0.001f, 10.0f, 0.002f, 20.0f
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
        }

        [Fact]
        public void Metadata_GetMetadataSize_ReturnsPositiveValue()
        {
            // Arrange
            var metadata = new SparsePruningMetadata<double>(
                new int[] { 1, 3, 5, 7 },
                8,
                0.5,
                0.5);

            // Act
            var size = metadata.GetMetadataSize();

            // Assert
            Assert.True(size > 0);
        }

        [Fact]
        public void Compress_HighSparsityTarget_PrunesMostWeights()
        {
            // Arrange
            var compression = new SparsePruningCompression<double>(sparsityTarget: 0.95);
            var weights = new double[100];
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble();
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (compressedWeights, metadata) = compression.Compress(weightsVector);
            var sparseMetadata = (SparsePruningMetadata<double>)metadata;

            // Assert
            Assert.True(compressedWeights.Length <= 10); // ~5% of 100
            Assert.True(sparseMetadata.ActualSparsity >= 0.9);
        }
    }
}
