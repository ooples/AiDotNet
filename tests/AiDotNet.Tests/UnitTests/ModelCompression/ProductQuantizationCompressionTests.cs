using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class ProductQuantizationCompressionTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new ProductQuantizationCompression<double>(
                numSubvectors: 8,
                numCentroids: 256,
                maxIterations: 100,
                tolerance: 1e-6,
                randomSeed: 42);

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithInvalidNumSubvectors_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new ProductQuantizationCompression<double>(numSubvectors: 0));
            Assert.Throws<ArgumentException>(() =>
                new ProductQuantizationCompression<double>(numSubvectors: -1));
        }

        [Fact]
        public void Constructor_WithInvalidNumCentroids_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new ProductQuantizationCompression<double>(numCentroids: 0));
            Assert.Throws<ArgumentException>(() =>
                new ProductQuantizationCompression<double>(numCentroids: -1));
        }

        [Fact]
        public void Compress_WithValidWeights_ReturnsCompressedData()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<double>(
                numSubvectors: 4, numCentroids: 16, randomSeed: 42);
            var weights = new Vector<double>(new double[] {
                1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1,
                5.0, 5.1, 6.0, 6.1, 7.0, 7.1, 8.0, 8.1
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);

            // Assert
            Assert.NotNull(compressedWeights);
            Assert.NotNull(metadata);
            Assert.IsType<ProductQuantizationMetadata<double>>(metadata);
        }

        [Fact]
        public void Compress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => compression.Compress(null!));
        }

        [Fact]
        public void Compress_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compression.Compress(new Vector<double>(Array.Empty<double>())));
        }

        [Fact]
        public void Compress_ProducesProductQuantizationMetadata()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<double>(
                numSubvectors: 2, numCentroids: 8, randomSeed: 42);
            var weights = new Vector<double>(new double[] {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            });

            // Act
            var (_, metadata) = compression.Compress(weights);
            var pqMetadata = (ProductQuantizationMetadata<double>)metadata;

            // Assert
            Assert.Equal(2, pqMetadata.NumSubvectors);
            Assert.NotNull(pqMetadata.Codebooks);
            Assert.Equal(8, pqMetadata.OriginalLength);
        }

        [Fact]
        public void Decompress_ReconstructsApproximateWeights()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<double>(
                numSubvectors: 2, numCentroids: 16, randomSeed: 42);
            var originalWeights = new Vector<double>(new double[] {
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(originalWeights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(originalWeights.Length, decompressedWeights.Length);
        }

        [Fact]
        public void Decompress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<double>();
            var metadata = new ProductQuantizationMetadata<double>(
                new double[][] { new double[] { 1.0, 2.0 } },
                2, 1, 4, 4);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(null!, metadata));
        }

        [Fact]
        public void GetCompressedSize_ReturnsPositiveSize()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<double>(
                numSubvectors: 4, numCentroids: 16, randomSeed: 42);
            var weights = new double[64];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = i / 10.0;
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (compressedWeights, metadata) = compression.Compress(weightsVector);
            var compressedSize = compression.GetCompressedSize(compressedWeights, metadata);

            // Assert
            Assert.True(compressedSize > 0);
        }

        [Fact]
        public void Compress_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var compression = new ProductQuantizationCompression<float>(
                numSubvectors: 2, numCentroids: 8, randomSeed: 42);
            var weights = new Vector<float>(new float[] {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
        }

        [Fact]
        public void Compress_WithReproducibleSeed_ProducesSameResults()
        {
            // Arrange
            var compression1 = new ProductQuantizationCompression<double>(
                numSubvectors: 2, numCentroids: 8, randomSeed: 42);
            var compression2 = new ProductQuantizationCompression<double>(
                numSubvectors: 2, numCentroids: 8, randomSeed: 42);
            var weights = new Vector<double>(new double[] {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            });

            // Act
            var (compressed1, _) = compression1.Compress(weights);
            var (compressed2, _) = compression2.Compress(weights);

            // Assert
            Assert.Equal(compressed1.Length, compressed2.Length);
            for (int i = 0; i < compressed1.Length; i++)
            {
                Assert.Equal(compressed1[i], compressed2[i]);
            }
        }

        [Fact]
        public void Metadata_GetMetadataSize_ReturnsPositiveValue()
        {
            // Arrange
            var codebooks = new double[][] {
                new double[] { 1.0, 2.0, 3.0, 4.0 },
                new double[] { 5.0, 6.0, 7.0, 8.0 }
            };
            var metadata = new ProductQuantizationMetadata<double>(
                codebooks, 2, 2, 4, 8);

            // Act
            var size = metadata.GetMetadataSize();

            // Assert
            Assert.True(size > 0);
        }
    }
}
