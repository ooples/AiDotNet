using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class LowRankFactorizationCompressionTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new LowRankFactorizationCompression<double>(
                targetRank: 10,
                energyThreshold: 0.95,
                maxIterations: 100,
                tolerance: 1e-6);

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithNegativeTargetRank_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new LowRankFactorizationCompression<double>(targetRank: -1));
        }

        [Fact]
        public void Constructor_WithInvalidEnergyThreshold_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new LowRankFactorizationCompression<double>(energyThreshold: 0));
            Assert.Throws<ArgumentException>(() =>
                new LowRankFactorizationCompression<double>(energyThreshold: 1.5));
        }

        [Fact]
        public void Constructor_WithInvalidMaxIterations_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new LowRankFactorizationCompression<double>(maxIterations: 0));
        }

        [Fact]
        public void Compress_WithValidWeights_ReturnsCompressedData()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>(
                targetRank: 0, energyThreshold: 0.95);
            var weights = new Vector<double>(new double[] {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);

            // Assert
            Assert.NotNull(compressedWeights);
            Assert.NotNull(metadata);
            Assert.IsType<LowRankFactorizationMetadata<double>>(metadata);
        }

        [Fact]
        public void Compress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => compression.Compress(null!));
        }

        [Fact]
        public void Compress_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compression.Compress(new Vector<double>(Array.Empty<double>())));
        }

        [Fact]
        public void Compress_ProducesLowRankFactorizationMetadata()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>(
                targetRank: 2, energyThreshold: 0.95);
            var weights = new Vector<double>(new double[] {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
            });

            // Act
            var (_, metadata) = compression.Compress(weights);
            var lrMetadata = (LowRankFactorizationMetadata<double>)metadata;

            // Assert
            Assert.True(lrMetadata.Rows > 0);
            Assert.True(lrMetadata.Cols > 0);
            Assert.True(lrMetadata.Rank > 0);
            Assert.Equal(9, lrMetadata.OriginalLength);
        }

        [Fact]
        public void Decompress_ReconstructsApproximateWeights()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>(
                energyThreshold: 0.99);
            var originalWeights = new Vector<double>(new double[] {
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
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
            var compression = new LowRankFactorizationCompression<double>();
            var metadata = new LowRankFactorizationMetadata<double>(3, 3, 2, 9);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(null!, metadata));
        }

        [Fact]
        public void GetCompressedSize_ReturnsCorrectSize()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>(
                targetRank: 5);
            var weights = new double[100];
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
            var compression = new LowRankFactorizationCompression<float>(
                energyThreshold: 0.95);
            var weights = new Vector<float>(new float[] {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
        }

        [Fact]
        public void Compress_WithSpecificRank_RespectsRank()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>(targetRank: 2);
            var weights = new double[36]; // 6x6 matrix
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = i / 10.0;
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (_, metadata) = compression.Compress(weightsVector);
            var lrMetadata = (LowRankFactorizationMetadata<double>)metadata;

            // Assert
            Assert.True(lrMetadata.Rank <= 2);
        }

        [Fact]
        public void Metadata_GetMetadataSize_ReturnsPositiveValue()
        {
            // Arrange
            var metadata = new LowRankFactorizationMetadata<double>(10, 10, 5, 100);

            // Act
            var size = metadata.GetMetadataSize();

            // Assert
            Assert.True(size > 0);
        }

        [Fact]
        public void Compress_WithLargeWeights_CompletesInReasonableTime()
        {
            // Arrange
            var compression = new LowRankFactorizationCompression<double>(
                targetRank: 10, maxIterations: 50);
            var weights = new double[1000];
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble();
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var startTime = DateTime.Now;
            var (compressedWeights, _) = compression.Compress(weightsVector);
            var elapsed = DateTime.Now - startTime;

            // Assert
            Assert.True(elapsed.TotalSeconds < 10, "Compression took too long");
            Assert.NotNull(compressedWeights);
        }
    }
}
