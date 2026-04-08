using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class WeightClusteringCompressionTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new WeightClusteringCompression<double>(
                numClusters: 256,
                maxIterations: 100,
                tolerance: 1e-6,
                randomSeed: 42);

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithInvalidNumClusters_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new WeightClusteringCompression<double>(numClusters: 0));
            Assert.Throws<ArgumentException>(() =>
                new WeightClusteringCompression<double>(numClusters: -1));
        }

        [Fact]
        public void Constructor_WithInvalidMaxIterations_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new WeightClusteringCompression<double>(maxIterations: 0));
            Assert.Throws<ArgumentException>(() =>
                new WeightClusteringCompression<double>(maxIterations: -1));
        }

        [Fact]
        public void Compress_WithValidWeights_ReturnsCompressedData()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<double>(new double[] { 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);

            // Assert
            Assert.NotNull(compressedWeights);
            Assert.NotNull(metadata);
            Assert.Equal(weights.Length, compressedWeights.Length);
            Assert.IsType<WeightClusteringMetadata<double>>(metadata);
        }

        [Fact]
        public void Compress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => compression.Compress(null!));
        }

        [Fact]
        public void Compress_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => compression.Compress(new Vector<double>(Array.Empty<double>())));
        }

        [Fact]
        public void Compress_ProducesClusteringMetadata()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>(
                numClusters: 3, randomSeed: 42);
            var weights = new Vector<double>(new double[] { 1.0, 1.0, 5.0, 5.0, 10.0, 10.0 });

            // Act
            var (_, metadata) = compression.Compress(weights);
            var clusterMetadata = (WeightClusteringMetadata<double>)metadata;

            // Assert
            Assert.Equal(3, clusterMetadata.NumClusters);
            Assert.Equal(3, clusterMetadata.ClusterCenters.Length);
            Assert.Equal(6, clusterMetadata.OriginalLength);
        }

        [Fact]
        public void Decompress_ReconstructsApproximateWeights()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var originalWeights = new Vector<double>(new double[] { 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(originalWeights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(originalWeights.Length, decompressedWeights.Length);
            // Verify weights are approximately reconstructed
            for (int i = 0; i < originalWeights.Length; i++)
            {
                Assert.True(Math.Abs(originalWeights[i] - decompressedWeights[i]) < 0.5,
                    $"Weight {i}: original={originalWeights[i]}, decompressed={decompressedWeights[i]}");
            }
        }

        [Fact]
        public void Decompress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>();
            var metadata = new WeightClusteringMetadata<double>(
                new double[] { 1.0, 2.0 },
                2,
                10);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(null!, metadata));
        }

        [Fact]
        public void GetCompressedSize_ReturnsReasonableSize()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>(
                numClusters: 256, randomSeed: 42);
            var weights = new double[1000];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = i / 100.0;
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (compressedWeights, metadata) = compression.Compress(weightsVector);
            var compressedSize = compression.GetCompressedSize(compressedWeights, metadata);
            var originalSize = weights.Length * sizeof(double);

            // Assert
            Assert.True(compressedSize > 0);
            // Compressed size should be significantly smaller than original
            Assert.True(compressedSize < originalSize);
        }

        [Fact]
        public void CalculateCompressionRatio_WithValidSizes_ReturnsCorrectRatio()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>();
            long originalSize = 1000;
            long compressedSize = 100;

            // Act
            var ratio = compression.CalculateCompressionRatio(originalSize, compressedSize);

            // Assert
            Assert.Equal(10.0, ratio);
        }

        [Fact]
        public void CalculateCompressionRatio_WithZeroCompressedSize_ThrowsException()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compression.CalculateCompressionRatio(1000, 0));
        }

        [Fact]
        public void Compress_WithFewerWeightsThanClusters_AdjustsClusters()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>(
                numClusters: 100, randomSeed: 42);
            var weights = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var (_, metadata) = compression.Compress(weights);
            var clusterMetadata = (WeightClusteringMetadata<double>)metadata;

            // Assert
            Assert.True(clusterMetadata.NumClusters <= weights.Length);
        }

        [Fact]
        public void Compress_WithIdenticalWeights_CreatesOneCluster()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>(
                numClusters: 10, randomSeed: 42);
            var weights = new Vector<double>(new double[] { 5.0, 5.0, 5.0, 5.0, 5.0 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            // All decompressed weights should be approximately 5.0
            foreach (var weight in decompressedWeights)
            {
                Assert.True(Math.Abs(weight - 5.0) < 0.001);
            }
        }

        [Fact]
        public void Compress_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var compression = new WeightClusteringCompression<float>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<float>(new float[] { 1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                Assert.True(Math.Abs(weights[i] - decompressedWeights[i]) < 0.5f);
            }
        }

        [Fact]
        public void Compress_WithReproducibleSeed_ProducesSameResults()
        {
            // Arrange
            var compression1 = new WeightClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var compression2 = new WeightClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

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
        public void Compress_AchievesReasonableCompressionRatio()
        {
            // Arrange
            var compression = new WeightClusteringCompression<double>(
                numClusters: 256, randomSeed: 42);
            var weights = new double[10000];
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble() * 10.0;
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (compressedWeights, metadata) = compression.Compress(weightsVector);
            var originalSize = weights.Length * sizeof(double);
            var compressedSize = compression.GetCompressedSize(compressedWeights, metadata);
            var ratio = compression.CalculateCompressionRatio(originalSize, compressedSize);

            // Assert
            // Should achieve reasonable compression with 256 clusters
            // The theoretical max is ~2x but practical results may be slightly lower
            Assert.True(ratio > 1.8, $"Compression ratio {ratio:F2} is too low");
        }
    }
}
