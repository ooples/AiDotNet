using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class HybridHuffmanClusteringCompressionTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new HybridHuffmanClusteringCompression<double>();

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithCustomParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 32,
                maxIterations: 50,
                tolerance: 1e-5,
                huffmanPrecision: 4,
                randomSeed: 42);

            // Assert
            Assert.NotNull(compression);
        }

        #endregion

        #region Compress Tests

        [Fact]
        public void Compress_WithValidWeights_ReturnsCompressedData()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);

            // Assert
            Assert.NotNull(compressedWeights);
            Assert.NotNull(metadata);
        }

        [Fact]
        public void Compress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => compression.Compress(null!));
        }

        [Fact]
        public void Compress_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>();
            var weights = new Vector<double>(Array.Empty<double>());

            // Act & Assert
            Assert.Throws<ArgumentException>(() => compression.Compress(weights));
        }

        [Fact]
        public void Compress_ProducesHybridMetadata()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
            });

            // Act
            var (_, metadata) = compression.Compress(weights);

            // Assert
            Assert.IsType<HybridCompressionMetadata<double>>(metadata);
            var hybridMetadata = (HybridCompressionMetadata<double>)metadata;
            Assert.NotNull(hybridMetadata.ClusteringMetadata);
            Assert.NotNull(hybridMetadata.HuffmanMetadata);
        }

        #endregion

        #region Decompress Tests

        [Fact]
        public void Decompress_ReconstructsWeights()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 8, randomSeed: 42);
            var originalWeights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
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
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 });
            var (_, metadata) = compression.Compress(weights);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(null!, metadata));
        }

        [Fact]
        public void Decompress_WithNullMetadata_ThrowsException()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>();
            var weights = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(weights, null!));
        }

        #endregion

        #region GetCompressedSize Tests

        [Fact]
        public void GetCompressedSize_ReturnsPositiveSize()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<double>(new double[] {
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var compressedSize = compression.GetCompressedSize(compressedWeights, metadata);

            // Assert
            Assert.True(compressedSize > 0);
        }

        [Fact]
        public void GetCompressedSize_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4 });
            var (_, metadata) = compression.Compress(weights);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.GetCompressedSize((Vector<double>)null!, metadata));
        }

        [Fact]
        public void GetCompressedSize_WithNullMetadata_ThrowsException()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>();
            var weights = new Vector<double>(new double[] { 1.0, 2.0 });

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.GetCompressedSize(weights, null!));
        }

        #endregion

        #region Round-Trip Tests

        [Fact]
        public void CompressAndDecompress_RoundTrip_PreservesApproximateValues()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<double>(
                numClusters: 16, randomSeed: 42);
            var originalWeights = new Vector<double>(new double[] {
                0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(originalWeights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(originalWeights.Length, decompressedWeights.Length);

            // Values should be approximately preserved (within clustering tolerance)
            for (int i = 0; i < originalWeights.Length; i++)
            {
                Assert.True(Math.Abs(originalWeights[i] - decompressedWeights[i]) < 0.2,
                    $"Weight at index {i} differs too much: original={originalWeights[i]}, decompressed={decompressedWeights[i]}");
            }
        }

        #endregion

        #region Type-Specific Tests

        [Fact]
        public void Compress_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var compression = new HybridHuffmanClusteringCompression<float>(
                numClusters: 4, randomSeed: 42);
            var weights = new Vector<float>(new float[] {
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
        }

        #endregion

        #region Metadata Tests

        [Fact]
        public void HybridCompressionMetadata_Generic_Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5, 1.5 }, 2, 4);
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(0.5, 1, true, 0, null, null),
                new NumericDictionary<double, string>(), 4, 8);

            // Act
            var metadata = new HybridCompressionMetadata<double>(
                clusteringMetadata, huffmanMetadata, 10);

            // Assert
            Assert.NotNull(metadata);
            Assert.Equal(CompressionType.HybridHuffmanClustering, metadata.Type);
            Assert.Equal(10, metadata.OriginalLength);
        }

        [Fact]
        public void HybridCompressionMetadata_Generic_WithNullClustering_ThrowsException()
        {
            // Arrange
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(0.5, 1, true, 0, null, null),
                new NumericDictionary<double, string>(), 4, 8);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridCompressionMetadata<double>(null!, huffmanMetadata, 10));
        }

        [Fact]
        public void HybridCompressionMetadata_Generic_WithNullHuffman_ThrowsException()
        {
            // Arrange
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5 }, 1, 4);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridCompressionMetadata<double>(clusteringMetadata, null!, 10));
        }

        [Fact]
        public void HybridCompressionMetadata_Generic_GetMetadataSize_ReturnsPositiveValue()
        {
            // Arrange
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5, 1.5 }, 2, 4);
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(0.5, 1, true, 0, null, null),
                new NumericDictionary<double, string>(), 4, 8);
            var metadata = new HybridCompressionMetadata<double>(
                clusteringMetadata, huffmanMetadata, 10);

            // Act
            var size = metadata.GetMetadataSize();

            // Assert
            Assert.True(size > 0);
        }

#pragma warning disable CS0618
        [Fact]
        public void HybridCompressionMetadata_Legacy_Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var metadata = new HybridCompressionMetadata(
                new object(), new object());

            // Assert
            Assert.NotNull(metadata);
            Assert.NotNull(metadata.ClusteringMetadata);
            Assert.NotNull(metadata.HuffmanMetadata);
        }

        [Fact]
        public void HybridCompressionMetadata_Legacy_WithNullClustering_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridCompressionMetadata(null!, new object()));
        }

        [Fact]
        public void HybridCompressionMetadata_Legacy_WithNullHuffman_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new HybridCompressionMetadata(new object(), null!));
        }
#pragma warning restore CS0618

        #endregion
    }
}
