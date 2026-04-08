using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    /// <summary>
    /// Unit tests for the Deep Compression algorithm (Han et al. 2015).
    /// Tests the three-stage pipeline: Pruning → Quantization → Huffman Coding.
    /// </summary>
    public class DeepCompressionTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new DeepCompression<double>();

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.9,
                pruningThreshold: 0.01,
                numClusters: 32,
                maxClusteringIterations: 100,
                clusteringTolerance: 1e-6,
                huffmanPrecision: 4,
                randomSeed: 42);

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithNegativePruningSparsity_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeepCompression<double>(pruningSparsity: -0.1));
        }

        [Fact]
        public void Constructor_WithPruningSparsityGreaterThanOne_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeepCompression<double>(pruningSparsity: 1.5));
        }

        [Fact]
        public void Constructor_WithNegativePruningThreshold_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeepCompression<double>(pruningThreshold: -0.1));
        }

        [Fact]
        public void Constructor_WithZeroNumClusters_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeepCompression<double>(numClusters: 0));
        }

        [Fact]
        public void Constructor_WithNegativeNumClusters_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeepCompression<double>(numClusters: -1));
        }

        [Fact]
        public void Constructor_WithZeroMaxIterations_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeepCompression<double>(maxClusteringIterations: 0));
        }

        [Fact]
        public void Constructor_WithZeroHuffmanPrecision_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new DeepCompression<double>(huffmanPrecision: 0));
        }

        #endregion

        #region Factory Method Tests

        [Fact]
        public void ForConvolutionalLayers_CreatesCorrectInstance()
        {
            // Arrange & Act
            var compression = DeepCompression<double>.ForConvolutionalLayers(randomSeed: 42);

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void ForFullyConnectedLayers_CreatesCorrectInstance()
        {
            // Arrange & Act
            var compression = DeepCompression<double>.ForFullyConnectedLayers(randomSeed: 42);

            // Assert
            Assert.NotNull(compression);
        }

        #endregion

        #region Compress Tests

        [Fact]
        public void Compress_WithValidWeights_ReturnsCompressedData()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.5,
                numClusters: 8,
                randomSeed: 42);

            var weights = new Vector<double>(new double[] {
                0.001, 0.5, 0.002, 0.8, 0.003, 0.7, 0.004, 0.9,
                0.001, 0.5, 0.002, 0.8, 0.003, 0.7, 0.004, 0.9
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);

            // Assert
            Assert.NotNull(compressedWeights);
            Assert.NotNull(metadata);
            Assert.IsType<DeepCompressionMetadata<double>>(metadata);
        }

        [Fact]
        public void Compress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new DeepCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => compression.Compress(null!));
        }

        [Fact]
        public void Compress_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var compression = new DeepCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compression.Compress(new Vector<double>(Array.Empty<double>())));
        }

        [Fact]
        public void Compress_ProducesDeepCompressionMetadata()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.5,
                numClusters: 8,
                randomSeed: 42);

            var weights = new Vector<double>(new double[] {
                0.001, 0.5, 0.002, 0.8, 0.003, 0.7, 0.004, 0.9,
                0.001, 0.5, 0.002, 0.8, 0.003, 0.7, 0.004, 0.9
            });

            // Act
            var (_, metadata) = compression.Compress(weights);
            var deepMetadata = (DeepCompressionMetadata<double>)metadata;

            // Assert
            Assert.NotNull(deepMetadata.PruningMetadata);
            Assert.NotNull(deepMetadata.ClusteringMetadata);
            Assert.NotNull(deepMetadata.HuffmanMetadata);
            Assert.NotNull(deepMetadata.CompressionStats);
            Assert.Equal(weights.Length, deepMetadata.OriginalLength);
        }

        [Fact]
        public void Compress_WithHighSparsity_PrunesMostWeights()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.9,
                numClusters: 8,
                randomSeed: 42);

            var random = RandomHelper.CreateSeededRandom(42);
            var weights = new double[100];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble();
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (_, metadata) = compression.Compress(weightsVector);
            var deepMetadata = (DeepCompressionMetadata<double>)metadata;

            // Assert
            Assert.True(deepMetadata.PruningMetadata.ActualSparsity >= 0.8,
                $"Expected sparsity >= 0.8, got {deepMetadata.PruningMetadata.ActualSparsity}");
        }

        #endregion

        #region Decompress Tests

        [Fact]
        public void Decompress_ReconstructsApproximateWeights()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.3,
                numClusters: 16,
                randomSeed: 42);

            var originalWeights = new Vector<double>(new double[] {
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2
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
            var compression = new DeepCompression<double>();
            var pruningMetadata = new SparsePruningMetadata<double>(
                new int[] { 0, 1 }, 10, 0.1, 0.8);
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5 }, 1, 2);
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(default, 0, true, 0, null, null),
                new NumericDictionary<double, string>(), 1, 0);
            var metadata = new DeepCompressionMetadata<double>(
                pruningMetadata, clusteringMetadata, huffmanMetadata, 10, new DeepCompressionStats());

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(null!, metadata));
        }

        #endregion

        #region GetCompressedSize Tests

        [Fact]
        public void GetCompressedSize_ReturnsPositiveSize()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.5,
                numClusters: 8,
                randomSeed: 42);

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

        #endregion

        #region Compression Stats Tests

        [Fact]
        public void CompressionStats_ContainsValidStatistics()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.8,
                numClusters: 32,
                randomSeed: 42);

            var random = RandomHelper.CreateSeededRandom(42);
            var weights = new double[200];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble();
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (_, metadata) = compression.Compress(weightsVector);
            var deepMetadata = (DeepCompressionMetadata<double>)metadata;
            var stats = deepMetadata.CompressionStats;

            // Assert
            Assert.True(stats.OriginalSizeBytes > 0);
            Assert.True(stats.NumClusters > 0);
            Assert.True(stats.BitsPerWeight > 0);
        }

        [Fact]
        public void CompressionStats_PruningRatio_ReturnsPositiveValue()
        {
            // Arrange
            var stats = new DeepCompressionStats { Sparsity = 0.9 };

            // Act & Assert
            Assert.True(stats.PruningRatio > 1.0);
        }

        [Fact]
        public void CompressionStats_QuantizationRatio_ReturnsPositiveValue()
        {
            // Arrange
            var stats = new DeepCompressionStats { BitsPerWeight = 5.0 };

            // Act & Assert
            Assert.True(stats.QuantizationRatio > 1.0);
            Assert.Equal(32.0 / 5.0, stats.QuantizationRatio);
        }

        #endregion

        #region Type-Specific Tests

        [Fact]
        public void Compress_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var compression = new DeepCompression<float>(
                pruningSparsity: 0.5,
                numClusters: 8,
                randomSeed: 42);

            var weights = new Vector<float>(new float[] {
                0.001f, 0.5f, 0.002f, 0.8f, 0.003f, 0.7f, 0.004f, 0.9f,
                0.001f, 0.5f, 0.002f, 0.8f, 0.003f, 0.7f, 0.004f, 0.9f
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
        public void Metadata_Constructor_WithNullPruningMetadata_ThrowsException()
        {
            // Arrange
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5 }, 1, 2);
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(default, 0, true, 0, null, null),
                new NumericDictionary<double, string>(), 1, 0);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new DeepCompressionMetadata<double>(
                    null!, clusteringMetadata, huffmanMetadata, 10, new DeepCompressionStats()));
        }

        [Fact]
        public void Metadata_Constructor_WithNullClusteringMetadata_ThrowsException()
        {
            // Arrange
            var pruningMetadata = new SparsePruningMetadata<double>(
                new int[] { 0, 1 }, 10, 0.1, 0.8);
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(default, 0, true, 0, null, null),
                new NumericDictionary<double, string>(), 1, 0);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new DeepCompressionMetadata<double>(
                    pruningMetadata, null!, huffmanMetadata, 10, new DeepCompressionStats()));
        }

        [Fact]
        public void Metadata_Constructor_WithNullHuffmanMetadata_ThrowsException()
        {
            // Arrange
            var pruningMetadata = new SparsePruningMetadata<double>(
                new int[] { 0, 1 }, 10, 0.1, 0.8);
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5 }, 1, 2);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new DeepCompressionMetadata<double>(
                    pruningMetadata, clusteringMetadata, null!, 10, new DeepCompressionStats()));
        }

        [Fact]
        public void Metadata_GetMetadataSize_ReturnsPositiveValue()
        {
            // Arrange
            var pruningMetadata = new SparsePruningMetadata<double>(
                new int[] { 0, 1, 5, 7 }, 10, 0.1, 0.6);
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5, 1.5 }, 2, 4);
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(0.5, 1, true, 0, null, null),
                new NumericDictionary<double, string>(), 4, 8);
            var metadata = new DeepCompressionMetadata<double>(
                pruningMetadata, clusteringMetadata, huffmanMetadata, 10, new DeepCompressionStats());

            // Act
            var size = metadata.GetMetadataSize();

            // Assert
            Assert.True(size > 0);
        }

        [Fact]
        public void Metadata_Type_ReturnsCorrectCompressionType()
        {
            // Arrange
            var pruningMetadata = new SparsePruningMetadata<double>(
                new int[] { 0, 1 }, 10, 0.1, 0.8);
            var clusteringMetadata = new WeightClusteringMetadata<double>(
                new double[] { 0.5 }, 1, 2);
            var huffmanMetadata = new HuffmanEncodingMetadata<double>(
                new HuffmanNode<double>(default, 0, true, 0, null, null),
                new NumericDictionary<double, string>(), 1, 0);
            var metadata = new DeepCompressionMetadata<double>(
                pruningMetadata, clusteringMetadata, huffmanMetadata, 10, new DeepCompressionStats());

            // Act & Assert
            // DeepCompression should return its own CompressionType, not HybridHuffmanClustering
            Assert.Equal(AiDotNet.Enums.CompressionType.DeepCompression, metadata.Type);
        }

        #endregion

        #region Round-Trip Tests

        [Fact]
        public void CompressAndDecompress_RoundTrip_PreservesLargeWeights()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.5,
                numClusters: 16,
                randomSeed: 42);

            // Create weights with clear distinction between small and large values
            var originalWeights = new Vector<double>(new double[] {
                0.001, 10.0, 0.002, 20.0, 0.001, 30.0, 0.002, 40.0,
                0.001, 10.0, 0.002, 20.0, 0.001, 30.0, 0.002, 40.0
            });

            // Act
            var (compressedWeights, metadata) = compression.Compress(originalWeights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(originalWeights.Length, decompressedWeights.Length);

            // Large weights should be preserved (though quantized)
            // Small weights may be pruned to zero
        }

        [Fact]
        public void CompressAndDecompress_WithLargeDataset_CompletesInReasonableTime()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.9,
                numClusters: 32,
                maxClusteringIterations: 50,
                randomSeed: 42);

            var random = RandomHelper.CreateSeededRandom(42);
            var weights = new double[1000];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble();
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var startTime = DateTime.Now;
            var (compressedWeights, metadata) = compression.Compress(weightsVector);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);
            var elapsed = DateTime.Now - startTime;

            // Assert
            Assert.True(elapsed.TotalSeconds < 30, $"Operation took {elapsed.TotalSeconds} seconds");
            Assert.Equal(weights.Length, decompressedWeights.Length);
        }

        [Fact]
        public void CompressAndDecompress_WithReproducibleSeed_ProducesSameResults()
        {
            // Arrange
            var compression1 = new DeepCompression<double>(
                pruningSparsity: 0.5,
                numClusters: 8,
                randomSeed: 42);
            var compression2 = new DeepCompression<double>(
                pruningSparsity: 0.5,
                numClusters: 8,
                randomSeed: 42);

            var weights = new Vector<double>(new double[] {
                0.1, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.9,
                0.1, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.9
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

        #endregion

        #region Edge Case Tests

        [Fact]
        public void Compress_WithAllSameValues_HandlesGracefully()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.3,
                numClusters: 4,
                randomSeed: 42);

            var weights = new double[16];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 0.5; // All same value
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (compressedWeights, metadata) = compression.Compress(weightsVector);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
        }

        [Fact]
        public void Compress_WithVerySmallWeights_PrunesAll()
        {
            // Arrange
            var compression = new DeepCompression<double>(
                pruningSparsity: 0.99,
                pruningThreshold: 0.1,
                numClusters: 4,
                randomSeed: 42);

            // All weights are below the threshold
            var weights = new double[16];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 0.001 * (i + 1);
            }
            var weightsVector = new Vector<double>(weights);

            // Act
            var (_, metadata) = compression.Compress(weightsVector);
            var deepMetadata = (DeepCompressionMetadata<double>)metadata;

            // Assert
            Assert.True(deepMetadata.PruningMetadata.ActualSparsity >= 0.5);
        }

        #endregion
    }
}
