using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class HuffmanEncodingCompressionTests
    {
        [Fact]
        public void Constructor_WithValidPrecision_CreatesInstance()
        {
            // Arrange & Act
            var compression = new HuffmanEncodingCompression<double>(precision: 6);

            // Assert
            Assert.NotNull(compression);
        }

        [Fact]
        public void Constructor_WithNegativePrecision_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new HuffmanEncodingCompression<double>(precision: -1));
        }

        [Fact]
        public void Compress_WithValidWeights_ReturnsCompressedData()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>(precision: 2);
            var weights = new Vector<double>(new double[] { 1.11, 1.12, 1.11, 2.22, 2.22, 2.22, 3.33 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);

            // Assert
            Assert.NotNull(compressedWeights);
            Assert.NotNull(metadata);
            Assert.IsType<HuffmanEncodingMetadata<double>>(metadata);
        }

        [Fact]
        public void Compress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => compression.Compress(null!));
        }

        [Fact]
        public void Compress_WithEmptyWeights_ThrowsException()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => compression.Compress(new Vector<double>(Array.Empty<double>())));
        }

        [Fact]
        public void Decompress_ReconstructsExactWeights()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>(precision: 2);
            var originalWeights = new Vector<double>(new double[] { 1.11, 1.12, 1.11, 2.22, 2.22, 2.22, 3.33 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(originalWeights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(originalWeights.Length, decompressedWeights.Length);
            // Verify weights are reconstructed with precision rounding
            for (int i = 0; i < originalWeights.Length; i++)
            {
                var expected = Math.Round(originalWeights[i], 2);
                Assert.Equal(expected, decompressedWeights[i], 2);
            }
        }

        [Fact]
        public void Decompress_WithNullWeights_ThrowsException()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>();
            var huffmanTree = new HuffmanNode<double>(
                value: 1.0,
                frequency: 1,
                isLeaf: true,
                id: 0,
                left: null,
                right: null);
            var metadata = new HuffmanEncodingMetadata<double>(
                huffmanTree: huffmanTree,
                encodingTable: new AiDotNet.LinearAlgebra.NumericDictionary<double, string>(),
                originalLength: 10,
                bitLength: 10);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compression.Decompress(null!, metadata));
        }

        [Fact]
        public void Compress_WithFrequentValues_AchievesBetterCompression()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>(precision: 1);
            // Create weights with highly skewed frequency distribution
            var weightsArray = new double[1000];
            for (int i = 0; i < 900; i++) weightsArray[i] = 1.0; // 90% are 1.0
            for (int i = 900; i < 990; i++) weightsArray[i] = 2.0; // 9% are 2.0
            for (int i = 990; i < 1000; i++) weightsArray[i] = 3.0; // 1% are 3.0
            var weights = new Vector<double>(weightsArray);

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var originalSize = weightsArray.Length * sizeof(double);
            var compressedSize = compression.GetCompressedSize(compressedWeights, metadata);

            // Assert
            // Huffman encoding should compress well with skewed distributions
            Assert.True(compressedSize < originalSize);
        }

        [Fact]
        public void GetCompressedSize_ReturnsPositiveValue()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>(precision: 2);
            var weights = new Vector<double>(new double[] { 1.0, 1.0, 2.0, 2.0, 3.0 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var compressedSize = compression.GetCompressedSize(compressedWeights, metadata);

            // Assert
            Assert.True(compressedSize > 0);
        }

        [Fact]
        public void Compress_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<float>(precision: 2);
            var weights = new Vector<float>(new float[] { 1.1f, 1.1f, 2.2f, 2.2f, 3.3f });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                Assert.Equal(Math.Round(weights[i], 2), decompressedWeights[i], 2);
            }
        }

        [Fact]
        public void Compress_WithDifferentPrecisions_ProducesDifferentResults()
        {
            // Arrange
            var compression1 = new HuffmanEncodingCompression<double>(precision: 1);
            var compression2 = new HuffmanEncodingCompression<double>(precision: 4);
            var weights = new Vector<double>(new double[] { 1.12345, 2.23456, 3.34567, 4.45678 });

            // Act
            var (_, metadata1) = compression1.Compress(weights);
            var (_, metadata2) = compression2.Compress(weights);
            var huffmanMetadata1 = (HuffmanEncodingMetadata<double>)metadata1;
            var huffmanMetadata2 = (HuffmanEncodingMetadata<double>)metadata2;

            // Assert
            // Higher precision should have more unique values in encoding table
            Assert.True(huffmanMetadata2.EncodingTable.Count >= huffmanMetadata1.EncodingTable.Count);
        }

        [Fact]
        public void Compress_IsLosslessWithinPrecision()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>(precision: 3);
            var weights = new Vector<double>(new double[] { 1.123456, 2.234567, 3.345678, 1.123456 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            // Should be lossless within the specified precision
            for (int i = 0; i < weights.Length; i++)
            {
                var expected = Math.Round(weights[i], 3);
                Assert.Equal(expected, Math.Round(decompressedWeights[i], 3), 3);
            }
        }

        [Fact]
        public void CalculateCompressionRatio_WithValidSizes_ReturnsCorrectRatio()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>();
            long originalSize = 1000;
            long compressedSize = 200;

            // Act
            var ratio = compression.CalculateCompressionRatio(originalSize, compressedSize);

            // Assert
            Assert.Equal(5.0, ratio);
        }

        [Fact]
        public void Compress_WithSingleUniqueValue_HandlesDegenerateCase()
        {
            // Arrange
            var compression = new HuffmanEncodingCompression<double>(precision: 1);
            var weights = new Vector<double>(new double[] { 5.0, 5.0, 5.0, 5.0 });

            // Act
            var (compressedWeights, metadata) = compression.Compress(weights);
            var decompressedWeights = compression.Decompress(compressedWeights, metadata);

            // Assert
            Assert.Equal(weights.Length, decompressedWeights.Length);
            foreach (var weight in decompressedWeights)
            {
                Assert.Equal(5.0, weight, 1);
            }
        }
    }
}
