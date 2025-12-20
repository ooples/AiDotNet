using System;
using System.Text;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class CompressionHelperTests
    {
        [Fact]
        public void Compress_WithNullData_ThrowsArgumentNullException()
        {
            // Arrange
            var config = new CompressionConfig();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => CompressionHelper.Compress(null!, config));
        }

        [Fact]
        public void Compress_WithNullConfig_ThrowsArgumentNullException()
        {
            // Arrange
            var data = new byte[] { 1, 2, 3, 4, 5 };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => CompressionHelper.Compress(data, null!));
        }

        [Fact]
        public void Compress_WithNoneMode_ReturnsOriginalData()
        {
            // Arrange
            var data = new byte[] { 1, 2, 3, 4, 5 };
            var config = new CompressionConfig { Mode = ModelCompressionMode.None };

            // Act
            var result = CompressionHelper.Compress(data, config);

            // Assert
            Assert.Equal(data, result);
        }

        [Fact]
        public void Compress_WithAutomaticMode_CompressesData()
        {
            // Arrange
            var data = Encoding.UTF8.GetBytes(new string('a', 1000)); // Repetitive data compresses well
            var config = new CompressionConfig { Mode = ModelCompressionMode.Automatic };

            // Act
            var result = CompressionHelper.Compress(data, config);

            // Assert
            Assert.NotEqual(data, result);
            Assert.True(CompressionHelper.IsCompressedData(result));
        }

        [Fact]
        public void DecompressIfNeeded_WithNullData_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => CompressionHelper.DecompressIfNeeded(null!));
        }

        [Fact]
        public void DecompressIfNeeded_WithUncompressedData_ReturnsOriginalData()
        {
            // Arrange
            var data = new byte[] { 1, 2, 3, 4, 5 };

            // Act
            var result = CompressionHelper.DecompressIfNeeded(data);

            // Assert
            Assert.Equal(data, result);
        }

        [Fact]
        public void CompressAndDecompress_RoundTrip_ReturnsOriginalData()
        {
            // Arrange
            var originalData = Encoding.UTF8.GetBytes("Test data for compression round trip testing");
            var config = new CompressionConfig { Mode = ModelCompressionMode.Full };

            // Act
            var compressedData = CompressionHelper.Compress(originalData, config);
            var decompressedData = CompressionHelper.DecompressIfNeeded(compressedData);

            // Assert
            Assert.Equal(originalData, decompressedData);
        }

        [Fact]
        public void CompressAndDecompress_LargeData_RoundTrip_ReturnsOriginalData()
        {
            // Arrange
            var random = RandomHelper.CreateSeededRandom(42);
            var originalData = new byte[100000];
            random.NextBytes(originalData);
            var config = new CompressionConfig { Mode = ModelCompressionMode.Full };

            // Act
            var compressedData = CompressionHelper.Compress(originalData, config);
            var decompressedData = CompressionHelper.DecompressIfNeeded(compressedData);

            // Assert
            Assert.Equal(originalData, decompressedData);
        }

        [Fact]
        public void IsCompressedData_WithCompressedData_ReturnsTrue()
        {
            // Arrange
            var data = Encoding.UTF8.GetBytes("Test data");
            var config = new CompressionConfig { Mode = ModelCompressionMode.Full };
            var compressedData = CompressionHelper.Compress(data, config);

            // Act
            var result = CompressionHelper.IsCompressedData(compressedData);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void IsCompressedData_WithUncompressedData_ReturnsFalse()
        {
            // Arrange
            var data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

            // Act
            var result = CompressionHelper.IsCompressedData(data);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void IsCompressedData_WithNullData_ReturnsFalse()
        {
            // Act
            var result = CompressionHelper.IsCompressedData(null!);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void IsCompressedData_WithTooShortData_ReturnsFalse()
        {
            // Arrange
            var data = new byte[] { 1, 2, 3 };

            // Act
            var result = CompressionHelper.IsCompressedData(data);

            // Assert
            Assert.False(result);
        }

        [Theory]
        [InlineData(CompressionType.HuffmanEncoding)]
        [InlineData(CompressionType.WeightClustering)]
        [InlineData(CompressionType.HybridHuffmanClustering)]
        public void Compress_WithDifferentTypes_WorksCorrectly(CompressionType type)
        {
            // Arrange
            var data = Encoding.UTF8.GetBytes(new string('x', 500));
            var config = new CompressionConfig
            {
                Mode = ModelCompressionMode.Full,
                Type = type
            };

            // Act
            var compressedData = CompressionHelper.Compress(data, config);
            var decompressedData = CompressionHelper.DecompressIfNeeded(compressedData);

            // Assert
            Assert.Equal(data, decompressedData);
        }

        [Fact]
        public void GetCompressionStats_ReturnsCorrectStatistics()
        {
            // Arrange
            var originalData = Encoding.UTF8.GetBytes(new string('a', 1000));
            var config = new CompressionConfig { Mode = ModelCompressionMode.Full };
            var compressedData = CompressionHelper.Compress(originalData, config);

            // Act
            var stats = CompressionHelper.GetCompressionStats(originalData, compressedData);

            // Assert
            Assert.Equal(originalData.Length, stats.originalSize);
            Assert.True(stats.compressedSize > 0);
            Assert.True(stats.ratio > 1.0); // Should be compression
            Assert.True(stats.savedPercent > 0); // Should save space
        }

        [Fact]
        public void GetCompressionStats_WithNullData_ReturnsZeros()
        {
            // Act
            var stats = CompressionHelper.GetCompressionStats(null!, null!);

            // Assert
            Assert.Equal(0, stats.originalSize);
            Assert.Equal(0, stats.compressedSize);
            Assert.Equal(1.0, stats.ratio);
            Assert.Equal(0.0, stats.savedPercent);
        }

        [Fact]
        public void Compress_WithWeightsOnlyMode_CompressesData()
        {
            // Arrange
            var data = Encoding.UTF8.GetBytes(new string('w', 1000));
            var config = new CompressionConfig { Mode = ModelCompressionMode.WeightsOnly };

            // Act
            var result = CompressionHelper.Compress(data, config);

            // Assert
            Assert.True(CompressionHelper.IsCompressedData(result));
        }

        [Fact]
        public void Compress_AchievesSignificantCompression_ForRepetitiveData()
        {
            // Arrange
            var repetitiveData = Encoding.UTF8.GetBytes(new string('z', 10000));
            var config = new CompressionConfig { Mode = ModelCompressionMode.Full };

            // Act
            var compressedData = CompressionHelper.Compress(repetitiveData, config);
            var stats = CompressionHelper.GetCompressionStats(repetitiveData, compressedData);

            // Assert
            Assert.True(stats.ratio > 5.0, $"Expected ratio > 5.0, got {stats.ratio}");
            Assert.True(stats.savedPercent > 80.0, $"Expected > 80% saved, got {stats.savedPercent}%");
        }
    }
}
