using System;
using AiDotNet.ModelCompression;
using Xunit;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class CompressionMetricsTests
    {
        [Fact]
        public void CalculateDerivedMetrics_ComputesCompressionRatio()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 1000,
                CompressedSize = 100
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(10.0, metrics.CompressionRatio);
        }

        [Fact]
        public void CalculateDerivedMetrics_ComputesSizeReductionPercentage()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 1000,
                CompressedSize = 250
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(75.0, metrics.SizeReductionPercentage);
        }

        [Fact]
        public void CalculateDerivedMetrics_ComputesInferenceSpeedup()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalInferenceTimeMs = 100.0,
                CompressedInferenceTimeMs = 50.0
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(2.0, metrics.InferenceSpeedup);
        }

        [Fact]
        public void CalculateDerivedMetrics_ComputesAccuracyLoss()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.93
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(0.02, metrics.AccuracyLoss, 10);
        }

        [Fact]
        public void MeetsQualityThreshold_WithAcceptableMetrics_ReturnsTrue()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 1000,
                CompressedSize = 100,
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.94
            };
            metrics.CalculateDerivedMetrics();

            // Act
            var meetsThreshold = metrics.MeetsQualityThreshold(
                maxAccuracyLossPercentage: 2.0,
                minCompressionRatio: 2.0);

            // Assert
            Assert.True(meetsThreshold);
        }

        [Fact]
        public void MeetsQualityThreshold_WithExcessiveAccuracyLoss_ReturnsFalse()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 1000,
                CompressedSize = 100,
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.90 // 5% loss
            };
            metrics.CalculateDerivedMetrics();

            // Act
            var meetsThreshold = metrics.MeetsQualityThreshold(
                maxAccuracyLossPercentage: 2.0,
                minCompressionRatio: 2.0);

            // Assert
            Assert.False(meetsThreshold);
        }

        [Fact]
        public void MeetsQualityThreshold_WithInsufficientCompression_ReturnsFalse()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 1000,
                CompressedSize = 600, // Only 1.67x compression
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.94
            };
            metrics.CalculateDerivedMetrics();

            // Act
            var meetsThreshold = metrics.MeetsQualityThreshold(
                maxAccuracyLossPercentage: 2.0,
                minCompressionRatio: 2.0);

            // Assert
            Assert.False(meetsThreshold);
        }

        [Fact]
        public void ToString_ReturnsFormattedSummary()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                CompressionTechnique = "Weight Clustering",
                OriginalSize = 1000000,
                CompressedSize = 100000,
                OriginalParameterCount = 1000,
                EffectiveParameterCount = 256,
                OriginalInferenceTimeMs = 100.0,
                CompressedInferenceTimeMs = 50.0,
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.94,
                CompressionTimeMs = 1000.0,
                DecompressionTimeMs = 50.0
            };
            metrics.CalculateDerivedMetrics();

            // Act
            var summary = metrics.ToString();

            // Assert
            Assert.Contains("Weight Clustering", summary);
            Assert.Contains("10.00x", summary); // Compression ratio
            Assert.Contains("2.00x", summary); // Inference speedup
        }

        [Fact]
        public void CalculateDerivedMetrics_WithZeroCompressedSize_HandlesGracefully()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 1000,
                CompressedSize = 0
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            // Should not crash, but ratio won't be meaningful
            Assert.Equal(0.0, metrics.CompressionRatio);
        }

        [Fact]
        public void CalculateDerivedMetrics_WithZeroInferenceTime_HandlesGracefully()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalInferenceTimeMs = 100.0,
                CompressedInferenceTimeMs = 0.0
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            // Should not crash, but speedup won't be meaningful
            Assert.Equal(0.0, metrics.InferenceSpeedup);
        }

        [Fact]
        public void CalculateDerivedMetrics_WithAllMetrics_ComputesAllValues()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 10000,
                CompressedSize = 1000,
                OriginalInferenceTimeMs = 200.0,
                CompressedInferenceTimeMs = 100.0,
                OriginalAccuracy = 0.98,
                CompressedAccuracy = 0.97
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(10.0, metrics.CompressionRatio);
            Assert.Equal(90.0, metrics.SizeReductionPercentage);
            Assert.Equal(2.0, metrics.InferenceSpeedup);
            Assert.Equal(0.01, metrics.AccuracyLoss, 10);
        }

        [Fact]
        public void MeetsQualityThreshold_WithCustomThresholds_UsesProvidedValues()
        {
            // Arrange
            var metrics = new CompressionMetrics
            {
                OriginalSize = 1000,
                CompressedSize = 100,
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.91 // 4% loss
            };
            metrics.CalculateDerivedMetrics();

            // Act
            var meetsStrictThreshold = metrics.MeetsQualityThreshold(
                maxAccuracyLossPercentage: 2.0,
                minCompressionRatio: 5.0);
            var meetsRelaxedThreshold = metrics.MeetsQualityThreshold(
                maxAccuracyLossPercentage: 5.0,
                minCompressionRatio: 5.0);

            // Assert
            Assert.False(meetsStrictThreshold);
            Assert.True(meetsRelaxedThreshold);
        }
    }
}
