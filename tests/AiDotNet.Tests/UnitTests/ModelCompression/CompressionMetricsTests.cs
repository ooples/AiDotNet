using System;
using AiDotNet.ModelCompression;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.ModelCompression
{
    public class CompressionMetricsTests
    {
        [Fact(Timeout = 60000)]
        public async Task CalculateDerivedMetrics_ComputesCompressionRatio()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 1000,
                CompressedSize = 100
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(10.0, metrics.CompressionRatio);
        }

        [Fact(Timeout = 60000)]
        public async Task CalculateDerivedMetrics_ComputesSizeReductionPercentage()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 1000,
                CompressedSize = 250
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(75.0, metrics.SizeReductionPercentage);
        }

        [Fact(Timeout = 60000)]
        public async Task CalculateDerivedMetrics_ComputesInferenceSpeedup()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
            {
                OriginalInferenceTimeMs = 100.0,
                CompressedInferenceTimeMs = 50.0
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(2.0, metrics.InferenceSpeedup);
        }

        [Fact(Timeout = 60000)]
        public async Task CalculateDerivedMetrics_ComputesAccuracyLoss()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
            {
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.93
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(0.02, metrics.AccuracyLoss, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task MeetsQualityThreshold_WithAcceptableMetrics_ReturnsTrue()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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

        [Fact(Timeout = 60000)]
        public async Task MeetsQualityThreshold_WithExcessiveAccuracyLoss_ReturnsFalse()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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

        [Fact(Timeout = 60000)]
        public async Task MeetsQualityThreshold_WithInsufficientCompression_ReturnsFalse()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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

        [Fact(Timeout = 60000)]
        public async Task ToString_ReturnsFormattedSummary()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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

        [Fact(Timeout = 60000)]
        public async Task CalculateDerivedMetrics_WithZeroCompressedSize_HandlesGracefully()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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

        [Fact(Timeout = 60000)]
        public async Task CalculateDerivedMetrics_WithZeroInferenceTime_HandlesGracefully()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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

        [Fact(Timeout = 60000)]
        public async Task CalculateDerivedMetrics_WithAllMetrics_ComputesAllValues()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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
            Assert.Equal(90.0, metrics.SizeReductionPercentage, 2); // Use precision for floating-point comparison
            Assert.Equal(2.0, metrics.InferenceSpeedup);
            Assert.Equal(0.01, metrics.AccuracyLoss, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task MeetsQualityThreshold_WithCustomThresholds_UsesProvidedValues()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
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

        [Fact(Timeout = 60000)]
        public async Task CalculateCompositeFitness_ReturnsValueInRange()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 10000,
                CompressedSize = 1000,
                OriginalInferenceTimeMs = 100.0,
                CompressedInferenceTimeMs = 50.0,
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.94
            };
            metrics.CalculateDerivedMetrics();

            // Act
            var fitness = metrics.CalculateCompositeFitness();

            // Assert
            Assert.True(fitness >= 0.0 && fitness <= 1.0);
        }

        [Fact(Timeout = 60000)]
        public async Task IsBetterThan_WithHigherFitness_ReturnsTrue()
        {
            // Arrange
            var better = new CompressionMetrics<double>
            {
                OriginalSize = 10000,
                CompressedSize = 500, // 20x compression
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.94
            };
            better.CalculateDerivedMetrics();

            var worse = new CompressionMetrics<double>
            {
                OriginalSize = 10000,
                CompressedSize = 5000, // Only 2x compression
                OriginalAccuracy = 0.95,
                CompressedAccuracy = 0.90 // More accuracy loss
            };
            worse.CalculateDerivedMetrics();

            // Act & Assert
            Assert.True(better.IsBetterThan(worse));
            Assert.False(worse.IsBetterThan(better));
        }

        [Fact(Timeout = 60000)]
        public async Task IsBetterThan_WithNullOther_ReturnsTrue()
        {
            // Arrange
            var metrics = new CompressionMetrics<double>
            {
                OriginalSize = 1000,
                CompressedSize = 100
            };
            metrics.CalculateDerivedMetrics();

            // Act & Assert
            Assert.True(metrics.IsBetterThan(null!));
        }

        [Fact(Timeout = 60000)]
        public async Task FromDeepCompressionStats_CreatesMetricsCorrectly()
        {
            // Arrange
            var stats = new DeepCompressionStats
            {
                OriginalSizeBytes = 1000000,
                CompressedSizeBytes = 50000,
                Sparsity = 0.9,
                BitsPerWeight = 5.0
            };

            // Act
            var metrics = CompressionMetrics<double>.FromDeepCompressionStats(stats, "Test Compression");

            // Assert
            Assert.Equal(1000000, metrics.OriginalSize);
            Assert.Equal(50000, metrics.CompressedSize);
            Assert.Equal(0.9, metrics.Sparsity);
            Assert.Equal(5.0, metrics.BitsPerWeight);
            Assert.Equal("Test Compression", metrics.CompressionTechnique);
        }

        [Fact(Timeout = 60000)]
        public async Task CompressionMetrics_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var metrics = new CompressionMetrics<float>
            {
                OriginalSize = 1000,
                CompressedSize = 100,
                OriginalAccuracy = 0.95f,
                CompressedAccuracy = 0.94f
            };

            // Act
            metrics.CalculateDerivedMetrics();

            // Assert
            Assert.Equal(10.0f, metrics.CompressionRatio);
            Assert.True(metrics.MeetsQualityThreshold(2.0, 2.0));
        }

        [Fact(Timeout = 60000)]
        public async Task NewProperties_HaveDefaultValues()
        {
            // Arrange & Act
            var metrics = new CompressionMetrics<double>();

            // Assert
            Assert.Equal(0.0, metrics.Sparsity);
            Assert.Equal(0.0, metrics.BitsPerWeight);
            Assert.Equal(0.0, metrics.MemoryBandwidthSavings);
            Assert.Equal(0.0, metrics.ReconstructionError);
        }
    }
}
