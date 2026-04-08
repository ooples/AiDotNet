using System;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Interpretability
{
    /// <summary>
    /// Unit tests for the BiasDetector class.
    /// </summary>
    public class BiasDetectorTests
    {
        [Fact]
        public void DetectBias_WithMismatchedLengths_ThrowsArgumentException()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            Vector<double> predictions = new Vector<double>(new double[] { 1, 0, 1, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 1 });

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => detector.DetectBias(predictions, sensitiveFeature));
            Assert.Contains("must match", exception.Message);
        }

        [Fact]
        public void DetectBias_WithSingleGroup_ReturnsNoBias()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            Vector<double> predictions = new Vector<double>(new double[] { 1, 0, 1, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias);
            Assert.Contains("Insufficient groups", result.Message);
        }

        [Fact]
        public void DetectBias_WithBalancedGroups_ReturnsNoBias()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            // Group 0: 2 out of 4 positive = 50%
            // Group 1: 2 out of 4 positive = 50%
            Vector<double> predictions = new Vector<double>(new double[] { 1, 1, 0, 0, 1, 1, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias);
            Assert.Equal(1.0, result.DisparateImpactRatio, 3);
            Assert.Equal(0.0, result.StatisticalParityDifference, 3);
        }

        [Fact]
        public void DetectBias_WithUnbalancedGroups_DetectsBias()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            // Group 0: 4 out of 4 positive = 100%
            // Group 1: 0 out of 4 positive = 0%
            Vector<double> predictions = new Vector<double>(new double[] { 1, 1, 1, 1, 0, 0, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.True(result.HasBias);
            Assert.Equal(0.0, result.DisparateImpactRatio, 3);
            Assert.Equal(1.0, result.StatisticalParityDifference, 3);
            Assert.Contains("Bias detected", result.Message);
        }

        [Fact]
        public void DetectBias_WithModerateDisparity_DetectsBias()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            // Group 0: 3 out of 4 positive = 75%
            // Group 1: 2 out of 4 positive = 50%
            // Disparate Impact = 50/75 = 0.667 < 0.8
            Vector<double> predictions = new Vector<double>(new double[] { 1, 1, 1, 0, 1, 1, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.True(result.HasBias);
            Assert.True(result.DisparateImpactRatio < 0.8);
            Assert.Equal(0.25, result.StatisticalParityDifference, 3);
        }

        [Fact]
        public void DetectBias_WithActualLabels_ComputesAdditionalMetrics()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            Vector<double> predictions = new Vector<double>(new double[] { 1, 1, 0, 0, 1, 0, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 });
            Vector<double> actualLabels = new Vector<double>(new double[] { 1, 1, 1, 0, 1, 1, 0, 0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

            // Assert
            Assert.NotNull(result.GroupTruePositiveRates);
            Assert.NotNull(result.GroupFalsePositiveRates);
            Assert.NotNull(result.GroupPrecisions);
            Assert.Equal(2, result.GroupTruePositiveRates.Count);
            Assert.Equal(2, result.GroupFalsePositiveRates.Count);
            Assert.Equal(2, result.GroupPrecisions.Count);
        }

        [Fact]
        public void DetectBias_WithActualLabels_ComputesEqualOpportunity()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            // Group 0: TPR = 2/2 = 100% (2 actual positives, both predicted correctly)
            // Group 1: TPR = 1/2 = 50% (2 actual positives, 1 predicted correctly)
            Vector<double> predictions = new Vector<double>(new double[] { 1, 1, 0, 0, 1, 0, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 });
            Vector<double> actualLabels = new Vector<double>(new double[] { 1, 1, 0, 0, 1, 1, 0, 0 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature, actualLabels);

            // Assert
            Assert.NotEqual(0.0, result.EqualOpportunityDifference);
            Assert.True(result.HasBias);
        }

        [Fact]
        public void DetectBias_WithThreeGroups_HandlesMultipleGroups()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            Vector<double> predictions = new Vector<double>(new double[] { 1, 1, 0, 1, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 1, 1, 2, 2 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.Equal(3, result.GroupPositiveRates.Count);
            Assert.Equal(3, result.GroupSizes.Count);
        }

        [Fact]
        public void DetectBias_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<float>();
            Vector<float> predictions = new Vector<float>(new float[] { 1, 1, 0, 0, 1, 1, 0, 0 });
            Vector<float> sensitiveFeature = new Vector<float>(new float[] { 0, 0, 0, 0, 1, 1, 1, 1 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.False(result.HasBias);
            Assert.Equal(1.0f, result.DisparateImpactRatio, 3);
        }

        [Fact]
        public void DetectBias_WithAllZeroPredictions_HandlesGracefully()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            Vector<double> predictions = new Vector<double>(new double[] { 0, 0, 0, 0, 0, 0, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            // When all predictions are zero, disparate impact should be 1.0 (no disparity)
            Assert.Equal(1.0, result.DisparateImpactRatio, 3);
            Assert.Equal(0.0, result.StatisticalParityDifference, 3);
            Assert.False(result.HasBias);
        }

        [Fact]
        public void DetectBias_GroupStatistics_AreCorrect()
        {
            // Arrange
            var detector = new DisparateImpactBiasDetector<double>();
            Vector<double> predictions = new Vector<double>(new double[] { 1, 1, 1, 0, 1, 0, 0, 0 });
            Vector<double> sensitiveFeature = new Vector<double>(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 });

            // Act
            var result = detector.DetectBias(predictions, sensitiveFeature);

            // Assert
            Assert.Equal(4, result.GroupSizes["0"]);
            Assert.Equal(4, result.GroupSizes["1"]);
            Assert.Equal(0.75, result.GroupPositiveRates["0"], 3); // 3 out of 4
            Assert.Equal(0.25, result.GroupPositiveRates["1"], 3); // 1 out of 4
        }
    }
}
