using AiDotNet.LinearAlgebra;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Statistics
{
    /// <summary>
    /// Integration tests for statistical functions with mathematically verified results.
    /// These tests ensure mathematical correctness of statistical calculations.
    /// </summary>
    public class StatisticsIntegrationTests
    {
        [Fact]
        public void Mean_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 2.0; data[1] = 4.0; data[2] = 6.0; data[3] = 8.0; data[4] = 10.0;
            // Mean = (2 + 4 + 6 + 8 + 10) / 5 = 30 / 5 = 6.0

            // Act
            var mean = StatisticsHelper.Mean(data);

            // Assert
            Assert.Equal(6.0, mean, precision: 10);
        }

        [Fact]
        public void Variance_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 2.0; data[1] = 4.0; data[2] = 6.0; data[3] = 8.0; data[4] = 10.0;
            // Mean = 6.0
            // Variance = [(2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2] / 5
            //          = [16 + 4 + 0 + 4 + 16] / 5 = 40 / 5 = 8.0

            // Act
            var variance = StatisticsHelper.Variance(data);

            // Assert
            Assert.Equal(8.0, variance, precision: 10);
        }

        [Fact]
        public void StandardDeviation_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 2.0; data[1] = 4.0; data[2] = 6.0; data[3] = 8.0; data[4] = 10.0;
            // Variance = 8.0
            // Standard Deviation = sqrt(8.0) ≈ 2.828427

            // Act
            var stdDev = StatisticsHelper.StandardDeviation(data);

            // Assert
            Assert.Equal(2.8284271247461903, stdDev, precision: 10);
        }

        [Fact]
        public void Median_OddCount_ReturnsMiddleValue()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 1.0; data[1] = 3.0; data[2] = 5.0; data[3] = 7.0; data[4] = 9.0;
            // Median of [1, 3, 5, 7, 9] = 5.0

            // Act
            var median = StatisticsHelper.Median(data);

            // Assert
            Assert.Equal(5.0, median, precision: 10);
        }

        [Fact]
        public void Median_EvenCount_ReturnsAverageOfMiddleTwo()
        {
            // Arrange
            var data = new Vector<double>(4);
            data[0] = 1.0; data[1] = 2.0; data[2] = 3.0; data[3] = 4.0;
            // Median of [1, 2, 3, 4] = (2 + 3) / 2 = 2.5

            // Act
            var median = StatisticsHelper.Median(data);

            // Assert
            Assert.Equal(2.5, median, precision: 10);
        }

        [Fact]
        public void Covariance_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 2.0; y[1] = 4.0; y[2] = 6.0; y[3] = 8.0; y[4] = 10.0;
            // Perfect linear relationship: y = 2x
            // Mean(x) = 3.0, Mean(y) = 6.0
            // Cov(x,y) = E[(x - mean(x))(y - mean(y))]
            //          = [(-2)(-4) + (-1)(-2) + (0)(0) + (1)(2) + (2)(4)] / 5
            //          = [8 + 2 + 0 + 2 + 8] / 5 = 20 / 5 = 4.0

            // Act
            var covariance = StatisticsHelper.Covariance(x, y);

            // Assert
            Assert.Equal(4.0, covariance, precision: 10);
        }

        [Fact]
        public void Correlation_PerfectPositive_ReturnsOne()
        {
            // Arrange
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 2.0; y[1] = 4.0; y[2] = 6.0; y[3] = 8.0; y[4] = 10.0;
            // Perfect linear relationship: y = 2x, correlation = 1.0

            // Act
            var correlation = StatisticsHelper.Correlation(x, y);

            // Assert
            Assert.Equal(1.0, correlation, precision: 10);
        }

        [Fact]
        public void Correlation_PerfectNegative_ReturnsMinusOne()
        {
            // Arrange
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 10.0; y[1] = 8.0; y[2] = 6.0; y[3] = 4.0; y[4] = 2.0;
            // Perfect negative linear relationship, correlation = -1.0

            // Act
            var correlation = StatisticsHelper.Correlation(x, y);

            // Assert
            Assert.Equal(-1.0, correlation, precision: 10);
        }

        [Fact]
        public void ZScore_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 2.0; data[1] = 4.0; data[2] = 6.0; data[3] = 8.0; data[4] = 10.0;
            // Mean = 6.0, Std Dev = 2.828427
            // Z-score for 10.0 = (10.0 - 6.0) / 2.828427 ≈ 1.414

            // Act
            var zScore = StatisticsHelper.ZScore(10.0, data);

            // Assert
            Assert.Equal(1.4142135623730951, zScore, precision: 10);
        }

        [Fact]
        public void Percentile_50th_EqualsMedian()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 1.0; data[1] = 2.0; data[2] = 3.0; data[3] = 4.0; data[4] = 5.0;

            // Act
            var percentile50 = StatisticsHelper.Percentile(data, 50);
            var median = StatisticsHelper.Median(data);

            // Assert
            Assert.Equal(median, percentile50, precision: 10);
        }

        [Fact]
        public void InterquartileRange_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(7);
            data[0] = 1.0; data[1] = 2.0; data[2] = 3.0; data[3] = 4.0;
            data[4] = 5.0; data[5] = 6.0; data[6] = 7.0;
            // Q1 (25th percentile) = 2.0
            // Q3 (75th percentile) = 6.0
            // IQR = Q3 - Q1 = 4.0

            // Act
            var iqr = StatisticsHelper.InterquartileRange(data);

            // Assert
            Assert.Equal(4.0, iqr, precision: 10);
        }

        [Fact]
        public void Skewness_SymmetricDistribution_ReturnsZero()
        {
            // Arrange - Symmetric data around mean
            var data = new Vector<double>(5);
            data[0] = -2.0; data[1] = -1.0; data[2] = 0.0; data[3] = 1.0; data[4] = 2.0;

            // Act
            var skewness = StatisticsHelper.Skewness(data);

            // Assert - Symmetric distribution has skewness ≈ 0
            Assert.True(Math.Abs(skewness) < 0.01);
        }

        [Fact]
        public void Kurtosis_NormalDistribution_ReturnsExpectedValue()
        {
            // Arrange - Approximately normal distributed data
            var data = new Vector<double>(9);
            data[0] = -3.0; data[1] = -2.0; data[2] = -1.0;
            data[3] = -0.5; data[4] = 0.0; data[5] = 0.5;
            data[6] = 1.0; data[7] = 2.0; data[8] = 3.0;

            // Act
            var kurtosis = StatisticsHelper.Kurtosis(data);

            // Assert - Should be close to 3.0 for normal distribution (excess kurtosis ≈ 0)
            Assert.True(kurtosis >= 2.0 && kurtosis <= 4.0);
        }

        [Fact]
        public void Mode_FindsMostFrequentValue()
        {
            // Arrange
            var data = new Vector<double>(10);
            data[0] = 1.0; data[1] = 2.0; data[2] = 2.0; data[3] = 3.0; data[4] = 3.0;
            data[5] = 3.0; data[6] = 4.0; data[7] = 4.0; data[8] = 5.0; data[9] = 5.0;
            // Mode = 3.0 (appears 3 times)

            // Act
            var mode = StatisticsHelper.Mode(data);

            // Assert
            Assert.Equal(3.0, mode, precision: 10);
        }

        [Fact]
        public void Range_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 1.0; data[1] = 3.0; data[2] = 5.0; data[3] = 7.0; data[4] = 9.0;
            // Range = Max - Min = 9.0 - 1.0 = 8.0

            // Act
            var range = StatisticsHelper.Range(data);

            // Assert
            Assert.Equal(8.0, range, precision: 10);
        }

        [Fact]
        public void GeometricMean_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(4);
            data[0] = 1.0; data[1] = 2.0; data[2] = 4.0; data[3] = 8.0;
            // Geometric Mean = (1 * 2 * 4 * 8)^(1/4) = 64^0.25 = 2.828427

            // Act
            var geoMean = StatisticsHelper.GeometricMean(data);

            // Assert
            Assert.Equal(2.8284271247461903, geoMean, precision: 10);
        }

        [Fact]
        public void HarmonicMean_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(3);
            data[0] = 1.0; data[1] = 2.0; data[2] = 4.0;
            // Harmonic Mean = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 ≈ 1.714286

            // Act
            var harmMean = StatisticsHelper.HarmonicMean(data);

            // Assert
            Assert.Equal(1.7142857142857142, harmMean, precision: 10);
        }

        [Fact]
        public void CoefficientOfVariation_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 2.0; data[1] = 4.0; data[2] = 6.0; data[3] = 8.0; data[4] = 10.0;
            // Mean = 6.0, Std Dev = 2.828427
            // CV = (Std Dev / Mean) * 100 = (2.828427 / 6.0) * 100 ≈ 47.14%

            // Act
            var cv = StatisticsHelper.CoefficientOfVariation(data);

            // Assert
            Assert.Equal(47.140452079103168, cv, precision: 8);
        }
    }
}
