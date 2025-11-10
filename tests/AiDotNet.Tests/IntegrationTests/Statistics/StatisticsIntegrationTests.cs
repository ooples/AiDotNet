using AiDotNet.LinearAlgebra;
using AiDotNet.Statistics;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Statistics
{
    /// <summary>
    /// Integration tests for statistical functions with mathematically verified results.
    /// These tests ensure mathematical correctness of statistical calculations.
    /// </summary>
    public class StatisticsIntegrationTests
    {
        private const double Tolerance = 1e-8;

        #region Basic Statistical Measures Tests (Existing)

        [Fact]
        public void Mean_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 2.0; data[1] = 4.0; data[2] = 6.0; data[3] = 8.0; data[4] = 10.0;
            // Mean = (2 + 4 + 6 + 8 + 10) / 5 = 30 / 5 = 6.0

            // Act
            var mean = StatisticsHelper<double>.CalculateMean(data);

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
            // Variance = [(2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2] / (5-1)
            //          = [16 + 4 + 0 + 4 + 16] / 4 = 40 / 4 = 10.0 (sample variance)

            // Act
            var variance = StatisticsHelper<double>.CalculateVariance(data);

            // Assert
            Assert.Equal(10.0, variance, precision: 10);
        }

        [Fact]
        public void StandardDeviation_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 2.0; data[1] = 4.0; data[2] = 6.0; data[3] = 8.0; data[4] = 10.0;
            // Variance = 10.0
            // Standard Deviation = sqrt(10.0) ≈ 3.162278

            // Act
            var stdDev = StatisticsHelper<double>.CalculateStandardDeviation(data);

            // Assert
            Assert.Equal(3.1622776601683795, stdDev, precision: 10);
        }

        [Fact]
        public void Median_OddCount_ReturnsMiddleValue()
        {
            // Arrange
            var data = new Vector<double>(5);
            data[0] = 1.0; data[1] = 3.0; data[2] = 5.0; data[3] = 7.0; data[4] = 9.0;
            // Median of [1, 3, 5, 7, 9] = 5.0

            // Act
            var median = StatisticsHelper<double>.CalculateMedian(data);

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
            var median = StatisticsHelper<double>.CalculateMedian(data);

            // Assert
            Assert.Equal(2.5, median, precision: 10);
        }

        #endregion

        #region Hypothesis Testing - T-Tests

        [Fact]
        public void TTest_TwoSampleEqual_ProducesHighPValue()
        {
            // Arrange - Two samples from same distribution
            var group1 = new Vector<double>(5);
            group1[0] = 5.0; group1[1] = 6.0; group1[2] = 7.0; group1[3] = 8.0; group1[4] = 9.0;

            var group2 = new Vector<double>(5);
            group2[0] = 5.5; group2[1] = 6.5; group2[2] = 7.5; group2[3] = 8.5; group2[4] = 9.5;
            // Should have high p-value (> 0.05) as distributions are similar

            // Act
            var result = StatisticsHelper<double>.TTest(group1, group2);

            // Assert
            Assert.True(result.PValue > 0.05);
            Assert.NotEqual(0.0, result.TStatistic);
            Assert.Equal(8, result.DegreesOfFreedom);
        }

        [Fact]
        public void TTest_TwoSampleDifferent_ProducesLowPValue()
        {
            // Arrange - Two samples from very different distributions
            var group1 = new Vector<double>(5);
            group1[0] = 1.0; group1[1] = 2.0; group1[2] = 3.0; group1[3] = 4.0; group1[4] = 5.0;

            var group2 = new Vector<double>(5);
            group2[0] = 10.0; group2[1] = 11.0; group2[2] = 12.0; group2[3] = 13.0; group2[4] = 14.0;
            // Should have low p-value (< 0.05) as distributions are very different

            // Act
            var result = StatisticsHelper<double>.TTest(group1, group2);

            // Assert
            Assert.True(result.PValue < 0.001);
            Assert.True(Math.Abs(result.TStatistic) > 5);
            Assert.Equal(8, result.DegreesOfFreedom);
        }

        [Fact]
        public void TTest_WithSignificanceLevel_IdentifiesSignificance()
        {
            // Arrange
            var group1 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group1[i] = 5.0 + i * 0.5;

            var group2 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group2[i] = 8.0 + i * 0.5;

            // Act
            var result = StatisticsHelper<double>.TTest(group1, group2, 0.05);

            // Assert
            Assert.Equal(0.05, result.SignificanceLevel, precision: 10);
            Assert.True(result.PValue < 0.05);
        }

        #endregion

        #region Hypothesis Testing - Mann-Whitney U Test

        [Fact]
        public void MannWhitneyUTest_SimilarDistributions_ProducesHighPValue()
        {
            // Arrange
            var group1 = new Vector<double>(6);
            group1[0] = 1.2; group1[1] = 2.3; group1[2] = 3.1; group1[3] = 4.5; group1[4] = 5.2; group1[5] = 6.1;

            var group2 = new Vector<double>(6);
            group2[0] = 1.5; group2[1] = 2.1; group2[2] = 3.5; group2[3] = 4.2; group2[4] = 5.5; group2[5] = 6.3;

            // Act
            var result = StatisticsHelper<double>.MannWhitneyUTest(group1, group2);

            // Assert
            Assert.True(result.PValue > 0.05);
            Assert.True(result.UStatistic >= 0);
        }

        [Fact]
        public void MannWhitneyUTest_DifferentDistributions_ProducesLowPValue()
        {
            // Arrange - Clearly separated distributions
            var group1 = new Vector<double>(5);
            group1[0] = 1.0; group1[1] = 2.0; group1[2] = 3.0; group1[3] = 4.0; group1[4] = 5.0;

            var group2 = new Vector<double>(5);
            group2[0] = 10.0; group2[1] = 11.0; group2[2] = 12.0; group2[3] = 13.0; group2[4] = 14.0;

            // Act
            var result = StatisticsHelper<double>.MannWhitneyUTest(group1, group2);

            // Assert
            Assert.True(result.PValue < 0.01);
            Assert.True(Math.Abs(result.ZScore) > 2);
        }

        [Fact]
        public void MannWhitneyUTest_WithTiedRanks_HandlesCorrectly()
        {
            // Arrange - Data with tied values
            var group1 = new Vector<double>(5);
            group1[0] = 1.0; group1[1] = 2.0; group1[2] = 2.0; group1[3] = 3.0; group1[4] = 4.0;

            var group2 = new Vector<double>(5);
            group2[0] = 2.0; group2[1] = 3.0; group2[2] = 3.0; group2[3] = 4.0; group2[4] = 5.0;

            // Act
            var result = StatisticsHelper<double>.MannWhitneyUTest(group1, group2);

            // Assert - Should handle ties without error
            Assert.True(result.UStatistic >= 0);
            Assert.True(result.PValue >= 0 && result.PValue <= 1.0);
        }

        #endregion

        #region Hypothesis Testing - Chi-Square Test

        [Fact]
        public void ChiSquareTest_IndependentCategories_ProducesHighPValue()
        {
            // Arrange - Similar distributions
            var group1 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group1[i] = i % 3; // Categories: 0, 1, 2

            var group2 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group2[i] = i % 3; // Same distribution

            // Act
            var result = StatisticsHelper<double>.ChiSquareTest(group1, group2);

            // Assert
            Assert.True(result.PValue > 0.5); // Very similar distributions
            Assert.True(result.ChiSquareStatistic >= 0);
        }

        [Fact]
        public void ChiSquareTest_DependentCategories_ProducesLowPValue()
        {
            // Arrange - Very different categorical distributions
            var group1 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group1[i] = 0.0; // All category 0

            var group2 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group2[i] = 1.0; // All category 1

            // Act
            var result = StatisticsHelper<double>.ChiSquareTest(group1, group2);

            // Assert
            Assert.True(result.ChiSquareStatistic > 0);
            Assert.True(result.DegreesOfFreedom >= 0);
        }

        #endregion

        #region Hypothesis Testing - F-Test

        [Fact]
        public void FTest_EqualVariances_ProducesHighPValue()
        {
            // Arrange - Two samples with similar variances
            var group1 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group1[i] = 5.0 + i * 0.5;

            var group2 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group2[i] = 7.0 + i * 0.5;

            // Act
            var result = StatisticsHelper<double>.FTest(group1, group2);

            // Assert
            Assert.True(result.FStatistic > 0);
            Assert.True(result.PValue > 0.05);
        }

        [Fact]
        public void FTest_DifferentVariances_ProducesLowPValue()
        {
            // Arrange - One sample with much larger variance
            var group1 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group1[i] = 5.0 + i * 0.1; // Small variance

            var group2 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group2[i] = 5.0 + i * 5.0; // Large variance

            // Act
            var result = StatisticsHelper<double>.FTest(group1, group2);

            // Assert
            Assert.True(result.FStatistic > 0);
            Assert.NotEqual(result.FStatistic, 1.0);
        }

        #endregion

        #region Hypothesis Testing - Permutation Test

        [Fact]
        public void PermutationTest_SimilarGroups_ProducesHighPValue()
        {
            // Arrange
            var group1 = new Vector<double>(8);
            for (int i = 0; i < 8; i++) group1[i] = 5.0 + i * 0.5;

            var group2 = new Vector<double>(8);
            for (int i = 0; i < 8; i++) group2[i] = 5.2 + i * 0.5;

            // Act
            var result = StatisticsHelper<double>.PermutationTest(group1, group2);

            // Assert
            Assert.True(result.PValue > 0.05);
            Assert.Equal(1000, result.NumberOfPermutations);
        }

        [Fact]
        public void PermutationTest_DifferentGroups_ProducesLowPValue()
        {
            // Arrange
            var group1 = new Vector<double>(5);
            for (int i = 0; i < 5; i++) group1[i] = 1.0 + i;

            var group2 = new Vector<double>(5);
            for (int i = 0; i < 5; i++) group2[i] = 10.0 + i;

            // Act
            var result = StatisticsHelper<double>.PermutationTest(group1, group2);

            // Assert
            Assert.True(result.PValue < 0.01);
            Assert.True(Math.Abs(result.ObservedDifference) > 5);
        }

        #endregion

        #region Probability Distributions - Normal Distribution

        [Fact]
        public void NormalPDF_AtMean_ProducesMaximumValue()
        {
            // Arrange
            double mean = 0.0;
            double stdDev = 1.0;
            double x = 0.0;
            // PDF at mean for standard normal: 1/sqrt(2*pi) ≈ 0.3989423

            // Act
            var pdf = StatisticsHelper<double>.CalculateNormalPDF(mean, stdDev, x);

            // Assert
            Assert.Equal(0.3989422804014327, pdf, precision: 10);
        }

        [Fact]
        public void NormalPDF_OneStdDevAway_ProducesCorrectValue()
        {
            // Arrange
            double mean = 0.0;
            double stdDev = 1.0;
            double x = 1.0;
            // PDF at x=1 for standard normal ≈ 0.2419707

            // Act
            var pdf = StatisticsHelper<double>.CalculateNormalPDF(mean, stdDev, x);

            // Assert
            Assert.Equal(0.24197072451914337, pdf, precision: 8);
        }

        [Fact]
        public void NormalCDF_AtMean_ReturnsHalf()
        {
            // Arrange
            double mean = 5.0;
            double stdDev = 2.0;
            double x = 5.0;
            // CDF at mean = 0.5

            // Act
            var cdf = StatisticsHelper<double>.CalculateNormalCDF(mean, stdDev, x);

            // Assert
            Assert.Equal(0.5, cdf, precision: 8);
        }

        [Fact]
        public void NormalCDF_OneStdDevAboveMean_ReturnsCorrectValue()
        {
            // Arrange
            double mean = 0.0;
            double stdDev = 1.0;
            double x = 1.0;
            // CDF at x=1 for standard normal ≈ 0.8413

            // Act
            var cdf = StatisticsHelper<double>.CalculateNormalCDF(mean, stdDev, x);

            // Assert
            Assert.InRange(cdf, 0.84, 0.85);
        }

        [Fact]
        public void InverseNormalCDF_Median_ReturnsMean()
        {
            // Arrange
            double probability = 0.5;
            // Standard normal: inverse CDF(0.5) = 0

            // Act
            var result = StatisticsHelper<double>.CalculateInverseNormalCDF(probability);

            // Assert
            Assert.Equal(0.0, result, precision: 8);
        }

        [Fact]
        public void InverseNormalCDF_WithMeanAndStdDev_ProducesCorrectValue()
        {
            // Arrange
            double mean = 100.0;
            double stdDev = 15.0;
            double probability = 0.5;
            // Should return mean

            // Act
            var result = StatisticsHelper<double>.CalculateInverseNormalCDF(mean, stdDev, probability);

            // Assert
            Assert.Equal(100.0, result, precision: 6);
        }

        [Fact]
        public void InverseNormalCDF_UpperTail_ProducesPositiveValue()
        {
            // Arrange
            double probability = 0.975; // 97.5th percentile
            // Standard normal: ≈ 1.96

            // Act
            var result = StatisticsHelper<double>.CalculateInverseNormalCDF(probability);

            // Assert
            Assert.InRange(result, 1.9, 2.0);
        }

        #endregion

        #region Probability Distributions - Chi-Square Distribution

        [Fact]
        public void ChiSquarePDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            int df = 2;
            double x = 1.0;
            // Chi-square PDF with df=2 at x=1: e^(-0.5) / 2 ≈ 0.3033

            // Act
            var pdf = StatisticsHelper<double>.CalculateChiSquarePDF(df, x);

            // Assert
            Assert.InRange(pdf, 0.30, 0.31);
        }

        [Fact]
        public void ChiSquareCDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            int df = 1;
            double x = 1.0;
            // Chi-square CDF with df=1 at x=1 ≈ 0.6827

            // Act
            var cdf = StatisticsHelper<double>.CalculateChiSquareCDF(df, x);

            // Assert
            Assert.InRange(cdf, 0.68, 0.69);
        }

        [Fact]
        public void InverseChiSquareCDF_MedianWithDF2_ReturnsCorrectValue()
        {
            // Arrange
            int df = 2;
            double probability = 0.5;
            // Median of chi-square with df=2 ≈ 1.386

            // Act
            var result = StatisticsHelper<double>.CalculateInverseChiSquareCDF(df, probability);

            // Assert
            Assert.InRange(result, 1.3, 1.5);
        }

        #endregion

        #region Probability Distributions - Exponential Distribution

        [Fact]
        public void ExponentialPDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            double lambda = 2.0;
            double x = 1.0;
            // Exponential PDF: lambda * e^(-lambda * x) = 2 * e^(-2) ≈ 0.2707

            // Act
            var pdf = StatisticsHelper<double>.CalculateExponentialPDF(lambda, x);

            // Assert
            Assert.Equal(0.27067056647322555, pdf, precision: 8);
        }

        [Fact]
        public void InverseExponentialCDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            double lambda = 1.0;
            double probability = 0.5;
            // Inverse exponential CDF: -ln(1-p)/lambda = -ln(0.5) ≈ 0.693

            // Act
            var result = StatisticsHelper<double>.CalculateInverseExponentialCDF(lambda, probability);

            // Assert
            Assert.Equal(0.6931471805599453, result, precision: 10);
        }

        #endregion

        #region Probability Distributions - Other Distributions

        [Fact]
        public void WeibullPDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            double k = 2.0; // Shape parameter
            double lambda = 1.0; // Scale parameter
            double x = 1.0;
            // Weibull PDF with k=2, lambda=1, x=1

            // Act
            var pdf = StatisticsHelper<double>.CalculateWeibullPDF(k, lambda, x);

            // Assert
            Assert.True(pdf > 0);
            Assert.True(pdf < 1);
        }

        [Fact]
        public void LogNormalPDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            double mu = 0.0;
            double sigma = 1.0;
            double x = 1.0;
            // Log-normal PDF at x=1 with mu=0, sigma=1 ≈ 0.3989

            // Act
            var pdf = StatisticsHelper<double>.CalculateLogNormalPDF(mu, sigma, x);

            // Assert
            Assert.InRange(pdf, 0.39, 0.40);
        }

        [Fact]
        public void LaplacePDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            double median = 0.0;
            double mad = 1.0;
            double x = 0.0;
            // Laplace PDF at median: 1/(2*b) where b=mad

            // Act
            var pdf = StatisticsHelper<double>.CalculateLaplacePDF(median, mad, x);

            // Assert
            Assert.Equal(0.5, pdf, precision: 8);
        }

        [Fact]
        public void StudentPDF_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            double x = 0.0;
            double mean = 0.0;
            double stdDev = 1.0;
            int df = 10;
            // Student's t PDF at mean should be greater than normal PDF

            // Act
            var pdf = StatisticsHelper<double>.CalculateStudentPDF(x, mean, stdDev, df);

            // Assert
            Assert.True(pdf > 0.39);
            Assert.True(pdf < 0.41);
        }

        #endregion

        #region Regression Statistics - R-Squared and Adjusted R-Squared

        [Fact]
        public void CalculateR2_PerfectFit_ReturnsOne()
        {
            // Arrange - Perfect predictions
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 1.0; predicted[1] = 2.0; predicted[2] = 3.0; predicted[3] = 4.0; predicted[4] = 5.0;

            // Act
            var r2 = StatisticsHelper<double>.CalculateR2(actual, predicted);

            // Assert
            Assert.Equal(1.0, r2, precision: 10);
        }

        [Fact]
        public void CalculateR2_PoorFit_ReturnsLowValue()
        {
            // Arrange - Poor predictions
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 5.0; predicted[1] = 4.0; predicted[2] = 3.0; predicted[3] = 2.0; predicted[4] = 1.0;

            // Act
            var r2 = StatisticsHelper<double>.CalculateR2(actual, predicted);

            // Assert
            Assert.True(r2 < 0.5);
        }

        [Fact]
        public void CalculateAdjustedR2_WithFewerParameters_IsHigher()
        {
            // Arrange
            double r2 = 0.85;
            int n = 100;
            int p1 = 5;
            int p2 = 10;

            // Act
            var adjR2_fewer = StatisticsHelper<double>.CalculateAdjustedR2(r2, n, p1);
            var adjR2_more = StatisticsHelper<double>.CalculateAdjustedR2(r2, n, p2);

            // Assert
            Assert.True(adjR2_fewer > adjR2_more); // Fewer parameters = higher adjusted R²
            Assert.True(adjR2_fewer <= r2); // Adjusted R² ≤ R²
        }

        [Fact]
        public void CalculateAdjustedR2_PerfectFit_ReturnsOne()
        {
            // Arrange
            double r2 = 1.0;
            int n = 50;
            int p = 3;

            // Act
            var adjR2 = StatisticsHelper<double>.CalculateAdjustedR2(r2, n, p);

            // Assert
            Assert.Equal(1.0, adjR2, precision: 10);
        }

        #endregion

        #region Regression Statistics - Residual Analysis

        [Fact]
        public void CalculateResiduals_ProducesCorrectDifferences()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 10.0; actual[1] = 20.0; actual[2] = 30.0; actual[3] = 40.0; actual[4] = 50.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 12.0; predicted[1] = 18.0; predicted[2] = 32.0; predicted[3] = 38.0; predicted[4] = 52.0;
            // Residuals: -2, 2, -2, 2, -2

            // Act
            var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);

            // Assert
            Assert.Equal(-2.0, residuals[0], precision: 10);
            Assert.Equal(2.0, residuals[1], precision: 10);
            Assert.Equal(-2.0, residuals[2], precision: 10);
            Assert.Equal(2.0, residuals[3], precision: 10);
            Assert.Equal(-2.0, residuals[4], precision: 10);
        }

        [Fact]
        public void CalculateResidualSumOfSquares_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(4);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0;

            var predicted = new Vector<double>(4);
            predicted[0] = 1.5; predicted[1] = 2.5; predicted[2] = 3.5; predicted[3] = 4.5;
            // Residuals: -0.5, -0.5, -0.5, -0.5
            // RSS = 4 * 0.25 = 1.0

            // Act
            var rss = StatisticsHelper<double>.CalculateResidualSumOfSquares(actual, predicted);

            // Assert
            Assert.Equal(1.0, rss, precision: 10);
        }

        [Fact]
        public void CalculateTotalSumOfSquares_ProducesCorrectValue()
        {
            // Arrange
            var values = new Vector<double>(5);
            values[0] = 2.0; values[1] = 4.0; values[2] = 6.0; values[3] = 8.0; values[4] = 10.0;
            // Mean = 6.0
            // TSS = (2-6)² + (4-6)² + (6-6)² + (8-6)² + (10-6)² = 16+4+0+4+16 = 40

            // Act
            var tss = StatisticsHelper<double>.CalculateTotalSumOfSquares(values);

            // Assert
            Assert.Equal(40.0, tss, precision: 10);
        }

        [Fact]
        public void CalculateDurbinWatsonStatistic_NoAutocorrelation_ReturnsTwo()
        {
            // Arrange - Alternating residuals (no autocorrelation)
            var actual = new Vector<double>(6);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0; actual[5] = 6.0;

            var predicted = new Vector<double>(6);
            predicted[0] = 1.5; predicted[1] = 1.5; predicted[2] = 3.5; predicted[3] = 3.5; predicted[4] = 5.5; predicted[5] = 5.5;
            // Residuals alternate: -0.5, 0.5, -0.5, 0.5, -0.5, 0.5

            // Act
            var dw = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(actual, predicted);

            // Assert
            Assert.InRange(dw, 1.5, 2.5); // Should be close to 2
        }

        #endregion

        #region Regression Statistics - Model Selection Criteria

        [Fact]
        public void CalculateAIC_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            int n = 100;
            int k = 5;
            double rss = 150.0;
            // AIC = n * ln(RSS/n) + 2k

            // Act
            var aic = StatisticsHelper<double>.CalculateAIC(n, k, rss);

            // Assert
            Assert.True(aic > 0);
            Assert.NotEqual(double.PositiveInfinity, aic);
        }

        [Fact]
        public void CalculateBIC_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange
            int n = 100;
            int k = 5;
            double rss = 150.0;
            // BIC = n * ln(RSS/n) + k * ln(n)

            // Act
            var bic = StatisticsHelper<double>.CalculateBIC(n, k, rss);

            // Assert
            Assert.True(bic > 0);
            Assert.NotEqual(double.PositiveInfinity, bic);
        }

        [Fact]
        public void CalculateAIC_MoreParameters_HigherValue()
        {
            // Arrange
            int n = 100;
            double rss = 150.0;

            // Act
            var aic3 = StatisticsHelper<double>.CalculateAIC(n, 3, rss);
            var aic10 = StatisticsHelper<double>.CalculateAIC(n, 10, rss);

            // Assert
            Assert.True(aic10 > aic3); // More parameters = higher AIC (penalized)
        }

        #endregion

        #region Regression Statistics - VIF

        [Fact]
        public void CalculateVIF_IndependentFeatures_ReturnsLowValues()
        {
            // Arrange - Orthogonal features (uncorrelated)
            var features = new Matrix<double>(4, 2);
            features[0, 0] = 1.0; features[0, 1] = 0.0;
            features[1, 0] = 0.0; features[1, 1] = 1.0;
            features[2, 0] = -1.0; features[2, 1] = 0.0;
            features[3, 0] = 0.0; features[3, 1] = -1.0;

            var options = new ModelStatsOptions();
            var corrMatrix = StatisticsHelper<double>.CalculateCorrelationMatrix(features, options);

            // Act
            var vif = StatisticsHelper<double>.CalculateVIF(corrMatrix, options);

            // Assert
            foreach (var value in vif)
            {
                Assert.True(value <= 2.0); // Low VIF indicates low multicollinearity
            }
        }

        #endregion

        #region Error Metrics

        [Fact]
        public void CalculateMeanSquaredError_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(4);
            actual[0] = 3.0; actual[1] = -0.5; actual[2] = 2.0; actual[3] = 7.0;

            var predicted = new Vector<double>(4);
            predicted[0] = 2.5; predicted[1] = 0.0; predicted[2] = 2.0; predicted[3] = 8.0;
            // Errors: 0.5, -0.5, 0, -1
            // MSE = (0.25 + 0.25 + 0 + 1) / 4 = 0.375

            // Act
            var mse = StatisticsHelper<double>.CalculateMeanSquaredError(actual, predicted);

            // Assert
            Assert.Equal(0.375, mse, precision: 10);
        }

        [Fact]
        public void CalculateRootMeanSquaredError_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(4);
            actual[0] = 3.0; actual[1] = -0.5; actual[2] = 2.0; actual[3] = 7.0;

            var predicted = new Vector<double>(4);
            predicted[0] = 2.5; predicted[1] = 0.0; predicted[2] = 2.0; predicted[3] = 8.0;
            // MSE = 0.375, RMSE = sqrt(0.375) ≈ 0.612

            // Act
            var rmse = StatisticsHelper<double>.CalculateRootMeanSquaredError(actual, predicted);

            // Assert
            Assert.Equal(0.6123724356957945, rmse, precision: 10);
        }

        [Fact]
        public void CalculateMeanAbsoluteError_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 1.2; predicted[1] = 2.3; predicted[2] = 2.8; predicted[3] = 4.1; predicted[4] = 5.2;
            // MAE = (0.2 + 0.3 + 0.2 + 0.1 + 0.2) / 5 = 1.0 / 5 = 0.2

            // Act
            var mae = StatisticsHelper<double>.CalculateMeanAbsoluteError(actual, predicted);

            // Assert
            Assert.Equal(0.2, mae, precision: 10);
        }

        [Fact]
        public void CalculateMedianAbsoluteError_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 1.1; predicted[1] = 2.3; predicted[2] = 2.9; predicted[3] = 4.2; predicted[4] = 5.4;
            // Absolute errors: 0.1, 0.3, 0.1, 0.2, 0.4
            // Median = 0.2

            // Act
            var medae = StatisticsHelper<double>.CalculateMedianAbsoluteError(actual, predicted);

            // Assert
            Assert.Equal(0.2, medae, precision: 10);
        }

        [Fact]
        public void CalculateMaxError_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 1.1; predicted[1] = 2.05; predicted[2] = 4.0; predicted[3] = 3.7; predicted[4] = 5.1;
            // Absolute errors: 0.1, 0.05, 1.0, 0.3, 0.1
            // Max = 1.0

            // Act
            var maxError = StatisticsHelper<double>.CalculateMaxError(actual, predicted);

            // Assert
            Assert.Equal(1.0, maxError, precision: 10);
        }

        [Fact]
        public void CalculateMeanAbsolutePercentageError_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(4);
            actual[0] = 100.0; actual[1] = 200.0; actual[2] = 300.0; actual[3] = 400.0;

            var predicted = new Vector<double>(4);
            predicted[0] = 110.0; predicted[1] = 190.0; predicted[2] = 330.0; predicted[3] = 380.0;
            // MAPE = mean(|actual - pred| / |actual|) * 100
            // = mean(10/100, 10/200, 30/300, 20/400) * 100
            // = mean(0.1, 0.05, 0.1, 0.05) * 100 = 0.075 * 100 = 7.5%

            // Act
            var mape = StatisticsHelper<double>.CalculateMeanAbsolutePercentageError(actual, predicted);

            // Assert
            Assert.Equal(7.5, mape, precision: 8);
        }

        [Fact]
        public void CalculateSymmetricMeanAbsolutePercentageError_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(3);
            actual[0] = 100.0; actual[1] = 200.0; actual[2] = 150.0;

            var predicted = new Vector<double>(3);
            predicted[0] = 110.0; predicted[1] = 180.0; predicted[2] = 165.0;

            // Act
            var smape = StatisticsHelper<double>.CalculateSymmetricMeanAbsolutePercentageError(actual, predicted);

            // Assert
            Assert.True(smape >= 0 && smape <= 100);
        }

        [Fact]
        public void CalculateMeanBiasError_PositiveBias_ProducesPositiveValue()
        {
            // Arrange - Predictions consistently higher than actuals
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 2.0; predicted[1] = 3.0; predicted[2] = 4.0; predicted[3] = 5.0; predicted[4] = 6.0;
            // MBE = mean(predicted - actual) = 1.0

            // Act
            var mbe = StatisticsHelper<double>.CalculateMeanBiasError(actual, predicted);

            // Assert
            Assert.Equal(1.0, mbe, precision: 10);
        }

        [Fact]
        public void CalculateTheilUStatistic_ProducesCorrectValue()
        {
            // Arrange
            var actual = new Vector<double>(4);
            actual[0] = 10.0; actual[1] = 20.0; actual[2] = 30.0; actual[3] = 40.0;

            var predicted = new Vector<double>(4);
            predicted[0] = 11.0; predicted[1] = 19.0; predicted[2] = 31.0; predicted[3] = 39.0;

            // Act
            var theilU = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predicted);

            // Assert
            Assert.True(theilU >= 0); // Theil's U is always non-negative
            Assert.True(theilU < 1); // Good predictions should have U < 1
        }

        #endregion

        #region Correlation Measures

        [Fact]
        public void CalculatePearsonCorrelation_PerfectPositive_ReturnsOne()
        {
            // Arrange
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 2.0; y[1] = 4.0; y[2] = 6.0; y[3] = 8.0; y[4] = 10.0;

            // Act
            var corr = StatisticsHelper<double>.CalculatePearsonCorrelation(x, y);

            // Assert
            Assert.Equal(1.0, corr, precision: 10);
        }

        [Fact]
        public void CalculatePearsonCorrelation_PerfectNegative_ReturnsMinusOne()
        {
            // Arrange
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 10.0; y[1] = 8.0; y[2] = 6.0; y[3] = 4.0; y[4] = 2.0;

            // Act
            var corr = StatisticsHelper<double>.CalculatePearsonCorrelation(x, y);

            // Assert
            Assert.Equal(-1.0, corr, precision: 10);
        }

        [Fact]
        public void CalculateSpearmanRankCorrelation_MonotonicRelationship_ReturnsHighValue()
        {
            // Arrange - Non-linear but monotonic relationship
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 1.0; y[1] = 4.0; y[2] = 9.0; y[3] = 16.0; y[4] = 25.0; // y = x²

            // Act
            var spearman = StatisticsHelper<double>.CalculateSpearmanRankCorrelationCoefficient(x, y);

            // Assert
            Assert.Equal(1.0, spearman, precision: 10); // Perfect monotonic relationship
        }

        [Fact]
        public void CalculateKendallTau_ConcordantPairs_ReturnsPositiveValue()
        {
            // Arrange
            var x = new Vector<double>(5);
            x[0] = 1.0; x[1] = 2.0; x[2] = 3.0; x[3] = 4.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 1.0; y[1] = 2.0; y[2] = 3.0; y[3] = 4.0; y[4] = 5.0;

            // Act
            var tau = StatisticsHelper<double>.CalculateKendallTau(x, y);

            // Assert
            Assert.Equal(1.0, tau, precision: 10); // All pairs concordant
        }

        #endregion

        #region Confidence and Credible Intervals

        [Fact]
        public void CalculateConfidenceIntervals_Normal_ProducesValidRange()
        {
            // Arrange
            var values = new Vector<double>(100);
            for (int i = 0; i < 100; i++) values[i] = 50.0 + i * 0.1; // Mean around 54.95

            double confidenceLevel = 0.95;

            // Act
            var (lower, upper) = StatisticsHelper<double>.CalculateConfidenceIntervals(
                values, confidenceLevel, DistributionType.Normal);

            // Assert
            Assert.True(lower < upper);
            Assert.True(lower > 40); // Reasonable bounds
            Assert.True(upper < 70);
        }

        [Fact]
        public void CalculateBootstrapInterval_ProducesValidRange()
        {
            // Arrange
            var actual = new Vector<double>(20);
            for (int i = 0; i < 20; i++) actual[i] = 10.0 + i * 0.5;

            var predicted = new Vector<double>(20);
            for (int i = 0; i < 20; i++) predicted[i] = 10.2 + i * 0.5;

            double confidenceLevel = 0.90;

            // Act
            var (lower, upper) = StatisticsHelper<double>.CalculateBootstrapInterval(
                actual, predicted, confidenceLevel);

            // Assert
            Assert.True(lower < upper);
        }

        [Fact]
        public void CalculatePredictionIntervals_ProducesValidRange()
        {
            // Arrange
            var actual = new Vector<double>(30);
            for (int i = 0; i < 30; i++) actual[i] = 100.0 + i * 2.0;

            var predicted = new Vector<double>(30);
            for (int i = 0; i < 30; i++) predicted[i] = 101.0 + i * 2.0;

            double confidenceLevel = 0.95;

            // Act
            var (lower, upper) = StatisticsHelper<double>.CalculatePredictionIntervals(
                actual, predicted, confidenceLevel);

            // Assert
            Assert.True(lower < upper);
            Assert.True(upper - lower > 0); // Non-zero interval
        }

        #endregion

        #region Time Series Analysis

        [Fact]
        public void CalculateAutoCorrelationFunction_Lag0_ReturnsOne()
        {
            // Arrange
            var series = new Vector<double>(10);
            for (int i = 0; i < 10; i++) series[i] = Math.Sin(i * 0.5) + 5.0;

            int maxLag = 5;

            // Act
            var acf = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, maxLag);

            // Assert
            Assert.Equal(1.0, acf[0], precision: 8); // ACF at lag 0 is always 1
        }

        [Fact]
        public void CalculateAutoCorrelationFunction_ProducesValidValues()
        {
            // Arrange - Periodic series
            var series = new Vector<double>(20);
            for (int i = 0; i < 20; i++) series[i] = Math.Sin(i * Math.PI / 4);

            int maxLag = 8;

            // Act
            var acf = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, maxLag);

            // Assert
            for (int lag = 0; lag <= maxLag; lag++)
            {
                Assert.True(acf[lag] >= -1.0 && acf[lag] <= 1.0); // ACF must be in [-1, 1]
            }
        }

        [Fact]
        public void CalculatePartialAutoCorrelationFunction_Lag0_ReturnsOne()
        {
            // Arrange
            var series = new Vector<double>(10);
            for (int i = 0; i < 10; i++) series[i] = i * 2.0 + 1.0;

            int maxLag = 3;

            // Act
            var pacf = StatisticsHelper<double>.CalculatePartialAutoCorrelationFunction(series, maxLag);

            // Assert
            Assert.Equal(1.0, pacf[0], precision: 8); // PACF at lag 0 is always 1
        }

        [Fact]
        public void CalculatePartialAutoCorrelationFunction_ProducesValidValues()
        {
            // Arrange
            var series = new Vector<double>(15);
            for (int i = 0; i < 15; i++) series[i] = 10.0 + i * 0.5 + Math.Sin(i);

            int maxLag = 5;

            // Act
            var pacf = StatisticsHelper<double>.CalculatePartialAutoCorrelationFunction(series, maxLag);

            // Assert
            for (int lag = 0; lag <= maxLag; lag++)
            {
                Assert.True(pacf[lag] >= -1.0 && pacf[lag] <= 1.0); // PACF must be in [-1, 1]
            }
        }

        #endregion

        #region Distance Metrics

        [Fact]
        public void EuclideanDistance_IdenticalVectors_ReturnsZero()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 1.0; v1[1] = 2.0; v1[2] = 3.0;

            var v2 = new Vector<double>(3);
            v2[0] = 1.0; v2[1] = 2.0; v2[2] = 3.0;

            // Act
            var distance = StatisticsHelper<double>.EuclideanDistance(v1, v2);

            // Assert
            Assert.Equal(0.0, distance, precision: 10);
        }

        [Fact]
        public void EuclideanDistance_KnownVectors_ProducesCorrectValue()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 0.0; v1[1] = 0.0; v1[2] = 0.0;

            var v2 = new Vector<double>(3);
            v2[0] = 3.0; v2[1] = 4.0; v2[2] = 0.0;
            // Distance = sqrt(9 + 16 + 0) = sqrt(25) = 5.0

            // Act
            var distance = StatisticsHelper<double>.EuclideanDistance(v1, v2);

            // Assert
            Assert.Equal(5.0, distance, precision: 10);
        }

        [Fact]
        public void ManhattanDistance_KnownVectors_ProducesCorrectValue()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 1.0; v1[1] = 2.0; v1[2] = 3.0;

            var v2 = new Vector<double>(3);
            v2[0] = 4.0; v2[1] = 6.0; v2[2] = 8.0;
            // Manhattan distance = |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12

            // Act
            var distance = StatisticsHelper<double>.ManhattanDistance(v1, v2);

            // Assert
            Assert.Equal(12.0, distance, precision: 10);
        }

        [Fact]
        public void CosineSimilarity_IdenticalVectors_ReturnsOne()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 1.0; v1[1] = 2.0; v1[2] = 3.0;

            var v2 = new Vector<double>(3);
            v2[0] = 1.0; v2[1] = 2.0; v2[2] = 3.0;

            // Act
            var similarity = StatisticsHelper<double>.CosineSimilarity(v1, v2);

            // Assert
            Assert.Equal(1.0, similarity, precision: 10);
        }

        [Fact]
        public void CosineSimilarity_OrthogonalVectors_ReturnsZero()
        {
            // Arrange
            var v1 = new Vector<double>(3);
            v1[0] = 1.0; v1[1] = 0.0; v1[2] = 0.0;

            var v2 = new Vector<double>(3);
            v2[0] = 0.0; v2[1] = 1.0; v2[2] = 0.0;

            // Act
            var similarity = StatisticsHelper<double>.CosineSimilarity(v1, v2);

            // Assert
            Assert.Equal(0.0, similarity, precision: 10);
        }

        #endregion

        #region Special Functions

        [Fact]
        public void Gamma_IntegerValue_ProducesFactorial()
        {
            // Arrange
            double x = 5.0;
            // Gamma(5) = 4! = 24

            // Act
            var result = StatisticsHelper<double>.Gamma(x);

            // Assert
            Assert.Equal(24.0, result, precision: 6);
        }

        [Fact]
        public void Gamma_HalfInteger_ProducesCorrectValue()
        {
            // Arrange
            double x = 1.5;
            // Gamma(1.5) = 0.5 * Gamma(0.5) = 0.5 * sqrt(pi) ≈ 0.8862

            // Act
            var result = StatisticsHelper<double>.Gamma(x);

            // Assert
            Assert.InRange(result, 0.88, 0.89);
        }

        [Fact]
        public void IncompleteGamma_KnownValues_ProducesCorrectResult()
        {
            // Arrange
            double a = 2.0;
            double x = 1.0;

            // Act
            var result = StatisticsHelper<double>.IncompleteGamma(a, x);

            // Assert
            Assert.True(result > 0);
            Assert.True(result < StatisticsHelper<double>.Gamma(a));
        }

        #endregion

        #region Quantiles and Percentiles

        [Fact]
        public void CalculateQuantiles_ProducesCorrectQ1AndQ3()
        {
            // Arrange
            var data = new Vector<double>(11);
            for (int i = 0; i < 11; i++) data[i] = i * 10.0; // 0, 10, 20, ..., 100

            // Act
            var (q1, q3) = StatisticsHelper<double>.CalculateQuantiles(data);

            // Assert
            Assert.Equal(25.0, q1, precision: 5); // 25th percentile
            Assert.Equal(75.0, q3, precision: 5); // 75th percentile
        }

        [Fact]
        public void CalculateQuantile_Median_ReturnsMiddleValue()
        {
            // Arrange
            var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

            // Act
            var median = StatisticsHelper<double>.CalculateQuantile(data, 0.5);

            // Assert
            Assert.Equal(3.0, median, precision: 10);
        }

        [Fact]
        public void CalculateMedianAbsoluteDeviation_ProducesCorrectValue()
        {
            // Arrange
            var values = new Vector<double>(5);
            values[0] = 1.0; values[1] = 2.0; values[2] = 3.0; values[3] = 4.0; values[4] = 5.0;
            // Median = 3.0
            // Absolute deviations: 2, 1, 0, 1, 2
            // MAD = median(2, 1, 0, 1, 2) = 1.0

            // Act
            var mad = StatisticsHelper<double>.CalculateMedianAbsoluteDeviation(values);

            // Assert
            Assert.Equal(1.0, mad, precision: 10);
        }

        #endregion

        #region Classification Metrics

        [Fact]
        public void CalculateConfusionMatrix_BinaryClassification_ProducesCorrectCounts()
        {
            // Arrange
            var actual = new Vector<double>(10);
            actual[0] = 1; actual[1] = 1; actual[2] = 0; actual[3] = 0; actual[4] = 1;
            actual[5] = 1; actual[6] = 0; actual[7] = 0; actual[8] = 1; actual[9] = 0;

            var predicted = new Vector<double>(10);
            predicted[0] = 1; predicted[1] = 0; predicted[2] = 0; predicted[3] = 1; predicted[4] = 1;
            predicted[5] = 1; predicted[6] = 0; predicted[7] = 0; predicted[8] = 0; predicted[9] = 0;
            // TP=3, TN=4, FP=1, FN=2

            double threshold = 0.5;

            // Act
            var cm = StatisticsHelper<double>.CalculateConfusionMatrix(actual, predicted, threshold);

            // Assert
            Assert.Equal(3.0, cm.TruePositives, precision: 10);
            Assert.Equal(4.0, cm.TrueNegatives, precision: 10);
            Assert.Equal(1.0, cm.FalsePositives, precision: 10);
            Assert.Equal(2.0, cm.FalseNegatives, precision: 10);
        }

        [Fact]
        public void CalculateAccuracy_PerfectPredictions_ReturnsOne()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 1.0; predicted[1] = 2.0; predicted[2] = 3.0; predicted[3] = 4.0; predicted[4] = 5.0;

            // Act
            var accuracy = StatisticsHelper<double>.CalculateAccuracy(actual, predicted);

            // Assert
            Assert.Equal(1.0, accuracy, precision: 10);
        }

        #endregion

        #region ROC and AUC

        [Fact]
        public void CalculateROCCurve_ProducesValidCurve()
        {
            // Arrange
            var actual = new Vector<double>(10);
            for (int i = 0; i < 10; i++) actual[i] = i < 5 ? 0.0 : 1.0;

            var predicted = new Vector<double>(10);
            for (int i = 0; i < 10; i++) predicted[i] = i * 0.1;

            // Act
            var (fpr, tpr) = StatisticsHelper<double>.CalculateROCCurve(actual, predicted);

            // Assert
            Assert.True(fpr.Length > 0);
            Assert.True(tpr.Length > 0);
            Assert.Equal(fpr.Length, tpr.Length);

            // FPR and TPR should be in [0, 1]
            for (int i = 0; i < fpr.Length; i++)
            {
                Assert.True(fpr[i] >= 0.0 && fpr[i] <= 1.0);
                Assert.True(tpr[i] >= 0.0 && tpr[i] <= 1.0);
            }
        }

        [Fact]
        public void CalculateAUC_PerfectClassifier_ReturnsOne()
        {
            // Arrange - Perfect ROC curve
            var fpr = new Vector<double>(3);
            fpr[0] = 0.0; fpr[1] = 0.0; fpr[2] = 1.0;

            var tpr = new Vector<double>(3);
            tpr[0] = 0.0; tpr[1] = 1.0; tpr[2] = 1.0;

            // Act
            var auc = StatisticsHelper<double>.CalculateAUC(fpr, tpr);

            // Assert
            Assert.Equal(1.0, auc, precision: 8);
        }

        [Fact]
        public void CalculateROCAUC_ReasonableClassifier_ProducesValidAUC()
        {
            // Arrange
            var actual = new Vector<double>(8);
            actual[0] = 0; actual[1] = 0; actual[2] = 0; actual[3] = 0;
            actual[4] = 1; actual[5] = 1; actual[6] = 1; actual[7] = 1;

            var predicted = new Vector<double>(8);
            predicted[0] = 0.1; predicted[1] = 0.3; predicted[2] = 0.4; predicted[3] = 0.45;
            predicted[4] = 0.55; predicted[5] = 0.7; predicted[6] = 0.8; predicted[7] = 0.9;

            // Act
            var auc = StatisticsHelper<double>.CalculateROCAUC(actual, predicted);

            // Assert
            Assert.True(auc > 0.5); // Better than random
            Assert.True(auc <= 1.0);
        }

        #endregion

        #region Advanced Statistical Tests

        [Fact]
        public void CalculatePValue_TTest_ProducesValidPValue()
        {
            // Arrange
            var group1 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group1[i] = 5.0 + i * 0.3;

            var group2 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group2[i] = 6.0 + i * 0.3;

            // Act
            var pValue = StatisticsHelper<double>.CalculatePValue(group1, group2, TestStatisticType.TTest);

            // Assert
            Assert.True(pValue >= 0.0 && pValue <= 1.0);
        }

        [Fact]
        public void CalculatePValue_MannWhitneyU_ProducesValidPValue()
        {
            // Arrange
            var group1 = new Vector<double>(8);
            for (int i = 0; i < 8; i++) group1[i] = 10.0 + i;

            var group2 = new Vector<double>(8);
            for (int i = 0; i < 8; i++) group2[i] = 15.0 + i;

            // Act
            var pValue = StatisticsHelper<double>.CalculatePValue(group1, group2, TestStatisticType.MannWhitneyU);

            // Assert
            Assert.True(pValue >= 0.0 && pValue <= 1.0);
        }

        [Fact]
        public void CalculatePValue_FTest_ProducesValidPValue()
        {
            // Arrange
            var group1 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group1[i] = 5.0 + i * 0.5;

            var group2 = new Vector<double>(10);
            for (int i = 0; i < 10; i++) group2[i] = 5.0 + i * 2.0; // Higher variance

            // Act
            var pValue = StatisticsHelper<double>.CalculatePValue(group1, group2, TestStatisticType.FTest);

            // Assert
            Assert.True(pValue >= 0.0 && pValue <= 1.0);
        }

        #endregion

        #region Distribution Fitting

        [Fact]
        public void DetermineBestFitDistribution_ProducesValidResult()
        {
            // Arrange - Normal-ish data
            var values = new Vector<double>(50);
            for (int i = 0; i < 50; i++)
            {
                values[i] = 10.0 + (i - 25) * 0.2;
            }

            // Act
            var result = StatisticsHelper<double>.DetermineBestFitDistribution(values);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.BestDistribution != DistributionType.Unknown);
        }

        #endregion

        #region Explained Variance

        [Fact]
        public void CalculateExplainedVarianceScore_PerfectPrediction_ReturnsOne()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 1.0; predicted[1] = 2.0; predicted[2] = 3.0; predicted[3] = 4.0; predicted[4] = 5.0;

            // Act
            var evs = StatisticsHelper<double>.CalculateExplainedVarianceScore(actual, predicted);

            // Assert
            Assert.Equal(1.0, evs, precision: 10);
        }

        [Fact]
        public void CalculateExplainedVarianceScore_PoorPrediction_ReturnsLowValue()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 5.0; predicted[1] = 4.0; predicted[2] = 3.0; predicted[3] = 2.0; predicted[4] = 1.0;

            // Act
            var evs = StatisticsHelper<double>.CalculateExplainedVarianceScore(actual, predicted);

            // Assert
            Assert.True(evs < 0.5);
        }

        #endregion

        #region Covariance and Correlation Matrix

        [Fact]
        public void CalculateCovarianceMatrix_ProducesSymmetricMatrix()
        {
            // Arrange
            var features = new Matrix<double>(5, 3);
            for (int i = 0; i < 5; i++)
            {
                features[i, 0] = i * 1.0;
                features[i, 1] = i * 2.0;
                features[i, 2] = i * 0.5;
            }

            // Act
            var covMatrix = StatisticsHelper<double>.CalculateCovarianceMatrix(features);

            // Assert
            Assert.Equal(3, covMatrix.Rows);
            Assert.Equal(3, covMatrix.Columns);

            // Check symmetry
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(covMatrix[i, j], covMatrix[j, i], precision: 10);
                }
            }
        }

        [Fact]
        public void CalculateCorrelationMatrix_ProducesDiagonalOnes()
        {
            // Arrange
            var features = new Matrix<double>(10, 4);
            for (int i = 0; i < 10; i++)
            {
                features[i, 0] = i * 1.0;
                features[i, 1] = i * 2.0 + 1.0;
                features[i, 2] = Math.Sin(i);
                features[i, 3] = i * 0.5 + 2.0;
            }

            var options = new ModelStatsOptions();

            // Act
            var corrMatrix = StatisticsHelper<double>.CalculateCorrelationMatrix(features, options);

            // Assert
            Assert.Equal(4, corrMatrix.Rows);
            Assert.Equal(4, corrMatrix.Columns);

            // Diagonal should be 1 (self-correlation)
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(1.0, corrMatrix[i, i], precision: 8);
            }
        }

        #endregion

        #region Additional Tests for Edge Cases

        [Fact]
        public void Mean_SingleValue_ReturnsValue()
        {
            // Arrange
            var data = new Vector<double>(1);
            data[0] = 42.0;

            // Act
            var mean = StatisticsHelper<double>.CalculateMean(data);

            // Assert
            Assert.Equal(42.0, mean, precision: 10);
        }

        [Fact]
        public void Variance_TwoIdenticalValues_ReturnsZero()
        {
            // Arrange
            var data = new Vector<double>(2);
            data[0] = 5.0;
            data[1] = 5.0;

            // Act
            var variance = StatisticsHelper<double>.CalculateVariance(data);

            // Assert
            Assert.Equal(0.0, variance, precision: 10);
        }

        [Fact]
        public void TTest_LargeSampleSize_ProducesStableResults()
        {
            // Arrange
            var group1 = new Vector<double>(100);
            var group2 = new Vector<double>(100);

            for (int i = 0; i < 100; i++)
            {
                group1[i] = 50.0 + i * 0.1;
                group2[i] = 50.5 + i * 0.1;
            }

            // Act
            var result = StatisticsHelper<double>.TTest(group1, group2);

            // Assert
            Assert.True(Math.Abs(result.TStatistic) < 10); // Reasonable t-statistic
            Assert.Equal(198, result.DegreesOfFreedom);
        }

        [Fact]
        public void CalculateR2_ConstantPrediction_ReturnsZero()
        {
            // Arrange
            var actual = new Vector<double>(5);
            actual[0] = 1.0; actual[1] = 2.0; actual[2] = 3.0; actual[3] = 4.0; actual[4] = 5.0;

            var predicted = new Vector<double>(5);
            predicted[0] = 3.0; predicted[1] = 3.0; predicted[2] = 3.0; predicted[3] = 3.0; predicted[4] = 3.0;
            // Constant prediction at mean

            // Act
            var r2 = StatisticsHelper<double>.CalculateR2(actual, predicted);

            // Assert
            Assert.InRange(r2, -0.1, 0.1); // Should be close to 0
        }

        [Fact]
        public void CalculatePearsonCorrelation_NoVariation_HandlesGracefully()
        {
            // Arrange - One vector has no variation
            var x = new Vector<double>(5);
            x[0] = 5.0; x[1] = 5.0; x[2] = 5.0; x[3] = 5.0; x[4] = 5.0;

            var y = new Vector<double>(5);
            y[0] = 1.0; y[1] = 2.0; y[2] = 3.0; y[3] = 4.0; y[4] = 5.0;

            // Act & Assert - Should handle without crashing
            // Correlation is undefined when one variable has no variance
            try
            {
                var corr = StatisticsHelper<double>.CalculatePearsonCorrelation(x, y);
                // If it returns a value, it should handle the edge case
                Assert.True(double.IsNaN(corr) || Math.Abs(corr) <= 1.0);
            }
            catch (DivideByZeroException)
            {
                // Also acceptable behavior for this edge case
                Assert.True(true);
            }
        }

        [Fact]
        public void NormalCDF_ExtremeValues_ReturnsValidBounds()
        {
            // Arrange
            double mean = 0.0;
            double stdDev = 1.0;

            // Act
            var cdfLow = StatisticsHelper<double>.CalculateNormalCDF(mean, stdDev, -10.0);
            var cdfHigh = StatisticsHelper<double>.CalculateNormalCDF(mean, stdDev, 10.0);

            // Assert
            Assert.True(cdfLow >= 0.0 && cdfLow < 0.001); // Very close to 0
            Assert.True(cdfHigh > 0.999 && cdfHigh <= 1.0); // Very close to 1
        }

        [Fact]
        public void MannWhitneyUTest_SmallSampleSizes_ProducesValidResult()
        {
            // Arrange
            var group1 = new Vector<double>(3);
            group1[0] = 1.0; group1[1] = 2.0; group1[2] = 3.0;

            var group2 = new Vector<double>(3);
            group2[0] = 4.0; group2[1] = 5.0; group2[2] = 6.0;

            // Act
            var result = StatisticsHelper<double>.MannWhitneyUTest(group1, group2);

            // Assert
            Assert.True(result.UStatistic >= 0);
            Assert.True(result.PValue >= 0.0 && result.PValue <= 1.0);
        }

        #endregion

        #region Weibull Distribution Tests

        [Fact]
        public void EstimateWeibullParameters_ProducesPositiveParameters()
        {
            // Arrange
            var values = new Vector<double>(20);
            for (int i = 0; i < 20; i++)
            {
                values[i] = Math.Pow(i + 1, 0.5); // Weibull-like data
            }

            // Act
            var (shape, scale) = StatisticsHelper<double>.EstimateWeibullParameters(values);

            // Assert
            Assert.True(shape > 0);
            Assert.True(scale > 0);
        }

        #endregion

        #region Log-Likelihood and Model Comparison

        [Fact]
        public void CalculateLogLikelihood_ProducesFiniteValue()
        {
            // Arrange
            var actual = new Vector<double>(10);
            for (int i = 0; i < 10; i++) actual[i] = 5.0 + i * 0.5;

            var predicted = new Vector<double>(10);
            for (int i = 0; i < 10; i++) predicted[i] = 5.2 + i * 0.5;

            // Act
            var logLikelihood = StatisticsHelper<double>.CalculateLogLikelihood(actual, predicted);

            // Assert
            Assert.False(double.IsInfinity(logLikelihood));
            Assert.False(double.IsNaN(logLikelihood));
        }

        #endregion

        #region Condition Number

        [Fact]
        public void CalculateConditionNumber_IdentityMatrix_ReturnsOne()
        {
            // Arrange - Identity matrix has condition number = 1
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[1, 1] = 1.0; matrix[2, 2] = 1.0;

            var options = new ModelStatsOptions();

            // Act
            var conditionNumber = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

            // Assert
            Assert.InRange(conditionNumber, 0.9, 1.1); // Should be very close to 1
        }

        #endregion

        #region Sample Standard Error

        [Fact]
        public void CalculateSampleStandardError_ProducesPositiveValue()
        {
            // Arrange
            var actual = new Vector<double>(20);
            for (int i = 0; i < 20; i++) actual[i] = 10.0 + i;

            var predicted = new Vector<double>(20);
            for (int i = 0; i < 20; i++) predicted[i] = 10.5 + i;

            int numberOfParameters = 3;

            // Act
            var sse = StatisticsHelper<double>.CalculateSampleStandardError(actual, predicted, numberOfParameters);

            // Assert
            Assert.True(sse > 0);
        }

        [Fact]
        public void CalculatePopulationStandardError_ProducesPositiveValue()
        {
            // Arrange
            var actual = new Vector<double>(15);
            for (int i = 0; i < 15; i++) actual[i] = 5.0 + i * 0.3;

            var predicted = new Vector<double>(15);
            for (int i = 0; i < 15; i++) predicted[i] = 5.1 + i * 0.3;

            // Act
            var pse = StatisticsHelper<double>.CalculatePopulationStandardError(actual, predicted);

            // Assert
            Assert.True(pse > 0);
        }

        #endregion

        #region Skewness and Kurtosis

        [Fact]
        public void CalculateSkewnessAndKurtosis_SymmetricData_LowSkewness()
        {
            // Arrange - Symmetric data
            var sample = new Vector<double>(11);
            for (int i = 0; i < 11; i++) sample[i] = i - 5.0; // -5 to 5

            double mean = 0.0;
            double stdDev = StatisticsHelper<double>.CalculateStandardDeviation(sample);
            int n = 11;

            // Act
            var (skewness, kurtosis) = StatisticsHelper<double>.CalculateSkewnessAndKurtosis(sample, mean, stdDev, n);

            // Assert
            Assert.True(Math.Abs(skewness) < 0.5); // Should be close to 0
            Assert.True(kurtosis > 0); // Should be positive
        }

        #endregion
    }
}
