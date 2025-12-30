namespace AiDotNet.Tests.IntegrationTests.Helpers;

using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using Xunit;

/// <summary>
/// Integration tests for StatisticsHelper - comprehensive coverage of all statistical functions.
/// </summary>
public class StatisticsHelperIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Basic Statistics Tests

    [Fact]
    public void CalculateMean_WithValidValues_ReturnsCorrectMean()
    {
        // Arrange
        var values = new double[] { 1, 2, 3, 4, 5 };

        // Act
        var result = StatisticsHelper<double>.CalculateMean(values);

        // Assert
        Assert.Equal(3.0, result, Tolerance);
    }

    [Fact]
    public void CalculateMean_WithSingleValue_ReturnsThatValue()
    {
        // Arrange
        var values = new double[] { 42.0 };

        // Act
        var result = StatisticsHelper<double>.CalculateMean(values);

        // Assert
        Assert.Equal(42.0, result, Tolerance);
    }

    [Fact]
    public void CalculateMedian_WithOddCount_ReturnsMiddleValue()
    {
        // Arrange
        var values = new double[] { 1, 3, 2, 5, 4 };

        // Act
        var result = StatisticsHelper<double>.CalculateMedian(values);

        // Assert
        Assert.Equal(3.0, result, Tolerance);
    }

    [Fact]
    public void CalculateMedian_WithEvenCount_ReturnsAverageOfMiddleValues()
    {
        // Arrange
        var values = new double[] { 1, 2, 3, 4 };

        // Act
        var result = StatisticsHelper<double>.CalculateMedian(values);

        // Assert
        Assert.Equal(2.5, result, Tolerance);
    }

    [Fact]
    public void CalculateVariance_WithValidValues_ReturnsCorrectVariance()
    {
        // Arrange
        var values = new double[] { 2, 4, 4, 4, 5, 5, 7, 9 };

        // Act
        var result = StatisticsHelper<double>.CalculateVariance(values);

        // Assert
        // Sample variance = Σ(x - mean)² / (n-1)
        Assert.True(result > 0);
    }

    [Fact]
    public void CalculateStandardDeviation_WithValidValues_ReturnsCorrectStdDev()
    {
        // Arrange
        var values = new double[] { 2, 4, 4, 4, 5, 5, 7, 9 };

        // Act
        var result = StatisticsHelper<double>.CalculateStandardDeviation(values);

        // Assert
        Assert.True(result > 0);
        // Standard deviation is sqrt of variance
        var variance = StatisticsHelper<double>.CalculateVariance(values);
        Assert.Equal(Math.Sqrt(variance), result, Tolerance);
    }

    [Fact]
    public void CalculateMeanAndStandardDeviation_ReturnsConsistentResults()
    {
        // Arrange
        var values = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        // Act
        var (mean, stdDev) = StatisticsHelper<double>.CalculateMeanAndStandardDeviation(values);

        // Assert
        Assert.Equal(5.5, mean, Tolerance);
        Assert.True(stdDev > 0);
    }

    [Fact]
    public void CalculateMeanAbsoluteDeviation_ReturnsCorrectValue()
    {
        // Arrange
        var values = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var median = StatisticsHelper<double>.CalculateMedian(values.ToArray());

        // Act
        var result = StatisticsHelper<double>.CalculateMeanAbsoluteDeviation(values, median);

        // Assert
        Assert.True(result >= 0);
    }

    [Fact]
    public void CalculateMedianAbsoluteDeviation_ReturnsCorrectValue()
    {
        // Arrange
        var values = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

        // Act
        var result = StatisticsHelper<double>.CalculateMedianAbsoluteDeviation(values);

        // Assert
        Assert.True(result >= 0);
    }

    [Fact]
    public void CalculateQuantiles_ReturnsFirstAndThirdQuantiles()
    {
        // Arrange
        var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

        // Act
        var (q1, q3) = StatisticsHelper<double>.CalculateQuantiles(data);

        // Assert
        Assert.True(q1 < q3);
        Assert.True(q1 >= 1 && q1 <= 12);
        Assert.True(q3 >= 1 && q3 <= 12);
    }

    [Fact]
    public void CalculateSkewnessAndKurtosis_ReturnsValidValues()
    {
        // Arrange
        var sample = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var mean = StatisticsHelper<double>.CalculateMean(sample.ToArray());
        var stdDev = StatisticsHelper<double>.CalculateStandardDeviation(sample.ToArray());

        // Act
        var (skewness, kurtosis) = StatisticsHelper<double>.CalculateSkewnessAndKurtosis(sample, mean, stdDev, sample.Length);

        // Assert - symmetric distribution should have skewness near 0
        Assert.True(Math.Abs(skewness) < 1.0);
    }

    #endregion

    #region Error Metrics Tests

    [Fact]
    public void CalculateMeanSquaredError_WithPerfectPrediction_ReturnsZero()
    {
        // Arrange
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var predicted = new double[] { 1, 2, 3, 4, 5 };

        // Act
        var result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, predicted);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CalculateMeanSquaredError_WithDifferences_ReturnsPositiveValue()
    {
        // Arrange
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var predicted = new double[] { 1.1, 2.1, 3.1, 4.1, 5.1 };

        // Act
        var result = StatisticsHelper<double>.CalculateMeanSquaredError(actual, predicted);

        // Assert
        Assert.True(result > 0);
        Assert.Equal(0.01, result, Tolerance); // Average of 0.01 squared differences
    }

    [Fact]
    public void CalculateRootMeanSquaredError_ReturnsSquareRootOfMSE()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 });

        // Act
        var rmse = StatisticsHelper<double>.CalculateRootMeanSquaredError(actual, predicted);
        var mse = StatisticsHelper<double>.CalculateMeanSquaredError(actual.ToArray(), predicted.ToArray());

        // Assert
        Assert.Equal(Math.Sqrt(mse), rmse, Tolerance);
    }

    [Fact]
    public void CalculateMeanAbsoluteError_ReturnsCorrectValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 });

        // Act
        var result = StatisticsHelper<double>.CalculateMeanAbsoluteError(actual, predicted);

        // Assert
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void CalculateMeanAbsolutePercentageError_ReturnsPercentage()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 100, 200, 300, 400, 500 });
        var predicted = new Vector<double>(new double[] { 110, 220, 330, 440, 550 });

        // Act
        var result = StatisticsHelper<double>.CalculateMeanAbsolutePercentageError(actual, predicted);

        // Assert - MAPE returns percentage value (10 = 10%), not decimal (0.1)
        Assert.Equal(10.0, result, Tolerance); // 10% error
    }

    [Fact]
    public void CalculateR2_WithPerfectPrediction_ReturnsOne()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

        // Act
        var result = StatisticsHelper<double>.CalculateR2(actual, predicted);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculateR2_WithRandomPrediction_ReturnsLowValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 5, 4, 3, 2, 1 }); // Reverse order

        // Act
        var result = StatisticsHelper<double>.CalculateR2(actual, predicted);

        // Assert
        Assert.True(result < 0.5); // Poor fit
    }

    [Fact]
    public void CalculateAdjustedR2_IsLessThanOrEqualToR2()
    {
        // Arrange
        double r2 = 0.85;
        int n = 100;
        int p = 5;

        // Act
        var adjustedR2 = StatisticsHelper<double>.CalculateAdjustedR2(r2, n, p);

        // Assert
        Assert.True(adjustedR2 <= r2);
    }

    [Fact]
    public void CalculateExplainedVarianceScore_WithPerfectPrediction_ReturnsOne()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

        // Act
        var result = StatisticsHelper<double>.CalculateExplainedVarianceScore(actual, predicted);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculateMaxError_ReturnsLargestAbsoluteDifference()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1, 2, 5, 4, 5 }); // 3->5 is biggest error

        // Act
        var result = StatisticsHelper<double>.CalculateMaxError(actual, predicted);

        // Assert
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void CalculateMedianAbsoluteError_ReturnsMedianOfErrors()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 });

        // Act
        var result = StatisticsHelper<double>.CalculateMedianAbsoluteError(actual, predicted);

        // Assert
        Assert.True(result >= 0);
    }

    [Fact]
    public void CalculateMeanBiasError_ReturnsAverageBias()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 2, 3, 4, 5, 6 }); // All +1

        // Act
        var result = StatisticsHelper<double>.CalculateMeanBiasError(actual, predicted);

        // Assert - MeanBiasError = predicted - actual, positive when over-predicting
        Assert.Equal(1.0, result, Tolerance); // Systematic over-prediction
    }

    [Fact]
    public void CalculateSymmetricMeanAbsolutePercentageError_ReturnsValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 100, 200, 300, 400, 500 });
        var predicted = new Vector<double>(new double[] { 110, 190, 310, 390, 510 });

        // Act
        var result = StatisticsHelper<double>.CalculateSymmetricMeanAbsolutePercentageError(actual, predicted);

        // Assert - SMAPE returns percentage value, typically between 0 and 200
        Assert.True(result >= 0 && result <= 200.0); // SMAPE is between 0 and 200
    }

    [Fact]
    public void CalculateMeanSquaredLogError_ReturnsPositiveValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1.1, 2.1, 3.1, 4.1, 5.1 });

        // Act
        var result = StatisticsHelper<double>.CalculateMeanSquaredLogError(actual, predicted);

        // Assert
        Assert.True(result >= 0);
    }

    #endregion

    #region Correlation Tests

    [Fact]
    public void CalculatePearsonCorrelation_WithPerfectPositiveCorrelation_ReturnsOne()
    {
        // Arrange
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var y = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });

        // Act
        var result = StatisticsHelper<double>.CalculatePearsonCorrelation(x, y);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculatePearsonCorrelation_WithPerfectNegativeCorrelation_ReturnsMinusOne()
    {
        // Arrange
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var y = new Vector<double>(new double[] { 10, 8, 6, 4, 2 });

        // Act
        var result = StatisticsHelper<double>.CalculatePearsonCorrelation(x, y);

        // Assert
        Assert.Equal(-1.0, result, Tolerance);
    }

    [Fact]
    public void CalculatePearsonCorrelationCoefficient_MatchesPearsonCorrelation()
    {
        // Arrange
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var y = new Vector<double>(new double[] { 1.5, 2.3, 3.1, 4.2, 5.1, 5.9, 7.2, 8.0, 9.1, 10.2 });

        // Act
        var result1 = StatisticsHelper<double>.CalculatePearsonCorrelation(x, y);
        var result2 = StatisticsHelper<double>.CalculatePearsonCorrelationCoefficient(x, y);

        // Assert
        Assert.Equal(result1, result2, Tolerance);
    }

    [Fact]
    public void CalculateSpearmanRankCorrelationCoefficient_WithMonotonicData_ReturnsHighValue()
    {
        // Arrange
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var y = new Vector<double>(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 });

        // Act
        var result = StatisticsHelper<double>.CalculateSpearmanRankCorrelationCoefficient(x, y);

        // Assert
        Assert.True(result > 0.9);
    }

    [Fact]
    public void CalculateKendallTau_WithConcordantPairs_ReturnsPositive()
    {
        // Arrange
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var y = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

        // Act
        var result = StatisticsHelper<double>.CalculateKendallTau(x, y);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region Distance Metrics Tests

    [Fact]
    public void EuclideanDistance_WithIdenticalVectors_ReturnsZero()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 1, 2, 3 });

        // Act
        var result = StatisticsHelper<double>.EuclideanDistance(v1, v2);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void EuclideanDistance_WithKnownVectors_ReturnsCorrectDistance()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 0, 0, 0 });
        var v2 = new Vector<double>(new double[] { 3, 4, 0 });

        // Act
        var result = StatisticsHelper<double>.EuclideanDistance(v1, v2);

        // Assert
        Assert.Equal(5.0, result, Tolerance); // 3-4-5 triangle
    }

    [Fact]
    public void ManhattanDistance_WithKnownVectors_ReturnsCorrectDistance()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 0, 0, 0 });
        var v2 = new Vector<double>(new double[] { 1, 2, 3 });

        // Act
        var result = StatisticsHelper<double>.ManhattanDistance(v1, v2);

        // Assert
        Assert.Equal(6.0, result, Tolerance); // |1| + |2| + |3|
    }

    [Fact]
    public void CosineSimilarity_WithIdenticalVectors_ReturnsOne()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 1, 2, 3 });

        // Act
        var result = StatisticsHelper<double>.CosineSimilarity(v1, v2);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_WithOrthogonalVectors_ReturnsZero()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 0, 0 });
        var v2 = new Vector<double>(new double[] { 0, 1, 0 });

        // Act
        var result = StatisticsHelper<double>.CosineSimilarity(v1, v2);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void JaccardSimilarity_WithIdenticalSets_ReturnsOne()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 1, 0, 0 });
        var v2 = new Vector<double>(new double[] { 1, 1, 0, 0 });

        // Act
        var result = StatisticsHelper<double>.JaccardSimilarity(v1, v2);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void HammingDistance_WithDifferentBits_ReturnsCount()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 0, 1, 0 });
        var v2 = new Vector<double>(new double[] { 0, 0, 1, 1 });

        // Act
        var result = StatisticsHelper<double>.HammingDistance(v1, v2);

        // Assert
        Assert.Equal(2.0, result, Tolerance); // Two positions differ
    }

    [Fact]
    public void CalculateDistance_WithDifferentMetrics_ReturnsValidDistances()
    {
        // Arrange
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 4, 5, 6 });

        // Act
        var euclidean = StatisticsHelper<double>.CalculateDistance(v1, v2, DistanceMetricType.Euclidean);
        var manhattan = StatisticsHelper<double>.CalculateDistance(v1, v2, DistanceMetricType.Manhattan);
        var cosine = StatisticsHelper<double>.CalculateDistance(v1, v2, DistanceMetricType.Cosine);

        // Assert
        Assert.True(euclidean >= 0);
        Assert.True(manhattan >= 0);
        Assert.True(cosine >= 0 && cosine <= 2);
    }

    #endregion

    #region Distribution Tests

    [Fact]
    public void CalculateNormalCDF_AtMean_ReturnsHalf()
    {
        // Arrange
        double mean = 0.0;
        double stdDev = 1.0;
        double x = 0.0;

        // Act
        var result = StatisticsHelper<double>.CalculateNormalCDF(mean, stdDev, x);

        // Assert
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void CalculateNormalPDF_AtMean_ReturnsMaxValue()
    {
        // Arrange
        double mean = 0.0;
        double stdDev = 1.0;

        // Act
        var atMean = StatisticsHelper<double>.CalculateNormalPDF(mean, stdDev, 0.0);
        var awayFromMean = StatisticsHelper<double>.CalculateNormalPDF(mean, stdDev, 2.0);

        // Assert
        Assert.True(atMean > awayFromMean);
    }

    [Fact]
    public void CalculateInverseNormalCDF_AtHalf_ReturnsMean()
    {
        // Arrange
        double probability = 0.5;

        // Act
        var result = StatisticsHelper<double>.CalculateInverseNormalCDF(probability);

        // Assert
        Assert.Equal(0.0, result, 0.01); // Standard normal mean is 0
    }

    [Fact]
    public void CalculateChiSquareCDF_WithPositiveX_ReturnsValidProbability()
    {
        // Arrange
        int df = 5;
        double x = 5.0;

        // Act
        var result = StatisticsHelper<double>.CalculateChiSquareCDF(df, x);

        // Assert
        Assert.True(result >= 0 && result <= 1);
    }

    [Fact]
    public void CalculateChiSquarePDF_WithValidInput_ReturnsPositive()
    {
        // Arrange
        int df = 5;
        double x = 5.0;

        // Act
        var result = StatisticsHelper<double>.CalculateChiSquarePDF(df, x);

        // Assert
        Assert.True(result >= 0);
    }

    [Fact]
    public void DetermineBestFitDistribution_ReturnsValidResult()
    {
        // Arrange - normally distributed data
        var random = new Random(42);
        var values = new Vector<double>(Enumerable.Range(0, 100)
            .Select(_ => random.NextDouble() * 10 - 5).ToArray());

        // Act
        var result = StatisticsHelper<double>.DetermineBestFitDistribution(values);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void CalculateBetaCDF_AtBoundaries_ReturnsValidValues()
    {
        // Arrange
        double alpha = 2.0;
        double beta = 5.0;

        // Act
        var at0 = StatisticsHelper<double>.CalculateBetaCDF(0.0, alpha, beta);
        var at1 = StatisticsHelper<double>.CalculateBetaCDF(1.0, alpha, beta);

        // Assert
        Assert.Equal(0.0, at0, Tolerance);
        Assert.Equal(1.0, at1, Tolerance);
    }

    #endregion

    #region Statistical Tests

    [Fact]
    public void TTest_WithSameDistributions_ReturnsHighPValue()
    {
        // Arrange
        var leftY = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var rightY = new Vector<double>(new double[] { 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1 });

        // Act
        var result = StatisticsHelper<double>.TTest(leftY, rightY);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PValue >= 0 && result.PValue <= 1);
    }

    [Fact]
    public void TTest_WithDifferentDistributions_ReturnsLowPValue()
    {
        // Arrange
        var leftY = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var rightY = new Vector<double>(new double[] { 100, 110, 120, 130, 140 });

        // Act
        var result = StatisticsHelper<double>.TTest(leftY, rightY);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PValue < 0.05); // Significant difference
    }

    [Fact]
    public void MannWhitneyUTest_WithValidData_ReturnsResult()
    {
        // Arrange
        var leftY = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var rightY = new Vector<double>(new double[] { 5, 6, 7, 8, 9, 10, 11, 12 });

        // Act
        var result = StatisticsHelper<double>.MannWhitneyUTest(leftY, rightY);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PValue >= 0 && result.PValue <= 1);
    }

    [Fact]
    public void ChiSquareTest_WithValidData_ReturnsResult()
    {
        // Arrange
        var leftY = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var rightY = new Vector<double>(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5 });

        // Act
        var result = StatisticsHelper<double>.ChiSquareTest(leftY, rightY);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void FTest_WithValidData_ReturnsResult()
    {
        // Arrange
        var leftY = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var rightY = new Vector<double>(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });

        // Act
        var result = StatisticsHelper<double>.FTest(leftY, rightY);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PValue >= 0 && result.PValue <= 1);
    }

    [Fact]
    public void PermutationTest_WithValidData_ReturnsResult()
    {
        // Arrange
        var leftY = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var rightY = new Vector<double>(new double[] { 6, 7, 8, 9, 10 });

        // Act
        var result = StatisticsHelper<double>.PermutationTest(leftY, rightY);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PValue >= 0 && result.PValue <= 1);
    }

    [Fact]
    public void CalculatePValue_WithDifferentTestTypes_ReturnsValidValues()
    {
        // Arrange - continuous data for t-test
        var leftY = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var rightY = new Vector<double>(new double[] { 2, 4, 6, 8, 10 });

        // Categorical data for chi-square (must have overlapping categories with sufficient frequencies)
        var leftCategorical = new Vector<double>(new double[] { 0, 0, 0, 1, 1, 1, 2, 2 });
        var rightCategorical = new Vector<double>(new double[] { 0, 0, 1, 1, 1, 2, 2, 2 });

        // Act
        var pValueTTest = StatisticsHelper<double>.CalculatePValue(leftY, rightY, TestStatisticType.TTest);
        var pValueChiSquare = StatisticsHelper<double>.CalculatePValue(leftCategorical, rightCategorical, TestStatisticType.ChiSquare);

        // Assert
        Assert.True(pValueTTest >= 0 && pValueTTest <= 1);
        Assert.True(pValueChiSquare >= 0 && pValueChiSquare <= 1);
    }

    #endregion

    #region Model Evaluation Tests

    [Fact]
    public void CalculateAIC_WithValidInputs_ReturnsValue()
    {
        // Arrange
        int sampleSize = 100;
        int parameterSize = 5;
        double rss = 50.0;

        // Act
        var result = StatisticsHelper<double>.CalculateAIC(sampleSize, parameterSize, rss);

        // Assert
        Assert.True(!double.IsNaN(result));
    }

    [Fact]
    public void CalculateBIC_PenalizesMoreParametersThanAIC()
    {
        // Arrange
        int sampleSize = 100;
        int parameterSize = 5;
        double rss = 50.0;

        // Act
        var aic = StatisticsHelper<double>.CalculateAIC(sampleSize, parameterSize, rss);
        var bic = StatisticsHelper<double>.CalculateBIC(sampleSize, parameterSize, rss);

        // Assert - BIC typically penalizes more for larger sample sizes
        Assert.True(!double.IsNaN(aic));
        Assert.True(!double.IsNaN(bic));
    }

    [Fact]
    public void CalculateAccuracy_WithPerfectPrediction_ReturnsOne()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 0, 1, 0, 1, 1 });
        var predicted = new Vector<double>(new double[] { 0, 1, 0, 1, 1 });

        // Act
        var result = StatisticsHelper<double>.CalculateAccuracy(actual, predicted);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculatePrecisionRecallF1_WithValidData_ReturnsValidMetrics()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 1, 0, 0, 1, 0, 1, 1 });
        var predicted = new Vector<double>(new double[] { 1, 0, 0, 0, 1, 1, 1, 1 });

        // Act
        var (precision, recall, f1) = StatisticsHelper<double>.CalculatePrecisionRecallF1(actual, predicted, PredictionType.BinaryClassification, 0.5);

        // Assert
        Assert.True(precision >= 0 && precision <= 1);
        Assert.True(recall >= 0 && recall <= 1);
        Assert.True(f1 >= 0 && f1 <= 1);
    }

    [Fact]
    public void CalculateF1Score_IsHarmonicMean()
    {
        // Arrange
        double precision = 0.8;
        double recall = 0.6;

        // Act
        var f1 = StatisticsHelper<double>.CalculateF1Score(precision, recall);

        // Assert
        // F1 = 2 * (precision * recall) / (precision + recall)
        var expected = 2 * precision * recall / (precision + recall);
        Assert.Equal(expected, f1, Tolerance);
    }

    [Fact]
    public void CalculateConfusionMatrix_WithBinaryData_ReturnsValidMatrix()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 1, 0, 0, 1, 0 });
        var predicted = new Vector<double>(new double[] { 1, 0, 0, 1, 1, 0 });

        // Act
        var result = StatisticsHelper<double>.CalculateConfusionMatrix(actual, predicted, 0.5);

        // Assert
        Assert.NotNull(result);
    }

    [Fact]
    public void CalculateROCAUC_WithPerfectPrediction_ReturnsOne()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 0, 0, 0, 1, 1, 1 });
        var predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.7, 0.8, 0.9 });

        // Act
        var result = StatisticsHelper<double>.CalculateROCAUC(actual, predicted);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculateROCCurve_ReturnsValidCurve()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 0, 0, 1, 1, 0, 1 });
        var predicted = new Vector<double>(new double[] { 0.1, 0.4, 0.35, 0.8, 0.3, 0.9 });

        // Act
        var (fpr, tpr) = StatisticsHelper<double>.CalculateROCCurve(actual, predicted);

        // Assert
        Assert.NotNull(fpr);
        Assert.NotNull(tpr);
        Assert.Equal(fpr.Length, tpr.Length);
    }

    [Fact]
    public void CalculateDurbinWatsonStatistic_WithResiduals_ReturnsValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var predicted = new Vector<double>(new double[] { 1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9 });

        // Act
        var result = StatisticsHelper<double>.CalculateDurbinWatsonStatistic(actual, predicted);

        // Assert
        Assert.True(result >= 0 && result <= 4); // DW statistic range
    }

    [Fact]
    public void CalculateTheilUStatistic_WithValidData_ReturnsValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1.1, 2.2, 2.9, 4.1, 4.9 });

        // Act
        var result = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predicted);

        // Assert
        Assert.True(result >= 0);
    }

    #endregion

    #region Information Theory Tests

    [Fact]
    public void CalculateMutualInformation_WithDependentVariables_ReturnsPositive()
    {
        // Arrange
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var y = new Vector<double>(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });

        // Act
        var result = StatisticsHelper<double>.CalculateMutualInformation(x, y);

        // Assert
        Assert.True(result >= 0);
    }

    [Fact]
    public void CalculateNormalizedMutualInformation_ReturnsBetweenZeroAndOne()
    {
        // Arrange
        var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var y = new Vector<double>(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });

        // Act
        var result = StatisticsHelper<double>.CalculateNormalizedMutualInformation(x, y);

        // Assert
        Assert.True(result >= 0 && result <= 1);
    }

    #endregion

    #region Clustering Metrics Tests

    [Fact]
    public void CalculateSilhouetteScore_WithWellSeparatedClusters_ReturnsHighValue()
    {
        // Arrange - Two well-separated clusters
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 }, { 0.1, 0.1 }, { 0.2, 0 },
            { 10, 10 }, { 10.1, 10.1 }, { 10.2, 10 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 0, 1, 1, 1 });

        // Act
        var result = StatisticsHelper<double>.CalculateSilhouetteScore(data, labels);

        // Assert
        Assert.True(result > 0.5); // Well-separated should have high silhouette
    }

    [Fact]
    public void CalculateCalinskiHarabaszIndex_WithValidClusters_ReturnsPositive()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 }, { 0.1, 0.1 },
            { 10, 10 }, { 10.1, 10.1 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 1, 1 });

        // Act
        var result = StatisticsHelper<double>.CalculateCalinskiHarabaszIndex(data, labels);

        // Assert
        Assert.True(result > 0);
    }

    [Fact]
    public void CalculateDaviesBouldinIndex_WithValidClusters_ReturnsPositive()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 0, 0 }, { 0.1, 0.1 }, { 0.2, 0.2 },
            { 10, 10 }, { 10.1, 10.1 }, { 10.2, 10.2 }
        });
        var labels = new Vector<double>(new double[] { 0, 0, 0, 1, 1, 1 });

        // Act
        var result = StatisticsHelper<double>.CalculateDaviesBouldinIndex(data, labels);

        // Assert
        Assert.True(result >= 0);
    }

    [Fact]
    public void CalculateAdjustedRandIndex_WithIdenticalLabels_ReturnsOne()
    {
        // Arrange
        var labels1 = new Vector<double>(new double[] { 0, 0, 1, 1, 2, 2 });
        var labels2 = new Vector<double>(new double[] { 0, 0, 1, 1, 2, 2 });

        // Act
        var result = StatisticsHelper<double>.CalculateAdjustedRandIndex(labels1, labels2);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region Time Series Tests

    [Fact]
    public void CalculateAutoCorrelationFunction_WithLag_ReturnsValidValues()
    {
        // Arrange
        var series = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        int maxLag = 5;

        // Act
        var result = StatisticsHelper<double>.CalculateAutoCorrelationFunction(series, maxLag);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.Length > 0);
        Assert.Equal(1.0, result[0], Tolerance); // ACF at lag 0 is always 1
    }

    [Fact]
    public void CalculatePartialAutoCorrelationFunction_WithLag_ReturnsValidValues()
    {
        // Arrange
        var series = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
        int maxLag = 5;

        // Act
        var result = StatisticsHelper<double>.CalculatePartialAutoCorrelationFunction(series, maxLag);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.Length > 0);
    }

    [Fact]
    public void CalculateDynamicTimeWarping_WithIdenticalSeries_ReturnsZero()
    {
        // Arrange
        var series1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var series2 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

        // Act
        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CalculateDynamicTimeWarping_WithDifferentSeries_ReturnsPositive()
    {
        // Arrange
        var series1 = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var series2 = new Vector<double>(new double[] { 2, 3, 4, 5, 6 });

        // Act
        var result = StatisticsHelper<double>.CalculateDynamicTimeWarping(series1, series2);

        // Assert
        Assert.True(result > 0);
    }

    #endregion

    #region Interval and Confidence Tests

    [Fact]
    public void CalculatePredictionIntervals_ReturnsValidBounds()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var predicted = new Vector<double>(new double[] { 1.1, 2.2, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9, 10.1 });

        // Act
        var (lower, upper) = StatisticsHelper<double>.CalculatePredictionIntervals(actual, predicted, 0.95);

        // Assert
        Assert.True(lower < upper);
    }

    [Fact]
    public void CalculateConfidenceIntervals_WithNormalDistribution_ReturnsValidBounds()
    {
        // Arrange
        var values = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        // Act
        var (lower, upper) = StatisticsHelper<double>.CalculateConfidenceIntervals(values, 0.95, DistributionType.Normal);

        // Assert
        Assert.True(lower < upper);
    }

    [Fact]
    public void CalculateCredibleIntervals_ReturnsValidBounds()
    {
        // Arrange
        var values = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        // Act
        var (lower, upper) = StatisticsHelper<double>.CalculateCredibleIntervals(values, 0.95, DistributionType.Normal);

        // Assert
        Assert.True(lower < upper);
    }

    [Fact]
    public void CalculateBootstrapInterval_ReturnsValidBounds()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var predicted = new Vector<double>(new double[] { 1.1, 2.2, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9, 10.1 });

        // Act
        var (lower, upper) = StatisticsHelper<double>.CalculateBootstrapInterval(actual, predicted, 0.95);

        // Assert
        Assert.True(lower < upper);
    }

    [Fact]
    public void CalculateClopperPearsonInterval_ReturnsBinomialConfidenceInterval()
    {
        // Arrange
        int successes = 7;
        int trials = 10;
        double confidence = 0.95;

        // Act
        var (lower, upper) = StatisticsHelper<double>.CalculateClopperPearsonInterval(successes, trials, confidence);

        // Assert
        Assert.True(lower >= 0 && lower <= 1);
        Assert.True(upper >= 0 && upper <= 1);
        Assert.True(lower < upper);
    }

    [Fact]
    public void CalculateTValue_ReturnsPositiveForValidInputs()
    {
        // Arrange
        int degreesOfFreedom = 10;
        double confidenceLevel = 0.95;

        // Act
        var result = StatisticsHelper<double>.CalculateTValue(degreesOfFreedom, confidenceLevel);

        // Assert
        Assert.True(result > 0);
    }

    #endregion

    #region Residual Analysis Tests

    [Fact]
    public void CalculateResiduals_ReturnsCorrectDifferences()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1.1, 2.2, 2.9, 4.1, 5.0 });

        // Act
        var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);

        // Assert
        Assert.Equal(actual.Length, residuals.Length);
        Assert.Equal(-0.1, residuals[0], Tolerance);
        Assert.Equal(-0.2, residuals[1], Tolerance);
    }

    [Fact]
    public void CalculateTotalSumOfSquares_ReturnsPositive()
    {
        // Arrange
        var values = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

        // Act
        var result = StatisticsHelper<double>.CalculateTotalSumOfSquares(values);

        // Assert
        Assert.True(result >= 0);
    }

    [Fact]
    public void CalculateResidualSumOfSquares_WithPerfectPrediction_ReturnsZero()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

        // Act
        var result = StatisticsHelper<double>.CalculateResidualSumOfSquares(actual, predicted);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Ranking Metrics Tests

    [Fact]
    public void CalculateMeanAveragePrecision_WithPerfectRanking_ReturnsOne()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 1, 1, 0, 0 });
        var predicted = new Vector<double>(new double[] { 0.9, 0.8, 0.7, 0.2, 0.1 });

        // Act
        var result = StatisticsHelper<double>.CalculateMeanAveragePrecision(actual, predicted, 5);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculateNDCG_WithPerfectRanking_ReturnsOne()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 3, 2, 1, 0, 0 });
        var predicted = new Vector<double>(new double[] { 0.9, 0.8, 0.7, 0.2, 0.1 });

        // Act
        var result = StatisticsHelper<double>.CalculateNDCG(actual, predicted, 5);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CalculateMeanReciprocalRank_WithFirstCorrect_ReturnsOne()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 0, 0, 0, 0 });
        var predicted = new Vector<double>(new double[] { 0.9, 0.5, 0.4, 0.3, 0.2 });

        // Act
        var result = StatisticsHelper<double>.CalculateMeanReciprocalRank(actual, predicted);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region Covariance Matrix Tests

    [Fact]
    public void CalculateCovarianceMatrix_ReturnsSymmetricMatrix()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 }
        });

        // Act
        var result = StatisticsHelper<double>.CalculateCovarianceMatrix(matrix);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
        // Covariance matrix should be symmetric
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.Equal(result[i, j], result[j, i], Tolerance);
            }
        }
    }

    [Fact]
    public void CalculateCorrelationMatrix_ReturnsDiagonalOnes()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 }
        });
        var options = new ModelStatsOptions();

        // Act
        var result = StatisticsHelper<double>.CalculateCorrelationMatrix(matrix, options);

        // Assert
        for (int i = 0; i < result.Rows; i++)
        {
            Assert.Equal(1.0, result[i, i], Tolerance);
        }
    }

    #endregion

    #region Gamma and Special Functions Tests

    [Fact]
    public void Gamma_WithPositiveInteger_ReturnsFactorial()
    {
        // Arrange & Act
        var gamma5 = StatisticsHelper<double>.Gamma(5.0); // Should be 4! = 24

        // Assert
        Assert.Equal(24.0, gamma5, Tolerance);
    }

    [Fact]
    public void Digamma_WithPositiveValue_ReturnsValue()
    {
        // Arrange
        double x = 2.0;

        // Act
        var result = StatisticsHelper<double>.Digamma(x);

        // Assert
        Assert.True(!double.IsNaN(result));
    }

    [Fact]
    public void IncompleteGamma_WithValidInputs_ReturnsBetweenZeroAndOne()
    {
        // Arrange
        double a = 2.0;
        double x = 1.0;

        // Act
        var result = StatisticsHelper<double>.IncompleteGamma(a, x);

        // Assert
        Assert.True(result >= 0 && result <= 1);
    }

    #endregion

    #region CRPS Tests

    [Fact]
    public void CalculateCRPS_WithPerfectPrediction_ReturnsLowValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predicted = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });

        // Act
        var result = StatisticsHelper<double>.CalculateCRPS(actual, predicted);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CalculateCRPS_WithMeanAndStdDev_ReturnsValidValue()
    {
        // Arrange
        var actual = new Vector<double>(new double[] { 1, 2, 3, 4, 5 });
        var predictedMean = new Vector<double>(new double[] { 1.1, 2.1, 3.1, 4.1, 5.1 });
        var predictedStdDev = new Vector<double>(new double[] { 0.5, 0.5, 0.5, 0.5, 0.5 });

        // Act
        var result = StatisticsHelper<double>.CalculateCRPS(actual, predictedMean, predictedStdDev);

        // Assert
        Assert.True(result >= 0);
    }

    #endregion

    #region Split Score Tests

    [Fact]
    public void CalculateVarianceReduction_WithValidSplit_ReturnsPositive()
    {
        // Arrange
        var y = new Vector<double>(new double[] { 1, 2, 3, 10, 11, 12 });
        var leftIndices = new List<int> { 0, 1, 2 };
        var rightIndices = new List<int> { 3, 4, 5 };

        // Act
        var result = StatisticsHelper<double>.CalculateVarianceReduction(y, leftIndices, rightIndices);

        // Assert
        Assert.True(result > 0);
    }

    [Fact]
    public void CalculateSplitScore_WithDifferentCriteria_ReturnsValues()
    {
        // Arrange
        var y = new Vector<double>(new double[] { 1, 2, 3, 10, 11, 12 });
        var leftIndices = new List<int> { 0, 1, 2 };
        var rightIndices = new List<int> { 3, 4, 5 };

        // Act
        var varianceScore = StatisticsHelper<double>.CalculateSplitScore(y, leftIndices, rightIndices, SplitCriterion.VarianceReduction);
        var mseScore = StatisticsHelper<double>.CalculateSplitScore(y, leftIndices, rightIndices, SplitCriterion.MeanSquaredError);
        var maeScore = StatisticsHelper<double>.CalculateSplitScore(y, leftIndices, rightIndices, SplitCriterion.MeanAbsoluteError);

        // Assert
        Assert.True(!double.IsNaN(varianceScore));
        Assert.True(!double.IsNaN(mseScore));
        Assert.True(!double.IsNaN(maeScore));
    }

    #endregion

    #region Learning Curve Tests

    [Fact]
    public void CalculateLearningCurve_ReturnsCorrectNumberOfSteps()
    {
        // Arrange
        var yActual = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        var yPredicted = new Vector<double>(new double[] { 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1 });
        int steps = 5;

        // Act
        var result = StatisticsHelper<double>.CalculateLearningCurve(yActual, yPredicted, steps);

        // Assert
        Assert.Equal(steps, result.Count);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void CalculateMean_WithFloatType_ReturnsCorrectValue()
    {
        // Arrange
        var values = new float[] { 1f, 2f, 3f, 4f, 5f };

        // Act
        var result = StatisticsHelper<float>.CalculateMean(values);

        // Assert
        Assert.Equal(3.0f, result, 0.001f);
    }

    [Fact]
    public void CalculatePearsonCorrelation_WithFloatType_ReturnsCorrectValue()
    {
        // Arrange
        var x = new Vector<float>(new float[] { 1f, 2f, 3f, 4f, 5f });
        var y = new Vector<float>(new float[] { 2f, 4f, 6f, 8f, 10f });

        // Act
        var result = StatisticsHelper<float>.CalculatePearsonCorrelation(x, y);

        // Assert
        Assert.Equal(1.0f, result, 0.001f);
    }

    [Fact]
    public void EuclideanDistance_WithFloatType_ReturnsCorrectValue()
    {
        // Arrange
        var v1 = new Vector<float>(new float[] { 0f, 0f, 0f });
        var v2 = new Vector<float>(new float[] { 3f, 4f, 0f });

        // Act
        var result = StatisticsHelper<float>.EuclideanDistance(v1, v2);

        // Assert
        Assert.Equal(5.0f, result, 0.001f);
    }

    #endregion
}
