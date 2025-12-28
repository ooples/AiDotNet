using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for BasicStats with mathematically verified ground truth values.
/// All expected values verified against NumPy/SciPy as authoritative sources.
///
/// These tests ensure the mathematical correctness of fundamental statistics calculations.
/// If any test fails, the CODE must be fixed - never adjust the expected values.
/// </summary>
public class BasicStatsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Helper Methods

    private static BasicStats<T> CreateBasicStats<T>(T[] values)
    {
        var vector = new Vector<T>(values);
        return new BasicStats<T>(new BasicStatsInputs<T> { Values = vector });
    }

    #endregion

    #region Mean Tests

    /// <summary>
    /// Verified with NumPy: np.mean([1, 2, 3, 4, 5]) = 3.0
    /// </summary>
    [Fact]
    public void Mean_SimpleSequence_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Assert - NumPy verified: np.mean([1, 2, 3, 4, 5]) = 3.0
        Assert.Equal(3.0, stats.Mean, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean([2.5, 3.7, 4.1, 5.9, 6.3]) = 4.5
    /// </summary>
    [Fact]
    public void Mean_DecimalValues_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 2.5, 3.7, 4.1, 5.9, 6.3 });

        // Assert - NumPy verified: np.mean([2.5, 3.7, 4.1, 5.9, 6.3]) = 4.5
        Assert.Equal(4.5, stats.Mean, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean([-5, -3, 0, 3, 5]) = 0.0
    /// </summary>
    [Fact]
    public void Mean_MixedPositiveNegative_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { -5.0, -3.0, 0.0, 3.0, 5.0 });

        // Assert - NumPy verified: np.mean([-5, -3, 0, 3, 5]) = 0.0
        Assert.Equal(0.0, stats.Mean, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean([42]) = 42.0
    /// </summary>
    [Fact]
    public void Mean_SingleValue_ReturnsValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 42.0 });

        // Assert
        Assert.Equal(42.0, stats.Mean, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.mean([1e10, 2e10, 3e10]) = 2e10
    /// </summary>
    [Fact]
    public void Mean_LargeValues_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1e10, 2e10, 3e10 });

        // Assert - NumPy verified
        Assert.Equal(2e10, stats.Mean, 1e4); // Larger tolerance for large numbers
    }

    /// <summary>
    /// Verified with NumPy: np.mean([1e-10, 2e-10, 3e-10]) = 2e-10
    /// </summary>
    [Fact]
    public void Mean_SmallValues_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1e-10, 2e-10, 3e-10 });

        // Assert - NumPy verified
        Assert.Equal(2e-10, stats.Mean, 1e-16);
    }

    #endregion

    #region Variance Tests

    /// <summary>
    /// Verified with NumPy: np.var([1, 2, 3, 4, 5], ddof=0) = 2.0 (population variance)
    /// Note: BasicStats uses population variance (ddof=0), not sample variance (ddof=1).
    /// </summary>
    [Fact]
    public void Variance_SimpleSequence_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Assert - Population variance: np.var([1, 2, 3, 4, 5], ddof=0) = 2.0
        Assert.Equal(2.0, stats.Variance, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.var([5, 5, 5, 5, 5]) = 0.0
    /// </summary>
    [Fact]
    public void Variance_AllSameValues_ReturnsZero()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 5.0, 5.0, 5.0, 5.0, 5.0 });

        // Assert - Zero variance when all values are identical
        Assert.Equal(0.0, stats.Variance, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.var([10, 20, 30, 40, 50], ddof=0) = 200.0
    /// </summary>
    [Fact]
    public void Variance_LargerSpread_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        // Assert - NumPy verified: np.var([10, 20, 30, 40, 50], ddof=0) = 200.0
        Assert.Equal(200.0, stats.Variance, Tolerance);
    }

    #endregion

    #region Standard Deviation Tests

    /// <summary>
    /// Verified with NumPy:
    /// np.std([1, 2, 3, 4, 5], ddof=0) = 1.4142135623730951 (population)
    /// </summary>
    [Fact]
    public void StandardDeviation_SimpleSequence_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Assert - NumPy verified: np.std([1, 2, 3, 4, 5], ddof=0) = 1.4142135623730951
        Assert.Equal(1.4142135623730951, stats.StandardDeviation, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=0) = 2.0
    /// </summary>
    [Fact]
    public void StandardDeviation_KnownDataset_ReturnsExactValue()
    {
        // Arrange - Classic dataset with known std dev = 2.0
        var stats = CreateBasicStats(new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 });

        // Assert - NumPy verified: np.std([2, 4, 4, 4, 5, 5, 7, 9], ddof=0) = 2.0
        Assert.Equal(2.0, stats.StandardDeviation, Tolerance);
    }

    #endregion

    #region Min/Max Tests

    /// <summary>
    /// Verified with NumPy: np.min/max([3, 1, 4, 1, 5, 9, 2, 6]) = 1, 9
    /// </summary>
    [Fact]
    public void MinMax_UnsortedData_ReturnsCorrectValues()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0 });

        // Assert
        Assert.Equal(1.0, stats.Min, Tolerance);
        Assert.Equal(9.0, stats.Max, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.min/max([-10, -5, 0, 5, 10]) = -10, 10
    /// </summary>
    [Fact]
    public void MinMax_NegativeValues_ReturnsCorrectValues()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { -10.0, -5.0, 0.0, 5.0, 10.0 });

        // Assert
        Assert.Equal(-10.0, stats.Min, Tolerance);
        Assert.Equal(10.0, stats.Max, Tolerance);
    }

    #endregion

    #region Count Tests

    [Fact]
    public void Count_ReturnsCorrectCount()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Assert
        Assert.Equal(5, stats.N);
    }

    #endregion

    #region Skewness Tests

    /// <summary>
    /// Verified with SciPy: scipy.stats.skew([1, 2, 3, 4, 5], bias=False) = 0.0
    /// Symmetric distribution has zero skewness.
    /// </summary>
    [Fact]
    public void Skewness_SymmetricDistribution_ReturnsZero()
    {
        // Arrange - Symmetric distribution
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Assert - Symmetric data has skewness = 0
        Assert.Equal(0.0, stats.Skewness, Tolerance);
    }

    /// <summary>
    /// Verified with SciPy: scipy.stats.skew([1, 1, 1, 1, 5], bias=False) > 0
    /// Right-skewed distribution has positive skewness.
    /// </summary>
    [Fact]
    public void Skewness_RightSkewed_ReturnsPositive()
    {
        // Arrange - Right-skewed distribution
        var stats = CreateBasicStats(new[] { 1.0, 1.0, 1.0, 1.0, 5.0 });

        // Assert - Right-skewed has positive skewness
        Assert.True(stats.Skewness > 0, $"Right-skewed distribution should have positive skewness, got {stats.Skewness}");
    }

    /// <summary>
    /// Verified with SciPy: scipy.stats.skew([1, 5, 5, 5, 5], bias=False) < 0
    /// Left-skewed distribution has negative skewness.
    /// </summary>
    [Fact]
    public void Skewness_LeftSkewed_ReturnsNegative()
    {
        // Arrange - Left-skewed distribution
        var stats = CreateBasicStats(new[] { 1.0, 5.0, 5.0, 5.0, 5.0 });

        // Assert - Left-skewed has negative skewness
        Assert.True(stats.Skewness < 0, $"Left-skewed distribution should have negative skewness, got {stats.Skewness}");
    }

    #endregion

    #region Kurtosis Tests

    /// <summary>
    /// Verified with SciPy: Normal distribution has excess kurtosis = 0.
    /// scipy.stats.kurtosis([...], fisher=True) returns excess kurtosis.
    /// </summary>
    [Fact]
    public void Kurtosis_UniformDistribution_IsFinite()
    {
        // Arrange - Uniform distribution has negative excess kurtosis (platykurtic)
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

        // Assert - Kurtosis should be a finite value
        Assert.True(!double.IsNaN(stats.Kurtosis) && !double.IsInfinity(stats.Kurtosis), "Kurtosis should be a finite value");
    }

    #endregion

    #region Median Tests

    /// <summary>
    /// Verified with NumPy: np.median([1, 2, 3, 4, 5]) = 3.0 (odd count)
    /// </summary>
    [Fact]
    public void Median_OddCount_ReturnsMiddleValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Assert - NumPy verified: np.median([1, 2, 3, 4, 5]) = 3.0
        Assert.Equal(3.0, stats.Median, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.median([1, 2, 3, 4, 5, 6]) = 3.5 (even count)
    /// </summary>
    [Fact]
    public void Median_EvenCount_ReturnsAverageOfMiddleTwo()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

        // Assert - NumPy verified: np.median([1, 2, 3, 4, 5, 6]) = 3.5
        Assert.Equal(3.5, stats.Median, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.median([5, 2, 8, 1, 9]) = 5.0
    /// </summary>
    [Fact]
    public void Median_UnsortedData_ReturnsCorrectValue()
    {
        // Arrange - Unsorted data
        var stats = CreateBasicStats(new[] { 5.0, 2.0, 8.0, 1.0, 9.0 });

        // Assert - NumPy verified: np.median([5, 2, 8, 1, 9]) = 5.0
        Assert.Equal(5.0, stats.Median, Tolerance);
    }

    #endregion

    #region Quartile Tests

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5, 6, 7], [25, 75]) = [2.5, 5.5]
    /// </summary>
    [Fact]
    public void Quartiles_SevenValues_ReturnsExactValues()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        // Assert - NumPy verified
        Assert.Equal(2.5, stats.FirstQuartile, Tolerance);
        Assert.Equal(5.5, stats.ThirdQuartile, Tolerance);
    }

    /// <summary>
    /// IQR = Q3 - Q1
    /// </summary>
    [Fact]
    public void InterquartileRange_ReturnsQ3MinusQ1()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        // Assert - IQR = 5.5 - 2.5 = 3.0
        var expectedIQR = stats.ThirdQuartile - stats.FirstQuartile;
        Assert.Equal(expectedIQR, stats.InterquartileRange, Tolerance);
    }

    #endregion

    #region MAD (Median Absolute Deviation) Tests

    /// <summary>
    /// MAD = median(|x_i - median(x)|)
    /// For [1, 2, 3, 4, 5]: median=3, deviations=[2, 1, 0, 1, 2], MAD=1.0
    /// </summary>
    [Fact]
    public void MAD_SimpleSequence_ReturnsExactValue()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Assert - MAD = median([|1-3|, |2-3|, |3-3|, |4-3|, |5-3|]) = median([2, 1, 0, 1, 2]) = 1.0
        Assert.Equal(1.0, stats.MAD, Tolerance);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void Mean_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var vector = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        var stats = new BasicStats<float>(new BasicStatsInputs<float> { Values = vector });

        // Assert
        Assert.Equal(3.0f, stats.Mean, 1e-5f);
    }

    [Fact]
    public void Variance_FloatType_ReturnsCorrectValue()
    {
        // Arrange
        var vector = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
        var stats = new BasicStats<float>(new BasicStatsInputs<float> { Values = vector });

        // Assert - Population variance: np.var([1, 2, 3, 4, 5], ddof=0) = 2.0
        Assert.Equal(2.0f, stats.Variance, 1e-5f);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void TwoValues_CalculatesCorrectly()
    {
        // Arrange & Act
        var stats = CreateBasicStats(new[] { 10.0, 20.0 });

        // Assert
        Assert.Equal(15.0, stats.Mean, Tolerance);
        // Population variance: ((10-15)^2 + (20-15)^2) / 2 = 50 / 2 = 25
        Assert.Equal(25.0, stats.Variance, Tolerance);
    }

    [Fact]
    public void Empty_ReturnsZeroValues()
    {
        // Arrange & Act
        var stats = BasicStats<double>.Empty();

        // Assert
        Assert.Equal(0, stats.N);
        Assert.Equal(0.0, stats.Mean, Tolerance);
    }

    #endregion
}
