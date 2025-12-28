using AiDotNet.LinearAlgebra;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for Quartile with mathematically verified ground truth values.
/// All expected values verified against NumPy as authoritative source.
///
/// NumPy uses linear interpolation (method='linear') by default for percentiles.
/// These tests verify our implementation matches NumPy's behavior.
///
/// If any test fails, the CODE must be fixed - never adjust the expected values.
/// </summary>
public class QuartileIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Q1 (25th Percentile) Tests

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5, 6, 7], 25) = 2.5
    /// </summary>
    [Fact]
    public void Q1_SevenValues_ReturnsExactValue()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([1, 2, 3, 4, 5, 6, 7], 25) = 2.5
        Assert.Equal(2.5, quartile.Q1, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5, 6, 7, 8], 25) = 2.75
    /// </summary>
    [Fact]
    public void Q1_EightValues_ReturnsExactValue()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([1, 2, 3, 4, 5, 6, 7, 8], 25) = 2.75
        Assert.Equal(2.75, quartile.Q1, Tolerance);
    }

    #endregion

    #region Q2 (Median / 50th Percentile) Tests

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5, 6, 7], 50) = 4.0
    /// </summary>
    [Fact]
    public void Q2_SevenValues_ReturnsExactValue()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([1, 2, 3, 4, 5, 6, 7], 50) = 4.0
        Assert.Equal(4.0, quartile.Q2, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5, 6], 50) = 3.5
    /// </summary>
    [Fact]
    public void Q2_SixValues_ReturnsExactValue()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([1, 2, 3, 4, 5, 6], 50) = 3.5
        Assert.Equal(3.5, quartile.Q2, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.median([5, 2, 8, 1, 9, 3, 7]) = 5.0
    /// </summary>
    [Fact]
    public void Q2_UnsortedData_ReturnsCorrectMedian()
    {
        // Arrange - Unsorted data
        var data = new Vector<double>(new[] { 5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.median([5, 2, 8, 1, 9, 3, 7]) = 5.0
        Assert.Equal(5.0, quartile.Q2, Tolerance);
    }

    #endregion

    #region Q3 (75th Percentile) Tests

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5, 6, 7], 75) = 5.5
    /// </summary>
    [Fact]
    public void Q3_SevenValues_ReturnsExactValue()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([1, 2, 3, 4, 5, 6, 7], 75) = 5.5
        Assert.Equal(5.5, quartile.Q3, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5, 6, 7, 8], 75) = 6.25
    /// </summary>
    [Fact]
    public void Q3_EightValues_ReturnsExactValue()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([1, 2, 3, 4, 5, 6, 7, 8], 75) = 6.25
        Assert.Equal(6.25, quartile.Q3, Tolerance);
    }

    #endregion

    #region Complete Quartile Set Tests

    /// <summary>
    /// Verified with NumPy for data 1-100:
    /// np.percentile(range(1, 101), [25, 50, 75]) = [25.75, 50.5, 75.25]
    /// </summary>
    [Fact]
    public void AllQuartiles_LargeDataset_ReturnsExactValues()
    {
        // Arrange - Create data from 1 to 100
        var values = new double[100];
        for (int i = 0; i < 100; i++)
        {
            values[i] = i + 1.0;
        }
        var data = new Vector<double>(values);

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile(range(1, 101), [25, 50, 75])
        Assert.Equal(25.75, quartile.Q1, Tolerance);
        Assert.Equal(50.5, quartile.Q2, Tolerance);
        Assert.Equal(75.25, quartile.Q3, Tolerance);
    }

    /// <summary>
    /// When all values are the same, all quartiles should equal that value.
    /// </summary>
    [Fact]
    public void AllQuartiles_ConstantData_AllEqual()
    {
        // Arrange
        var data = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0, 5.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert
        Assert.Equal(5.0, quartile.Q1, Tolerance);
        Assert.Equal(5.0, quartile.Q2, Tolerance);
        Assert.Equal(5.0, quartile.Q3, Tolerance);
    }

    /// <summary>
    /// Quartiles must maintain Q1 <= Q2 <= Q3 relationship.
    /// </summary>
    [Fact]
    public void AllQuartiles_MaintainOrder()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert
        Assert.True(quartile.Q1 <= quartile.Q2, $"Q1 ({quartile.Q1}) should be <= Q2 ({quartile.Q2})");
        Assert.True(quartile.Q2 <= quartile.Q3, $"Q2 ({quartile.Q2}) should be <= Q3 ({quartile.Q3})");
    }

    #endregion

    #region Negative Values Tests

    /// <summary>
    /// Verified with NumPy: np.percentile([-7, -5, -3, -1, 1, 3, 5], [25, 50, 75]) = [-4., -1., 2.]
    /// Linear interpolation:
    /// - Q1: position = 6*0.25 = 1.5, between -5 and -3: -5 + 0.5*2 = -4
    /// - Q2: position = 6*0.5 = 3.0, exact: -1
    /// - Q3: position = 6*0.75 = 4.5, between 1 and 3: 1 + 0.5*2 = 2
    /// </summary>
    [Fact]
    public void AllQuartiles_NegativeValues_ReturnsExactValues()
    {
        // Arrange
        var data = new Vector<double>(new[] { -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([-7, -5, -3, -1, 1, 3, 5], [25, 50, 75])
        Assert.Equal(-4.0, quartile.Q1, Tolerance);
        Assert.Equal(-1.0, quartile.Q2, Tolerance);
        Assert.Equal(2.0, quartile.Q3, Tolerance);
    }

    #endregion

    #region Small Dataset Tests

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3], [25, 50, 75]) = [1.5, 2.0, 2.5]
    /// </summary>
    [Fact]
    public void AllQuartiles_ThreeValues_ReturnsExactValues()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified: np.percentile([1, 2, 3], [25, 50, 75])
        Assert.Equal(1.5, quartile.Q1, Tolerance);
        Assert.Equal(2.0, quartile.Q2, Tolerance);
        Assert.Equal(2.5, quartile.Q3, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4], [25, 50, 75]) = [1.75, 2.5, 3.25]
    /// </summary>
    [Fact]
    public void AllQuartiles_FourValues_ReturnsExactValues()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified
        Assert.Equal(1.75, quartile.Q1, Tolerance);
        Assert.Equal(2.5, quartile.Q2, Tolerance);
        Assert.Equal(3.25, quartile.Q3, Tolerance);
    }

    /// <summary>
    /// Verified with NumPy: np.percentile([1, 2, 3, 4, 5], [25, 50, 75]) = [2.0, 3.0, 4.0]
    /// </summary>
    [Fact]
    public void AllQuartiles_FiveValues_ReturnsExactValues()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified
        Assert.Equal(2.0, quartile.Q1, Tolerance);
        Assert.Equal(3.0, quartile.Q2, Tolerance);
        Assert.Equal(4.0, quartile.Q3, Tolerance);
    }

    #endregion

    #region Decimal Values Tests

    /// <summary>
    /// Verified with NumPy: np.percentile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [25, 50, 75]) = [0.25, 0.4, 0.55]
    /// </summary>
    [Fact]
    public void AllQuartiles_DecimalValues_ReturnsExactValues()
    {
        // Arrange
        var data = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 });

        // Act
        var quartile = new Quartile<double>(data);

        // Assert - NumPy verified
        Assert.Equal(0.25, quartile.Q1, Tolerance);
        Assert.Equal(0.4, quartile.Q2, Tolerance);
        Assert.Equal(0.55, quartile.Q3, Tolerance);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void AllQuartiles_FloatType_ReturnsCorrectValues()
    {
        // Arrange
        var data = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f });

        // Act
        var quartile = new Quartile<float>(data);

        // Assert
        Assert.Equal(2.5f, quartile.Q1, 1e-5f);
        Assert.Equal(4.0f, quartile.Q2, 1e-5f);
        Assert.Equal(5.5f, quartile.Q3, 1e-5f);
    }

    #endregion

    #region IQR (Interquartile Range) Tests

    /// <summary>
    /// IQR = Q3 - Q1. This is a fundamental statistical measure.
    /// Verified with NumPy: np.percentile([1,2,3,4,5,6,7], 75) - np.percentile([1,2,3,4,5,6,7], 25) = 3.0
    /// </summary>
    [Fact]
    public void IQR_CanBeCalculatedFromQ1AndQ3()
    {
        // Arrange
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 });

        // Act
        var quartile = new Quartile<double>(data);
        var iqr = quartile.Q3 - quartile.Q1;

        // Assert - IQR = 5.5 - 2.5 = 3.0
        Assert.Equal(3.0, iqr, Tolerance);
    }

    #endregion
}
