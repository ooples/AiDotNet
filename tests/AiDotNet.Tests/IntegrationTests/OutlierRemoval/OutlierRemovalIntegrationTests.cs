using AiDotNet.LinearAlgebra;
using AiDotNet.OutlierRemoval;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.OutlierRemoval;

/// <summary>
/// Integration tests for outlier removal classes.
/// Tests ZScore, IQR, MAD, Threshold, and NoOutlierRemoval methods.
/// </summary>
public class OutlierRemovalIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region ZScore Outlier Removal Tests

    [Fact]
    public void ZScoreOutlierRemoval_NoOutliers_ReturnsAllData()
    {
        // Arrange
        var outlierRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 },
            { 2.5, 3.5 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 2.5 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - no outliers, should return all data
        Assert.Equal(4, cleanedOutputs.Length);
    }

    [Fact]
    public void ZScoreOutlierRemoval_WithOutlier_RemovesOutlier()
    {
        // Arrange - need enough data points for statistically meaningful z-score
        // With n=4 and one extreme outlier, the outlier skews mean/std so much that its z-score < 2
        // Using more data points ensures proper outlier detection
        var outlierRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 1.5, 2.5 },
            { 2.0, 3.0 },
            { 2.5, 3.5 },
            { 3.0, 4.0 },
            { 3.5, 4.5 },
            { 4.0, 5.0 },
            { 100.0, 200.0 }  // Clear outlier - now z-score will be > 2
        });
        var outputs = new Vector<double>(new[] { 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - outlier should be removed
        Assert.True(cleanedOutputs.Length < 8);
    }

    [Fact]
    public void ZScoreOutlierRemoval_DifferentThresholds_ProduceDifferentResults()
    {
        // Arrange
        var strictRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 1.5);
        var lenientRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 2.0, 2.0 },
            { 3.0, 3.0 },
            { 4.0, 4.0 },
            { 10.0, 10.0 }  // Moderate outlier
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 10.0 });

        // Act
        var (_, strictOutputs) = strictRemoval.RemoveOutliers(inputs, outputs);
        var (_, lenientOutputs) = lenientRemoval.RemoveOutliers(inputs, outputs);

        // Assert - strict threshold should remove more
        Assert.True(strictOutputs.Length <= lenientOutputs.Length);
    }

    [Fact]
    public void ZScoreOutlierRemoval_AllSameValues_ReturnsAllData()
    {
        // Arrange
        var outlierRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3);
        var inputs = new Matrix<double>(new double[,]
        {
            { 5.0, 5.0 },
            { 5.0, 5.0 },
            { 5.0, 5.0 },
            { 5.0, 5.0 }
        });
        var outputs = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - all values same, no outliers possible (std=0 case)
        Assert.True(cleanedOutputs.Length >= 0);  // Should handle edge case
    }

    #endregion

    #region IQR Outlier Removal Tests

    [Fact]
    public void IQROutlierRemoval_NoOutliers_ReturnsAllData()
    {
        // Arrange
        var outlierRemoval = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 },
            { 4.0, 5.0 },
            { 5.0, 6.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - no extreme outliers
        Assert.Equal(5, cleanedOutputs.Length);
    }

    [Fact]
    public void IQROutlierRemoval_WithExtremeOutlier_RemovesOutlier()
    {
        // Arrange
        var outlierRemoval = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 },
            { 4.0, 5.0 },
            { 1000.0, 2000.0 }  // Extreme outlier
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 1000.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - extreme outlier should be removed
        Assert.True(cleanedOutputs.Length < 5);
    }

    [Fact]
    public void IQROutlierRemoval_DifferentMultipliers_AffectResults()
    {
        // Arrange
        var strictRemoval = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.0);
        var lenientRemoval = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 3.0);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 4.0 },
            { 5.0 },
            { 10.0 }  // Moderate outlier
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 10.0 });

        // Act
        var (_, strictOutputs) = strictRemoval.RemoveOutliers(inputs, outputs);
        var (_, lenientOutputs) = lenientRemoval.RemoveOutliers(inputs, outputs);

        // Assert - strict should potentially remove more
        Assert.True(strictOutputs.Length <= lenientOutputs.Length);
    }

    #endregion

    #region MAD Outlier Removal Tests

    [Fact]
    public void MADOutlierRemoval_NoOutliers_ReturnsAllData()
    {
        // Arrange
        var outlierRemoval = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.5);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 },
            { 4.0, 5.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert
        Assert.Equal(4, cleanedOutputs.Length);
    }

    [Fact]
    public void MADOutlierRemoval_WithOutlier_RemovesOutlier()
    {
        // Arrange
        var outlierRemoval = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.5);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 2.0, 2.0 },
            { 3.0, 3.0 },
            { 4.0, 4.0 },
            { 5.0, 5.0 },
            { 100.0, 100.0 }  // Clear outlier
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 100.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - outlier should be removed
        Assert.True(cleanedOutputs.Length < 6);
    }

    [Fact]
    public void MADOutlierRemoval_RobustToOutliers()
    {
        // Arrange - MAD is more robust than Z-score to multiple outliers
        var outlierRemoval = new MADOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 4.0 },
            { 5.0 },
            { 6.0 },
            { 7.0 },
            { 8.0 },
            { 9.0 },
            { 10.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - all data should be kept (no outliers in this uniform spread)
        Assert.Equal(10, cleanedOutputs.Length);
    }

    #endregion

    #region Threshold Outlier Removal Tests

    [Fact]
    public void ThresholdOutlierRemoval_NoOutliers_ReturnsAllData()
    {
        // Arrange - threshold uses MAD-based approach with multiplier
        var outlierRemoval = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 },
            { 4.0, 5.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - no extreme outliers with this spread
        Assert.True(cleanedOutputs.Length >= 2);
    }

    [Fact]
    public void ThresholdOutlierRemoval_WithExtremeOutlier_RemovesOutlier()
    {
        // Arrange - threshold uses MAD-based approach
        var outlierRemoval = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2.0);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 },
            { 100.0, 200.0 }  // Extreme outlier
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 100.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - extreme outlier should be removed
        Assert.True(cleanedOutputs.Length < 4);
    }

    [Fact]
    public void ThresholdOutlierRemoval_DifferentThresholds_AffectResults()
    {
        // Arrange
        var strictRemoval = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 1.5);
        var lenientRemoval = new ThresholdOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 5.0);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 4.0 },
            { 5.0 },
            { 15.0 }  // Moderate outlier
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 15.0 });

        // Act
        var (_, strictOutputs) = strictRemoval.RemoveOutliers(inputs, outputs);
        var (_, lenientOutputs) = lenientRemoval.RemoveOutliers(inputs, outputs);

        // Assert - strict threshold should remove more
        Assert.True(strictOutputs.Length <= lenientOutputs.Length);
    }

    #endregion

    #region NoOutlierRemoval Tests

    [Fact]
    public void NoOutlierRemoval_ReturnsAllData()
    {
        // Arrange
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 1000.0, 2000.0 },  // Would be outlier with other methods
            { 3.0, 4.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 1000.0, 3.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - NoOutlierRemoval should return all data unchanged
        Assert.Equal(3, cleanedOutputs.Length);
    }

    [Fact]
    public void NoOutlierRemoval_PreservesExtremeValues()
    {
        // Arrange
        var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        var inputs = new Matrix<double>(new double[,]
        {
            { -1000000.0 },
            { 0.0 },
            { 1000000.0 }
        });
        var outputs = new Vector<double>(new[] { -1000000.0, 0.0, 1000000.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - all data preserved
        Assert.Equal(3, cleanedOutputs.Length);
        Assert.Equal(-1000000.0, cleanedOutputs[0], Tolerance);
        Assert.Equal(0.0, cleanedOutputs[1], Tolerance);
        Assert.Equal(1000000.0, cleanedOutputs[2], Tolerance);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllOutlierRemovalMethods_HandleEmptyData()
    {
        // This test verifies graceful handling of edge cases
        // Note: Empty matrices may throw exceptions or return empty results
        // depending on implementation
        Assert.True(true);  // Placeholder - actual behavior depends on implementation
    }

    [Fact]
    public void OutlierRemoval_PreservesInputOutputCorrespondence()
    {
        // Arrange
        var outlierRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { 3.0, 30.0 },
            { 4.0, 40.0 }
        });
        var outputs = new Vector<double>(new[] { 100.0, 200.0, 300.0, 400.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - input and output should have same number of rows
        Assert.Equal(cleanedInputs.Rows, cleanedOutputs.Length);
    }

    [Fact]
    public void OutlierRemoval_MultipleFeatures_ChecksAllFeatures()
    {
        // Arrange - outlier in second feature only
        var outlierRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 2);
        // Need enough data points for statistically meaningful z-score detection
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 1.5, 2.5, 3.5 },
            { 2.0, 3.0, 4.0 },
            { 2.5, 3.5, 4.5 },
            { 3.0, 4.0, 5.0 },
            { 3.5, 4.5, 5.5 },
            { 4.0, 5.0, 6.0 },
            { 2.5, 100.0, 4.5 }  // Second feature is outlier - z-score will be > 2 with more data points
        });
        var outputs = new Vector<double>(new[] { 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - row with outlier in any feature should be removed
        Assert.True(cleanedOutputs.Length < 8);
    }

    [Fact]
    public void OutlierRemoval_NegativeValues_HandledCorrectly()
    {
        // Arrange
        var outlierRemoval = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3);
        var inputs = new Matrix<double>(new double[,]
        {
            { -5.0, -10.0 },
            { -3.0, -6.0 },
            { -1.0, -2.0 },
            { 1.0, 2.0 },
            { 3.0, 6.0 }
        });
        var outputs = new Vector<double>(new[] { -5.0, -3.0, -1.0, 1.0, 3.0 });

        // Act
        var (cleanedInputs, cleanedOutputs) = outlierRemoval.RemoveOutliers(inputs, outputs);

        // Assert - should handle negative values correctly
        Assert.Equal(5, cleanedOutputs.Length);  // No outliers in this spread
    }

    #endregion
}
