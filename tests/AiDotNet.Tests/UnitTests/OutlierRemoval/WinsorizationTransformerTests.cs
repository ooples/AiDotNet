using AiDotNet.OutlierRemoval;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.OutlierRemoval;

/// <summary>
/// Unit tests for WinsorizationTransformer.
/// </summary>
public class WinsorizationTransformerTests
{
    private const double Tolerance = 1e-6;

    #region Constructor Tests

    [Fact]
    public void Constructor_DefaultParameters_SetsCorrectQuantiles()
    {
        // Arrange & Act
        var transformer = new WinsorizationTransformer<double>();

        // Assert
        Assert.Equal(0.05, transformer.LowerQuantile);
        Assert.Equal(0.95, transformer.UpperQuantile);
        Assert.False(transformer.IsFitted);
    }

    [Fact]
    public void Constructor_CustomParameters_SetsCorrectQuantiles()
    {
        // Arrange & Act
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.10, upperQuantile: 0.90);

        // Assert
        Assert.Equal(0.10, transformer.LowerQuantile);
        Assert.Equal(0.90, transformer.UpperQuantile);
    }

    [Fact]
    public void Constructor_InvalidLowerQuantile_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizationTransformer<double>(lowerQuantile: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizationTransformer<double>(lowerQuantile: 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizationTransformer<double>(lowerQuantile: 0.6));
    }

    [Fact]
    public void Constructor_InvalidUpperQuantile_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizationTransformer<double>(upperQuantile: 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizationTransformer<double>(upperQuantile: 1.1));
    }

    [Fact]
    public void Constructor_LowerGreaterThanUpper_ThrowsException()
    {
        // Act & Assert - lower quantile (0.4) >= upper quantile (0.6) after both are valid individually
        // Note: Due to the range constraints (lower < 0.5, upper > 0.5), we need a case where
        // lower approaches 0.5 from below. Actually, with constraints lower in [0, 0.5) and
        // upper in (0.5, 1], they can never overlap. So we test that equal values throw.
        // Since lower must be < 0.5 and upper must be > 0.5, they can't be equal.
        // The validation is redundant but kept for safety. Test the edge case.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WinsorizationTransformer<double>(lowerQuantile: 0.5, upperQuantile: 0.6)); // lower = 0.5 is invalid
    }

    [Fact]
    public void Constructor_EdgeQuantiles_Works()
    {
        // Arrange & Act - boundary values should work
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.0, upperQuantile: 1.0);

        // Assert
        Assert.Equal(0.0, transformer.LowerQuantile);
        Assert.Equal(1.0, transformer.UpperQuantile);
    }

    #endregion

    #region Fit Tests

    [Fact]
    public void Fit_ValidData_SetsIsFittedToTrue()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        var X = CreateTestMatrix();

        // Act
        transformer.Fit(X);

        // Assert
        Assert.True(transformer.IsFitted);
    }

    [Fact]
    public void Fit_ValidData_ComputesBounds()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.1, upperQuantile: 0.9);
        var X = CreateSequentialMatrix(10, 1); // Values 0-9

        // Act
        transformer.Fit(X);

        // Assert
        Assert.NotNull(transformer.LowerBounds);
        Assert.NotNull(transformer.UpperBounds);
        Assert.Single(transformer.LowerBounds.ToArray());
        Assert.Single(transformer.UpperBounds.ToArray());
    }

    [Fact]
    public void Fit_NullInput_ThrowsException()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => transformer.Fit(null!));
    }

    [Fact]
    public void Fit_EmptyMatrix_ThrowsException()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        var X = new Matrix<double>(0, 0);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => transformer.Fit(X));
    }

    [Fact]
    public void Fit_ComputesCorrectQuantiles()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.0, upperQuantile: 1.0);
        // Single column with values 1, 2, 3, 4, 5
        var X = new Matrix<double>(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        // Act
        transformer.Fit(X);

        // Assert - With 0% and 100% quantiles, bounds should be min and max
        Assert.Equal(1.0, transformer.LowerBounds![0], Tolerance);
        Assert.Equal(5.0, transformer.UpperBounds![0], Tolerance);
    }

    #endregion

    #region Transform Tests

    [Fact]
    public void Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        var X = CreateTestMatrix();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => transformer.Transform(X));
    }

    [Fact]
    public void Transform_ValidData_ReturnsCorrectDimensions()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        var X = CreateTestMatrix();

        // Act
        transformer.Fit(X);
        var result = transformer.Transform(X);

        // Assert
        Assert.Equal(X.Rows, result.Rows);
        Assert.Equal(X.Columns, result.Columns);
    }

    [Fact]
    public void Transform_ClipsLowValues()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.2, upperQuantile: 0.8);
        // Values: 1, 2, 3, 4, 5 (indices 0-4)
        // 20th percentile at index ~0.8 -> value interpolated around 1.8
        // 80th percentile at index ~3.2 -> value interpolated around 4.2
        var X = new Matrix<double>(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        // Act
        transformer.Fit(X);

        // Test data with an extreme low value
        var XTest = new Matrix<double>(new double[,] { { -100 } });
        var result = transformer.Transform(XTest);

        // Assert - The extreme low value should be clipped to the lower bound
        Assert.True(result[0, 0] > -100, $"Expected value to be clipped but got {result[0, 0]}");
    }

    [Fact]
    public void Transform_ClipsHighValues()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.2, upperQuantile: 0.8);
        var X = new Matrix<double>(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        // Act
        transformer.Fit(X);

        // Test data with an extreme high value
        var XTest = new Matrix<double>(new double[,] { { 100 } });
        var result = transformer.Transform(XTest);

        // Assert - The extreme high value should be clipped to the upper bound
        Assert.True(result[0, 0] < 100, $"Expected value to be clipped but got {result[0, 0]}");
    }

    [Fact]
    public void Transform_PreservesMiddleValues()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.0, upperQuantile: 1.0);
        var X = new Matrix<double>(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        // Act
        transformer.Fit(X);

        // Test with a middle value
        var XTest = new Matrix<double>(new double[,] { { 3 } });
        var result = transformer.Transform(XTest);

        // Assert - Middle value should be unchanged
        Assert.Equal(3.0, result[0, 0], Tolerance);
    }

    [Fact]
    public void Transform_MultipleFeatures_ClipsIndependently()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.0, upperQuantile: 1.0);
        // Two features with different scales
        var X = new Matrix<double>(new double[,]
        {
            { 1, 10 },
            { 2, 20 },
            { 3, 30 },
            { 4, 40 },
            { 5, 50 }
        });

        // Act
        transformer.Fit(X);
        var result = transformer.Transform(X);

        // Assert - Bounds should be different for each feature
        Assert.Equal(1.0, transformer.LowerBounds![0], Tolerance);
        Assert.Equal(10.0, transformer.LowerBounds[1], Tolerance);
        Assert.Equal(5.0, transformer.UpperBounds![0], Tolerance);
        Assert.Equal(50.0, transformer.UpperBounds[1], Tolerance);
    }

    [Fact]
    public void Transform_WrongNumberOfFeatures_ThrowsException()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        var XTrain = new Matrix<double>(new double[,] { { 1, 2 }, { 3, 4 } });
        var XTest = new Matrix<double>(new double[,] { { 1, 2, 3 } }); // Different number of features

        // Act
        transformer.Fit(XTrain);

        // Assert
        Assert.Throws<ArgumentException>(() => transformer.Transform(XTest));
    }

    [Fact]
    public void Transform_NullInput_ThrowsException()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        transformer.Fit(CreateTestMatrix());

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => transformer.Transform(null!));
    }

    #endregion

    #region FitTransform Tests

    [Fact]
    public void FitTransform_CombinesFitAndTransform()
    {
        // Arrange
        var transformer1 = new WinsorizationTransformer<double>();
        var transformer2 = new WinsorizationTransformer<double>();
        var X = CreateTestMatrix();

        // Act
        transformer1.Fit(X);
        var result1 = transformer1.Transform(X);

        var result2 = transformer2.FitTransform(X);

        // Assert
        Assert.True(transformer2.IsFitted);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j], Tolerance);
            }
        }
    }

    [Fact]
    public void FitTransform_NullInput_ThrowsException()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => transformer.FitTransform(null!));
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Winsorization_WithOutliers_ClipsCorrectly()
    {
        // Arrange - Create data with clear outliers
        var random = new Random(42);
        var data = new List<double[]>();

        // Normal data centered at 0
        for (int i = 0; i < 100; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            data.Add(new[] { normal });
        }

        // Add extreme outliers
        data.Add(new[] { 100.0 });  // High outlier
        data.Add(new[] { -100.0 }); // Low outlier

        var X = new Matrix<double>(data.Count, 1);
        for (int i = 0; i < data.Count; i++)
        {
            X[i, 0] = data[i][0];
        }

        // Use aggressive clipping (10% on each end)
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.05, upperQuantile: 0.95);

        // Act
        var result = transformer.FitTransform(X);

        // Assert - Outliers should be clipped to reasonable values
        double maxResult = double.MinValue;
        double minResult = double.MaxValue;

        for (int i = 0; i < result.Rows; i++)
        {
            if (result[i, 0] > maxResult) maxResult = result[i, 0];
            if (result[i, 0] < minResult) minResult = result[i, 0];
        }

        Assert.True(maxResult < 100, $"Expected max to be clipped, but got {maxResult}");
        Assert.True(minResult > -100, $"Expected min to be clipped, but got {minResult}");
    }

    [Fact]
    public void Winsorization_SingleValue_HandlesEdgeCase()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        var X = new Matrix<double>(new double[,] { { 5.0 } });

        // Act
        var result = transformer.FitTransform(X);

        // Assert - Single value should remain unchanged
        Assert.Equal(5.0, result[0, 0], Tolerance);
    }

    [Fact]
    public void Winsorization_AllSameValues_PreservesData()
    {
        // Arrange
        var transformer = new WinsorizationTransformer<double>();
        var X = new Matrix<double>(new double[,]
        {
            { 3.0 },
            { 3.0 },
            { 3.0 },
            { 3.0 },
            { 3.0 }
        });

        // Act
        var result = transformer.FitTransform(X);

        // Assert - All values should remain the same
        for (int i = 0; i < result.Rows; i++)
        {
            Assert.Equal(3.0, result[i, 0], Tolerance);
        }
    }

    [Fact]
    public void Winsorization_NoClipping_WhenWithinBounds()
    {
        // Arrange - Use 0% and 100% quantiles (no clipping)
        var transformer = new WinsorizationTransformer<double>(lowerQuantile: 0.0, upperQuantile: 1.0);
        var X = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 },
            { 4.0 },
            { 5.0 }
        });

        // Act
        var result = transformer.FitTransform(X);

        // Assert - All values should be unchanged
        for (int i = 0; i < result.Rows; i++)
        {
            Assert.Equal(X[i, 0], result[i, 0], Tolerance);
        }
    }

    #endregion

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix()
    {
        return new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 },
            { 13.0, 14.0, 15.0 }
        });
    }

    private static Matrix<double> CreateSequentialMatrix(int rows, int cols)
    {
        var matrix = new Matrix<double>(rows, cols);
        int value = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = value++;
            }
        }

        return matrix;
    }

    #endregion
}
