using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Integration tests for Scalers (StandardScaler, MinMaxScaler, etc.).
/// Tests transformation accuracy, inverse transformation, and edge cases.
/// </summary>
public class ScalerIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(double[,] data)
    {
        return new Matrix<double>(data);
    }

    private static void AssertMatrixEqual(Matrix<double> expected, Matrix<double> actual, double tolerance = Tolerance)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);

        for (int i = 0; i < expected.Rows; i++)
        {
            for (int j = 0; j < expected.Columns; j++)
            {
                Assert.True(
                    Math.Abs(expected[i, j] - actual[i, j]) < tolerance,
                    $"Mismatch at [{i},{j}]: expected {expected[i, j]}, actual {actual[i, j]}");
            }
        }
    }

    #endregion

    #region StandardScaler Tests

    [Fact]
    public void StandardScaler_DefaultParameters_CentersAndScalesData()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Mean should be ~0, Std should be ~1
        for (int j = 0; j < result.Columns; j++)
        {
            double mean = 0;
            for (int i = 0; i < result.Rows; i++)
            {
                mean += result[i, j];
            }
            mean /= result.Rows;

            double variance = 0;
            for (int i = 0; i < result.Rows; i++)
            {
                variance += Math.Pow(result[i, j] - mean, 2);
            }
            // AiDotNet uses sample variance (N-1 denominator, Bessel's correction)
            // Note: This differs from sklearn which uses population variance (N denominator, ddof=0)
            variance /= (result.Rows - 1);
            double std = Math.Sqrt(variance);

            Assert.True(Math.Abs(mean) < 1e-10, $"Column {j} mean should be ~0, got {mean}");
            Assert.True(Math.Abs(std - 1.0) < 1e-10, $"Column {j} sample std should be ~1, got {std}");
        }
    }

    [Fact]
    public void StandardScaler_WithMeanOnly_CentersButDoesNotScale()
    {
        // Arrange
        var scaler = new StandardScaler<double>(withMean: true, withStd: false);
        var data = CreateTestMatrix(new double[,] { { 1 }, { 3 }, { 5 } }); // mean = 3

        // Act
        var result = scaler.FitTransform(data);

        // Assert
        Assert.True(Math.Abs(result[0, 0] - (-2.0)) < 1e-10); // 1 - 3 = -2
        Assert.True(Math.Abs(result[1, 0] - 0.0) < 1e-10);    // 3 - 3 = 0
        Assert.True(Math.Abs(result[2, 0] - 2.0) < 1e-10);    // 5 - 3 = 2
    }

    [Fact]
    public void StandardScaler_WithStdOnly_ScalesButDoesNotCenter()
    {
        // Arrange
        var scaler = new StandardScaler<double>(withMean: false, withStd: true);
        var data = CreateTestMatrix(new double[,] { { 0 }, { 10 }, { 20 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Values should be scaled but not centered
        // Original mean = 10, std = sqrt((100+0+100)/3) = sqrt(66.67)
        // Values should be: 0/std, 10/std, 20/std (no mean subtraction)
        Assert.True(Math.Abs(result[0, 0]) < 1e-10); // 0 / anything = 0
    }

    [Fact]
    public void StandardScaler_InverseTransform_RecoversOriginalData()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 100 }, { 2, 200 }, { 3, 300 } });

        // Act
        var transformed = scaler.FitTransform(data);
        var inversed = scaler.InverseTransform(transformed);

        // Assert
        AssertMatrixEqual(data, inversed, 1e-9);
    }

    [Fact]
    public void StandardScaler_SupportsInverseTransform_ReturnsTrue()
    {
        // Arrange
        var scaler = new StandardScaler<double>();

        // Assert
        Assert.True(scaler.SupportsInverseTransform);
    }

    [Fact]
    public void StandardScaler_ConstantColumn_HandlesGracefully()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 5, 1 }, { 5, 2 }, { 5, 3 } }); // Column 0 is constant

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Constant column should not cause NaN or Infinity
        for (int i = 0; i < result.Rows; i++)
        {
            Assert.False(double.IsNaN(result[i, 0]), $"NaN at [{i},0]");
            Assert.False(double.IsInfinity(result[i, 0]), $"Infinity at [{i},0]");
        }
    }

    [Fact]
    public void StandardScaler_SpecificColumns_TransformsOnlySpecifiedColumns()
    {
        // Arrange
        var scaler = new StandardScaler<double>(columnIndices: new[] { 1 }); // Only column 1
        var data = CreateTestMatrix(new double[,] { { 100, 1 }, { 200, 3 }, { 300, 5 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Column 0 should be unchanged, column 1 should be standardized
        Assert.Equal(100, result[0, 0]);
        Assert.Equal(200, result[1, 0]);
        Assert.Equal(300, result[2, 0]);

        // Column 1 should have mean ~0
        double col1Mean = (result[0, 1] + result[1, 1] + result[2, 1]) / 3;
        Assert.True(Math.Abs(col1Mean) < 1e-10);
    }

    [Fact]
    public void StandardScaler_Mean_ReturnsCorrectValues()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 } });

        // Act
        scaler.Fit(data);

        // Assert
        Assert.NotNull(scaler.Mean);
        Assert.True(Math.Abs(scaler.Mean[0] - 2.0) < 1e-10);  // Mean of [1,2,3] = 2
        Assert.True(Math.Abs(scaler.Mean[1] - 20.0) < 1e-10); // Mean of [10,20,30] = 20
    }

    [Fact]
    public void StandardScaler_StandardDeviation_ReturnsCorrectValues()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        // Using values where sample std is easy to calculate
        // For [0, 1, 2]: mean=1, sample var = (1+0+1)/(3-1) = 1, sample std = 1
        var data = CreateTestMatrix(new double[,] { { 0 }, { 1 }, { 2 } });

        // Act
        scaler.Fit(data);

        // Assert - AiDotNet StandardScaler uses sample std (N-1 denominator, Bessel's correction)
        // Note: This differs from sklearn which uses population variance (N denominator, ddof=0)
        Assert.NotNull(scaler.StandardDeviation);
        double expectedStd = 1.0; // Sample std of [0, 1, 2] = sqrt(2/2) = 1
        Assert.True(Math.Abs(scaler.StandardDeviation[0] - expectedStd) < 1e-10,
            $"Expected sample std {expectedStd}, got {scaler.StandardDeviation[0]}");
    }

    #endregion

    #region MinMaxScaler Tests

    [Fact]
    public void MinMaxScaler_DefaultRange_ScalesToZeroOne()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 0, 100 }, { 5, 200 }, { 10, 300 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - All values should be in [0, 1]
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.True(result[i, j] >= -1e-10 && result[i, j] <= 1.0 + 1e-10,
                    $"Value at [{i},{j}] = {result[i, j]} should be in [0, 1]");
            }
        }

        // Min should map to 0, max should map to 1
        Assert.True(Math.Abs(result[0, 0] - 0.0) < 1e-10); // 0 -> 0
        Assert.True(Math.Abs(result[2, 0] - 1.0) < 1e-10); // 10 -> 1
        Assert.True(Math.Abs(result[0, 1] - 0.0) < 1e-10); // 100 -> 0
        Assert.True(Math.Abs(result[2, 1] - 1.0) < 1e-10); // 300 -> 1
    }

    [Fact]
    public void MinMaxScaler_CustomRange_ScalesToSpecifiedRange()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        var data = CreateTestMatrix(new double[,] { { 0 }, { 5 }, { 10 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert
        Assert.True(Math.Abs(result[0, 0] - (-1.0)) < 1e-10); // 0 -> -1
        Assert.True(Math.Abs(result[1, 0] - 0.0) < 1e-10);    // 5 -> 0 (midpoint)
        Assert.True(Math.Abs(result[2, 0] - 1.0) < 1e-10);    // 10 -> 1
    }

    [Fact]
    public void MinMaxScaler_InverseTransform_RecoversOriginalData()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 100 }, { 5, 200 }, { 10, 300 } });

        // Act
        var transformed = scaler.FitTransform(data);
        var inversed = scaler.InverseTransform(transformed);

        // Assert
        AssertMatrixEqual(data, inversed, 1e-9);
    }

    [Fact]
    public void MinMaxScaler_CustomRange_InverseTransformWorks()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>(-10.0, 10.0);
        var data = CreateTestMatrix(new double[,] { { 0 }, { 50 }, { 100 } });

        // Act
        var transformed = scaler.FitTransform(data);
        var inversed = scaler.InverseTransform(transformed);

        // Assert
        AssertMatrixEqual(data, inversed, 1e-9);
    }

    [Fact]
    public void MinMaxScaler_ConstantColumn_MapsToMiddleOfRange()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 5, 1 }, { 5, 2 }, { 5, 3 } }); // Column 0 is constant

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Constant column should map to middle of range (0.5)
        Assert.True(Math.Abs(result[0, 0] - 0.5) < 1e-10);
        Assert.True(Math.Abs(result[1, 0] - 0.5) < 1e-10);
        Assert.True(Math.Abs(result[2, 0] - 0.5) < 1e-10);
    }

    [Fact]
    public void MinMaxScaler_InvalidRange_ThrowsArgumentException()
    {
        // Act & Assert - min >= max should throw
        Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(1.0, 0.0)); // max < min
        Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(1.0, 1.0)); // max == min
    }

    [Fact]
    public void MinMaxScaler_SupportsInverseTransform_ReturnsTrue()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();

        // Assert
        Assert.True(scaler.SupportsInverseTransform);
    }

    [Fact]
    public void MinMaxScaler_SpecificColumns_TransformsOnlySpecifiedColumns()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>(columnIndices: new[] { 0 }); // Only column 0
        var data = CreateTestMatrix(new double[,] { { 0, 100 }, { 5, 200 }, { 10, 300 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Column 0 should be scaled, column 1 should be unchanged
        Assert.True(Math.Abs(result[0, 0] - 0.0) < 1e-10);
        Assert.True(Math.Abs(result[2, 0] - 1.0) < 1e-10);
        Assert.Equal(100, result[0, 1]); // Unchanged
        Assert.Equal(200, result[1, 1]); // Unchanged
        Assert.Equal(300, result[2, 1]); // Unchanged
    }

    [Fact]
    public void MinMaxScaler_DataMin_ReturnsCorrectValues()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 5, 20 }, { 10, 30 } });

        // Act
        scaler.Fit(data);

        // Assert
        Assert.NotNull(scaler.DataMin);
        Assert.True(Math.Abs(scaler.DataMin[0] - 1.0) < 1e-10);
        Assert.True(Math.Abs(scaler.DataMin[1] - 10.0) < 1e-10);
    }

    [Fact]
    public void MinMaxScaler_DataMax_ReturnsCorrectValues()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 5, 20 }, { 10, 30 } });

        // Act
        scaler.Fit(data);

        // Assert
        Assert.NotNull(scaler.DataMax);
        Assert.True(Math.Abs(scaler.DataMax[0] - 10.0) < 1e-10);
        Assert.True(Math.Abs(scaler.DataMax[1] - 30.0) < 1e-10);
    }

    [Fact]
    public void MinMaxScaler_NegativeValues_ScalesCorrectly()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix(new double[,] { { -10 }, { 0 }, { 10 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert
        Assert.True(Math.Abs(result[0, 0] - 0.0) < 1e-10);  // -10 -> 0
        Assert.True(Math.Abs(result[1, 0] - 0.5) < 1e-10);  // 0 -> 0.5
        Assert.True(Math.Abs(result[2, 0] - 1.0) < 1e-10);  // 10 -> 1
    }

    #endregion

    #region Cross-Scaler Tests

    [Fact]
    public void Scalers_TransformNewData_UseFittedParameters()
    {
        // Arrange
        var standardScaler = new StandardScaler<double>();
        var minMaxScaler = new MinMaxScaler<double>();

        var trainData = CreateTestMatrix(new double[,] { { 0 }, { 10 } });
        var testData = CreateTestMatrix(new double[,] { { 20 } }); // Outside training range

        // Act
        standardScaler.Fit(trainData);
        minMaxScaler.Fit(trainData);

        var stdResult = standardScaler.Transform(testData);
        var mmResult = minMaxScaler.Transform(testData);

        // Assert - Values should extrapolate beyond training range
        // StandardScaler: (20 - 5) / std = positive value > 1
        Assert.True(stdResult[0, 0] > 1.0);

        // MinMaxScaler: (20 - 0) / (10 - 0) = 2.0
        Assert.True(Math.Abs(mmResult[0, 0] - 2.0) < 1e-10);
    }

    [Fact]
    public void Scalers_EmptyColumnIndices_ProcessesAllColumns()
    {
        // Arrange
        var scaler = new StandardScaler<double>(columnIndices: null);
        var data = CreateTestMatrix(new double[,] { { 1, 10, 100 }, { 2, 20, 200 }, { 3, 30, 300 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - All columns should be standardized
        for (int j = 0; j < result.Columns; j++)
        {
            double mean = (result[0, j] + result[1, j] + result[2, j]) / 3;
            Assert.True(Math.Abs(mean) < 1e-10, $"Column {j} mean should be ~0");
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Scaler_SingleRow_TransformsGracefully()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>(); // StandardScaler has issues with single row (std=0)
        var data = CreateTestMatrix(new double[,] { { 1, 2, 3 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Single row should map to middle of range
        Assert.Equal(1, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void Scaler_LargeValues_TransformsWithoutOverflow()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1e100 }, { 1e100 + 1 }, { 1e100 + 2 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Should not overflow
        for (int i = 0; i < result.Rows; i++)
        {
            Assert.False(double.IsNaN(result[i, 0]));
            Assert.False(double.IsInfinity(result[i, 0]));
        }
    }

    [Fact]
    public void Scaler_SmallValues_TransformsWithoutUnderflow()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1e-100 }, { 2e-100 }, { 3e-100 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Should not underflow to all zeros
        Assert.True(Math.Abs(result[0, 0] - result[2, 0]) > 1e-10, "Values should be different");
    }

    [Fact]
    public void Scaler_TransformWithoutFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => scaler.Transform(data));
    }

    [Fact]
    public void Scaler_InverseTransformWithoutFit_ThrowsInvalidOperationException()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => scaler.InverseTransform(data));
    }

    [Fact]
    public void Scaler_InvalidColumnIndex_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var scaler = new StandardScaler<double>(columnIndices: new[] { 5 }); // Invalid index
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } }); // Only 2 columns

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => scaler.Fit(data));
    }

    [Fact]
    public void Scaler_NegativeColumnIndex_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var scaler = new StandardScaler<double>(columnIndices: new[] { -1 }); // Negative index
        var data = CreateTestMatrix(new double[,] { { 1, 2 }, { 3, 4 } });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => scaler.Fit(data));
    }

    #endregion
}
