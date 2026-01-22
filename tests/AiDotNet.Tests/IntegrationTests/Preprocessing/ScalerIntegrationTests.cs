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
        // Original mean = 10, sample std = sqrt((100+0+100)/2) = sqrt(100) = 10
        // Values should be: 0/std, 10/std, 20/std (no mean subtraction)
        Assert.True(Math.Abs(result[0, 0]) < 1e-10); // 0 / 10 = 0
        Assert.True(Math.Abs(result[1, 0] - 1.0) < 1e-10); // 10 / 10 = 1
        Assert.True(Math.Abs(result[2, 0] - 2.0) < 1e-10); // 20 / 10 = 2
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

    #region RobustScaler Tests

    [Fact]
    public void RobustScaler_DefaultParameters_CentersAndScalesData()
    {
        // Arrange
        var scaler = new RobustScaler<double>();
        // Data with some outliers: [1, 2, 3, 4, 100]
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 100 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Median should be 3, IQR should be Q3-Q1
        // The median value (3) should be centered to 0
        Assert.Equal(5, result.Rows);
        Assert.Equal(1, result.Columns);

        // Find the index of the original median value (3)
        // After centering by median, 3 should become 0
        Assert.True(Math.Abs(result[2, 0]) < 0.5, "Median value should be centered close to 0");
    }

    [Fact]
    public void RobustScaler_WithCenteringOnly_CentersButDoesNotScale()
    {
        // Arrange
        var scaler = new RobustScaler<double>(withCentering: true, withScaling: false);
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Median is 3, so values become [-2, -1, 0, 1, 2]
        Assert.True(Math.Abs(result[0, 0] - (-2)) < Tolerance);
        Assert.True(Math.Abs(result[2, 0] - 0) < Tolerance);
        Assert.True(Math.Abs(result[4, 0] - 2) < Tolerance);
    }

    [Fact]
    public void RobustScaler_InverseTransform_RecoversOriginalData()
    {
        // Arrange
        var scaler = new RobustScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1, 10 }, { 2, 20 }, { 3, 30 }, { 4, 40 }, { 5, 50 } });

        // Act
        var transformed = scaler.FitTransform(data);
        var inversed = scaler.InverseTransform(transformed);

        // Assert
        AssertMatrixEqual(data, inversed, 1e-9);
    }

    [Fact]
    public void RobustScaler_MedianProperty_ReturnsCorrectValues()
    {
        // Arrange
        var scaler = new RobustScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });

        // Act
        scaler.Fit(data);

        // Assert - Median of [1,2,3,4,5] is 3
        Assert.NotNull(scaler.Median);
        Assert.True(Math.Abs(scaler.Median![0] - 3.0) < Tolerance);
    }

    [Fact]
    public void RobustScaler_SupportsInverseTransform_ReturnsTrue()
    {
        // Arrange
        var scaler = new RobustScaler<double>();

        // Assert
        Assert.True(scaler.SupportsInverseTransform);
    }

    #endregion

    #region MaxAbsScaler Tests

    [Fact]
    public void MaxAbsScaler_DefaultParameters_ScalesToNegativeOneToOne()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = CreateTestMatrix(new double[,] { { -5 }, { 0 }, { 10 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Max abs is 10, so values become [-0.5, 0, 1]
        Assert.True(Math.Abs(result[0, 0] - (-0.5)) < Tolerance);
        Assert.True(Math.Abs(result[1, 0] - 0) < Tolerance);
        Assert.True(Math.Abs(result[2, 0] - 1.0) < Tolerance);
    }

    [Fact]
    public void MaxAbsScaler_PreservesSparsity()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 0, 0 }, { 5, 0 }, { 0, 10 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Zero values should remain zero
        Assert.True(Math.Abs(result[0, 0]) < Tolerance);
        Assert.True(Math.Abs(result[0, 1]) < Tolerance);
        Assert.True(Math.Abs(result[1, 1]) < Tolerance);
        Assert.True(Math.Abs(result[2, 0]) < Tolerance);
    }

    [Fact]
    public void MaxAbsScaler_InverseTransform_RecoversOriginalData()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = CreateTestMatrix(new double[,] { { -5, 10 }, { 0, 20 }, { 10, -30 } });

        // Act
        var transformed = scaler.FitTransform(data);
        var inversed = scaler.InverseTransform(transformed);

        // Assert
        AssertMatrixEqual(data, inversed, 1e-9);
    }

    [Fact]
    public void MaxAbsScaler_MaxAbsoluteProperty_ReturnsCorrectValues()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();
        var data = CreateTestMatrix(new double[,] { { -5, 10 }, { 3, -20 } });

        // Act
        scaler.Fit(data);

        // Assert - Max abs of column 0 is 5, column 1 is 20
        Assert.NotNull(scaler.MaxAbsolute);
        Assert.True(Math.Abs(scaler.MaxAbsolute![0] - 5.0) < Tolerance);
        Assert.True(Math.Abs(scaler.MaxAbsolute![1] - 20.0) < Tolerance);
    }

    [Fact]
    public void MaxAbsScaler_SupportsInverseTransform_ReturnsTrue()
    {
        // Arrange
        var scaler = new MaxAbsScaler<double>();

        // Assert
        Assert.True(scaler.SupportsInverseTransform);
    }

    #endregion

    #region Normalizer Tests

    [Fact]
    public void Normalizer_L2Norm_NormalizesRowsToUnitLength()
    {
        // Arrange
        var normalizer = new Normalizer<double>(NormType.L2);
        // Row [3, 4] has L2 norm = 5, should become [0.6, 0.8]
        var data = CreateTestMatrix(new double[,] { { 3, 4 }, { 6, 8 } });

        // Act
        var result = normalizer.FitTransform(data);

        // Assert - Each row should have L2 norm = 1
        Assert.True(Math.Abs(result[0, 0] - 0.6) < Tolerance);
        Assert.True(Math.Abs(result[0, 1] - 0.8) < Tolerance);
        Assert.True(Math.Abs(result[1, 0] - 0.6) < Tolerance);
        Assert.True(Math.Abs(result[1, 1] - 0.8) < Tolerance);
    }

    [Fact]
    public void Normalizer_L1Norm_NormalizesRowsToSumOne()
    {
        // Arrange
        var normalizer = new Normalizer<double>(NormType.L1);
        // Row [2, 3] has L1 norm = 5, should become [0.4, 0.6]
        var data = CreateTestMatrix(new double[,] { { 2, 3 } });

        // Act
        var result = normalizer.FitTransform(data);

        // Assert - Sum of absolute values should be 1
        Assert.True(Math.Abs(result[0, 0] - 0.4) < Tolerance);
        Assert.True(Math.Abs(result[0, 1] - 0.6) < Tolerance);
    }

    [Fact]
    public void Normalizer_MaxNorm_NormalizesRowsByMaxValue()
    {
        // Arrange
        var normalizer = new Normalizer<double>(NormType.Max);
        // Row [2, 4] has max abs = 4, should become [0.5, 1]
        var data = CreateTestMatrix(new double[,] { { 2, 4 } });

        // Act
        var result = normalizer.FitTransform(data);

        // Assert - Max absolute value should be 1
        Assert.True(Math.Abs(result[0, 0] - 0.5) < Tolerance);
        Assert.True(Math.Abs(result[0, 1] - 1.0) < Tolerance);
    }

    [Fact]
    public void Normalizer_SupportsInverseTransform_ReturnsFalse()
    {
        // Arrange
        var normalizer = new Normalizer<double>();

        // Assert - Normalizer doesn't support inverse transform
        Assert.False(normalizer.SupportsInverseTransform);
    }

    [Fact]
    public void Normalizer_InverseTransform_ThrowsNotSupportedException()
    {
        // Arrange
        var normalizer = new Normalizer<double>();
        var data = CreateTestMatrix(new double[,] { { 3, 4 } });
        var transformed = normalizer.FitTransform(data);

        // Act & Assert
        Assert.Throws<NotSupportedException>(() => normalizer.InverseTransform(transformed));
    }

    #endregion

    #region Additional Edge Cases

    [Fact]
    public void Scaler_SingleFeature_TransformsCorrectly()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 } });

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Should handle single feature without issues
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);

        // Mean should be centered to 0
        double mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3;
        Assert.True(Math.Abs(mean) < Tolerance);
    }

    [Fact]
    public void Scaler_SingleSample_HandlesGracefully()
    {
        // Arrange
        var scaler = new MinMaxScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 5, 10 } }); // Single sample

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Single sample should not cause errors
        Assert.Equal(1, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void Scaler_AllSameValues_HandlesConstantFeature()
    {
        // Arrange
        var scaler = new StandardScaler<double>();
        var data = CreateTestMatrix(new double[,] { { 5, 1 }, { 5, 2 }, { 5, 3 } }); // First column is constant

        // Act
        var result = scaler.FitTransform(data);

        // Assert - Constant feature should not cause NaN or infinity
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
        for (int i = 0; i < 3; i++)
        {
            Assert.False(double.IsNaN(result[i, 0]), $"NaN at [{i},0]");
            Assert.False(double.IsInfinity(result[i, 0]), $"Infinity at [{i},0]");
        }
    }

    [Fact]
    public void Scaler_NullData_ThrowsArgumentNullException()
    {
        // Arrange
        var scaler = new StandardScaler<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => scaler.Fit(null!));
    }

    [Fact]
    public void RobustScaler_WithOutliers_IsRobust()
    {
        // Arrange
        var robustScaler = new RobustScaler<double>();
        var standardScaler = new StandardScaler<double>();

        // Data with extreme outlier
        var data = CreateTestMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 1000 } });

        // Act
        var robustResult = robustScaler.FitTransform(data);
        var standardResult = standardScaler.FitTransform(data);

        // Assert - RobustScaler should be less affected by outlier
        // The first four values should be closer together in robust scaling
        double robustRange = Math.Abs(robustResult[3, 0] - robustResult[0, 0]);
        double standardRange = Math.Abs(standardResult[3, 0] - standardResult[0, 0]);

        // In robust scaling, normal values should have a larger relative spread
        Assert.True(robustRange > standardRange, "RobustScaler should preserve relative distances of non-outlier data");
    }

    #endregion
}
