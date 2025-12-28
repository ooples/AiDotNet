using Xunit;
using AiDotNet.Helpers;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for RegressionHelper to verify regression utility operations.
/// </summary>
public class RegressionHelperIntegrationTests
{
    #region CenterAndScale Tests - Basic Functionality

    [Fact]
    public void CenterAndScale_SimpleData_ReturnsCorrectMeans()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1, 10 },
            { 2, 20 },
            { 3, 30 }
        });
        var y = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // xMean should be [2, 20] (mean of each column)
        Assert.Equal(2.0, xMean[0]);
        Assert.Equal(20.0, xMean[1]);
    }

    [Fact]
    public void CenterAndScale_SimpleData_ReturnsCorrectStandardDeviations()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1, 10 },
            { 2, 20 },
            { 3, 30 }
        });
        var y = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Standard deviation of [1,2,3] is 1.0 and [10,20,30] is 10.0
        Assert.Equal(1.0, xStd[0], 5);
        Assert.Equal(10.0, xStd[1], 5);
        Assert.Equal(100.0, yStd, 5);
    }

    [Fact]
    public void CenterAndScale_SimpleData_XScaledHasZeroMean()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1, 10 },
            { 2, 20 },
            { 3, 30 }
        });
        var y = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Each column of xScaled should have mean ≈ 0
        double col0Mean = (xScaled[0, 0] + xScaled[1, 0] + xScaled[2, 0]) / 3.0;
        double col1Mean = (xScaled[0, 1] + xScaled[1, 1] + xScaled[2, 1]) / 3.0;

        Assert.True(Math.Abs(col0Mean) < 1e-10, $"Column 0 mean should be 0, got {col0Mean}");
        Assert.True(Math.Abs(col1Mean) < 1e-10, $"Column 1 mean should be 0, got {col1Mean}");
    }

    [Fact]
    public void CenterAndScale_SimpleData_YScaledHasZeroMean()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1, 10 },
            { 2, 20 },
            { 3, 30 }
        });
        var y = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        double yScaledMean = yScaled.Sum() / yScaled.Length;
        Assert.True(Math.Abs(yScaledMean) < 1e-10, $"yScaled mean should be 0, got {yScaledMean}");
    }

    [Fact]
    public void CenterAndScale_SimpleData_XScaledHasUnitVariance()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1, 10 },
            { 2, 20 },
            { 3, 30 }
        });
        var y = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Each column of xScaled should have std ≈ 1
        var col0 = new Vector<double>(new[] { xScaled[0, 0], xScaled[1, 0], xScaled[2, 0] });
        var col1 = new Vector<double>(new[] { xScaled[0, 1], xScaled[1, 1], xScaled[2, 1] });

        double std0 = StatisticsHelper<double>.CalculateStandardDeviation(col0);
        double std1 = StatisticsHelper<double>.CalculateStandardDeviation(col1);

        Assert.Equal(1.0, std0, 5);
        Assert.Equal(1.0, std1, 5);
    }

    #endregion

    #region CenterAndScale Tests - Shape Preservation

    [Fact]
    public void CenterAndScale_PreservesMatrixShape()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        });
        var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        Assert.Equal(x.Rows, xScaled.Rows);
        Assert.Equal(x.Columns, xScaled.Columns);
        Assert.Equal(y.Length, yScaled.Length);
        Assert.Equal(x.Columns, xMean.Length);
        Assert.Equal(x.Columns, xStd.Length);
    }

    [Fact]
    public void CenterAndScale_SingleRow_ReturnsNaN()
    {
        // With a single row, standard deviation is 0, causing division by zero
        var x = new Matrix<double>(new double[,]
        {
            { 5, 10, 15 }
        });
        var y = new Vector<double>(new[] { 100.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // With single row, mean equals the values
        Assert.Equal(5.0, xMean[0]);
        Assert.Equal(10.0, xMean[1]);
        Assert.Equal(15.0, xMean[2]);

        // Single row std is 0, so scaling results in NaN (0/0)
        Assert.True(double.IsNaN(xScaled[0, 0]));
        Assert.True(double.IsNaN(xScaled[0, 1]));
        Assert.True(double.IsNaN(xScaled[0, 2]));
    }

    [Fact]
    public void CenterAndScale_SingleColumn_Works()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1 },
            { 2 },
            { 3 }
        });
        var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        Assert.Equal(1, xScaled.Columns);
        Assert.Equal(3, xScaled.Rows);
        Assert.Equal(2.0, xMean[0]);
    }

    #endregion

    #region CenterAndScale Tests - Float Type

    [Fact]
    public void CenterAndScale_Float_ReturnsCorrectValues()
    {
        var x = new Matrix<float>(new float[,]
        {
            { 1f, 10f },
            { 2f, 20f },
            { 3f, 30f }
        });
        var y = new Vector<float>(new[] { 100f, 200f, 300f });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<float>.CenterAndScale(x, y);

        Assert.Equal(2f, xMean[0], 4);
        Assert.Equal(20f, xMean[1], 4);
        Assert.Equal(1f, xStd[0], 4);
        Assert.Equal(10f, xStd[1], 4);
    }

    #endregion

    #region CenterAndScale Tests - Different Data Distributions

    [Fact]
    public void CenterAndScale_NegativeValues_HandlesCorrectly()
    {
        var x = new Matrix<double>(new double[,]
        {
            { -10, -100 },
            { 0, 0 },
            { 10, 100 }
        });
        var y = new Vector<double>(new[] { -50.0, 0.0, 50.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Mean of symmetric distribution should be 0
        Assert.Equal(0.0, xMean[0], 5);
        Assert.Equal(0.0, xMean[1], 5);

        // Verify scaling worked
        Assert.Equal(3, xScaled.Rows);
        Assert.Equal(2, xScaled.Columns);
    }

    [Fact]
    public void CenterAndScale_LargeValues_HandlesCorrectly()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1e6, 1e9 },
            { 2e6, 2e9 },
            { 3e6, 3e9 }
        });
        var y = new Vector<double>(new[] { 1e12, 2e12, 3e12 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Should still center correctly
        Assert.Equal(2e6, xMean[0], 0);
        Assert.Equal(2e9, xMean[1], 0);

        // Verify shape preserved
        Assert.Equal(3, xScaled.Rows);
        Assert.Equal(2, xScaled.Columns);
    }

    [Fact]
    public void CenterAndScale_SmallValues_HandlesCorrectly()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 1e-6, 1e-9 },
            { 2e-6, 2e-9 },
            { 3e-6, 3e-9 }
        });
        var y = new Vector<double>(new[] { 1e-12, 2e-12, 3e-12 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Should still center correctly with small values
        Assert.Equal(2e-6, xMean[0], 10);
        Assert.Equal(2e-9, xMean[1], 15);

        // Verify shape preserved
        Assert.Equal(3, xScaled.Rows);
        Assert.Equal(2, xScaled.Columns);
    }

    [Fact]
    public void CenterAndScale_MixedScaleFeatures_NormalizesCorrectly()
    {
        // Features with vastly different scales
        var x = new Matrix<double>(new double[,]
        {
            { 0.001, 1000 },
            { 0.002, 2000 },
            { 0.003, 3000 }
        });
        var y = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // After scaling, both columns should have similar ranges
        var col0 = new Vector<double>(new[] { xScaled[0, 0], xScaled[1, 0], xScaled[2, 0] });
        var col1 = new Vector<double>(new[] { xScaled[0, 1], xScaled[1, 1], xScaled[2, 1] });

        double std0 = StatisticsHelper<double>.CalculateStandardDeviation(col0);
        double std1 = StatisticsHelper<double>.CalculateStandardDeviation(col1);

        // Both should be approximately 1
        Assert.True(Math.Abs(std0 - 1.0) < 0.01, $"Std0 should be ~1, got {std0}");
        Assert.True(Math.Abs(std1 - 1.0) < 0.01, $"Std1 should be ~1, got {std1}");
    }

    #endregion

    #region CenterAndScale Tests - Known Values

    [Fact]
    public void CenterAndScale_KnownValues_VerifyCalculation()
    {
        // Use simple values for easy manual verification
        var x = new Matrix<double>(new double[,]
        {
            { 0, 0 },
            { 2, 4 }
        });
        var y = new Vector<double>(new[] { 0.0, 2.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Mean of [0, 2] is 1, mean of [0, 4] is 2
        Assert.Equal(1.0, xMean[0]);
        Assert.Equal(2.0, xMean[1]);

        // Sample std of [0, 2] = sqrt(((0-1)^2 + (2-1)^2) / (2-1)) = sqrt(2) ≈ 1.4142
        double expectedStd = Math.Sqrt(2);
        Assert.Equal(expectedStd, xStd[0], 5);

        // Verify the scaled values
        // (0 - 1) / sqrt(2) ≈ -0.7071, (2 - 1) / sqrt(2) ≈ 0.7071
        double expectedScaled = 1.0 / Math.Sqrt(2);
        Assert.Equal(-expectedScaled, xScaled[0, 0], 5);
        Assert.Equal(expectedScaled, xScaled[1, 0], 5);
    }

    #endregion

    #region CenterAndScale Tests - Large Dataset

    [Fact]
    public void CenterAndScale_LargeDataset_PerformsCorrectly()
    {
        int rows = 1000;
        int cols = 50;
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);

        // Fill with data
        var random = new Random(42);
        for (int i = 0; i < rows; i++)
        {
            y[i] = random.NextDouble() * 100;
            for (int j = 0; j < cols; j++)
            {
                // Each column has different mean and scale
                x[i, j] = j * 100 + random.NextDouble() * (j + 1);
            }
        }

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Verify shapes
        Assert.Equal(rows, xScaled.Rows);
        Assert.Equal(cols, xScaled.Columns);
        Assert.Equal(cols, xMean.Length);
        Assert.Equal(cols, xStd.Length);

        // Verify each column is approximately centered
        for (int j = 0; j < cols; j++)
        {
            double colSum = 0;
            for (int i = 0; i < rows; i++)
            {
                colSum += xScaled[i, j];
            }
            double colMean = colSum / rows;
            Assert.True(Math.Abs(colMean) < 0.01, $"Column {j} mean should be ~0, got {colMean}");
        }
    }

    #endregion

    #region CenterAndScale Tests - Reversibility

    [Fact]
    public void CenterAndScale_CanBeReversed()
    {
        var x = new Matrix<double>(new double[,]
        {
            { 5, 50 },
            { 10, 100 },
            { 15, 150 }
        });
        var y = new Vector<double>(new[] { 25.0, 50.0, 75.0 });

        var (xScaled, yScaled, xMean, xStd, yStd) = RegressionHelper<double>.CenterAndScale(x, y);

        // Reverse the scaling: original = scaled * std + mean
        var xRecovered = new Matrix<double>(x.Rows, x.Columns);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                xRecovered[i, j] = xScaled[i, j] * xStd[j] + xMean[j];
            }
        }

        // Verify recovery
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                Assert.Equal(x[i, j], xRecovered[i, j], 5);
            }
        }
    }

    #endregion
}
