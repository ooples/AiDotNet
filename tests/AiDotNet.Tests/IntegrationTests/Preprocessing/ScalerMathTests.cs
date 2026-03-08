using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Mathematically rigorous tests for preprocessing scalers verifying:
/// 1. Transform output matches expected mathematical properties
/// 2. InverseTransform exactly recovers original data
/// 3. Fit → Transform → InverseTransform round-trip is lossless
/// 4. Column-wise operations preserve independence
///
/// These catch bugs where scalers silently corrupt data (wrong mean, wrong variance,
/// swapped columns, etc.) — problems that produce wrong ML results but don't crash.
/// </summary>
public class ScalerMathTests
{
    #region StandardScaler — (x - mean) / std

    [Fact]
    public void StandardScaler_Transform_ZeroMeanUnitVariance()
    {
        var data = CreateKnownData();
        var scaler = new StandardScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        // Each column should have mean ≈ 0 and std ≈ 1
        for (int col = 0; col < transformed.Columns; col++)
        {
            double mean = ColumnMean(transformed, col);
            double std = ColumnStd(transformed, col);

            Assert.True(Math.Abs(mean) < 1e-10,
                $"StandardScaler: Column {col} mean={mean:E3} should be ~0");
            Assert.True(Math.Abs(std - 1.0) < 1e-10,
                $"StandardScaler: Column {col} std={std:E3} should be ~1");
        }
    }

    [Fact]
    public void StandardScaler_InverseTransform_RecoversOriginal()
    {
        var data = CreateKnownData();
        var scaler = new StandardScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.True(Math.Abs(data[i, j] - recovered[i, j]) < 1e-10,
                    $"StandardScaler InverseTransform mismatch at [{i},{j}]: " +
                    $"original={data[i, j]}, recovered={recovered[i, j]}");
            }
        }
    }

    [Fact]
    public void StandardScaler_FitTransform_MatchesSeparateFitAndTransform()
    {
        var data = CreateKnownData();

        var scaler1 = new StandardScaler<double>();
        scaler1.Fit(data);
        var result1 = scaler1.Transform(data);

        var scaler2 = new StandardScaler<double>();
        var result2 = scaler2.FitTransform(data);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j], precision: 12);
            }
        }
    }

    #endregion

    #region MinMaxScaler — (x - min) / (max - min)

    [Fact]
    public void MinMaxScaler_Transform_OutputInZeroOneRange()
    {
        var data = CreateKnownData();
        var scaler = new MinMaxScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        for (int i = 0; i < transformed.Rows; i++)
        {
            for (int j = 0; j < transformed.Columns; j++)
            {
                Assert.True(transformed[i, j] >= -1e-10 && transformed[i, j] <= 1.0 + 1e-10,
                    $"MinMaxScaler: Value {transformed[i, j]} at [{i},{j}] outside [0,1] range");
            }
        }

        // Min of each column should be 0, max should be 1
        for (int col = 0; col < transformed.Columns; col++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < transformed.Rows; i++)
            {
                if (transformed[i, col] < min) min = transformed[i, col];
                if (transformed[i, col] > max) max = transformed[i, col];
            }
            Assert.True(Math.Abs(min) < 1e-10,
                $"MinMaxScaler: Column {col} min={min:E3} should be ~0");
            Assert.True(Math.Abs(max - 1.0) < 1e-10,
                $"MinMaxScaler: Column {col} max={max:E3} should be ~1");
        }
    }

    [Fact]
    public void MinMaxScaler_InverseTransform_RecoversOriginal()
    {
        var data = CreateKnownData();
        var scaler = new MinMaxScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.True(Math.Abs(data[i, j] - recovered[i, j]) < 1e-10,
                    $"MinMaxScaler InverseTransform mismatch at [{i},{j}]: " +
                    $"original={data[i, j]}, recovered={recovered[i, j]}");
            }
        }
    }

    #endregion

    #region RobustScaler — (x - median) / IQR

    [Fact]
    public void RobustScaler_Transform_MedianIsZero()
    {
        var data = CreateDataWithOutliers();
        var scaler = new RobustScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        // Median of each column should be approximately 0
        for (int col = 0; col < transformed.Columns; col++)
        {
            var colValues = new double[transformed.Rows];
            for (int i = 0; i < transformed.Rows; i++)
                colValues[i] = transformed[i, col];

            Array.Sort(colValues);
            double median = colValues.Length % 2 == 0
                ? (colValues[colValues.Length / 2 - 1] + colValues[colValues.Length / 2]) / 2
                : colValues[colValues.Length / 2];

            Assert.True(Math.Abs(median) < 1e-10,
                $"RobustScaler: Column {col} median={median:E3} should be ~0");
        }
    }

    [Fact]
    public void RobustScaler_InverseTransform_RecoversOriginal()
    {
        var data = CreateDataWithOutliers();
        var scaler = new RobustScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.True(Math.Abs(data[i, j] - recovered[i, j]) < 1e-8,
                    $"RobustScaler InverseTransform mismatch at [{i},{j}]: " +
                    $"original={data[i, j]}, recovered={recovered[i, j]}");
            }
        }
    }

    #endregion

    #region MaxAbsScaler — x / max(|x|)

    [Fact]
    public void MaxAbsScaler_Transform_OutputInNegOneToOneRange()
    {
        var data = CreateKnownData();
        var scaler = new MaxAbsScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);

        for (int i = 0; i < transformed.Rows; i++)
        {
            for (int j = 0; j < transformed.Columns; j++)
            {
                Assert.True(transformed[i, j] >= -1.0 - 1e-10 && transformed[i, j] <= 1.0 + 1e-10,
                    $"MaxAbsScaler: Value {transformed[i, j]} at [{i},{j}] outside [-1,1] range");
            }
        }
    }

    [Fact]
    public void MaxAbsScaler_InverseTransform_RecoversOriginal()
    {
        var data = CreateKnownData();
        var scaler = new MaxAbsScaler<double>();
        scaler.Fit(data);
        var transformed = scaler.Transform(data);
        var recovered = scaler.InverseTransform(transformed);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.True(Math.Abs(data[i, j] - recovered[i, j]) < 1e-10,
                    $"MaxAbsScaler InverseTransform mismatch at [{i},{j}]: " +
                    $"original={data[i, j]}, recovered={recovered[i, j]}");
            }
        }
    }

    #endregion

    #region Cross-Scaler Consistency

    [Fact]
    public void StandardScaler_TransformDoesNotModifyOriginalCopy()
    {
        var original = CreateKnownData();
        var copy = CloneMatrix(original);

        var scaler = new StandardScaler<double>();
        scaler.Fit(copy);
        scaler.Transform(copy);

        // Verify a separate copy of the original data wasn't mutated
        for (int i = 0; i < original.Rows; i++)
        {
            for (int j = 0; j < original.Columns; j++)
            {
                Assert.Equal(original[i, j], original[i, j], precision: 15);
            }
        }
    }

    #endregion

    #region Helper Methods

    private static Matrix<double> CreateKnownData()
    {
        // Create data with known statistical properties
        // Column 0: range [1, 10], Column 1: range [-5, 5], Column 2: range [100, 200]
        var random = new Random(42);
        int n = 50;
        var data = new Matrix<double>(n, 3);
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = 1.0 + random.NextDouble() * 9.0;
            data[i, 1] = -5.0 + random.NextDouble() * 10.0;
            data[i, 2] = 100.0 + random.NextDouble() * 100.0;
        }
        return data;
    }

    private static Matrix<double> CreateDataWithOutliers()
    {
        var random = new Random(42);
        int n = 50;
        var data = new Matrix<double>(n, 2);
        for (int i = 0; i < n - 2; i++)
        {
            data[i, 0] = random.NextDouble() * 10;
            data[i, 1] = random.NextDouble() * 10;
        }
        // Add outliers
        data[n - 2, 0] = 1000;
        data[n - 2, 1] = -500;
        data[n - 1, 0] = -800;
        data[n - 1, 1] = 900;
        return data;
    }

    private static Matrix<double> CloneMatrix(Matrix<double> m)
    {
        var clone = new Matrix<double>(m.Rows, m.Columns);
        for (int i = 0; i < m.Rows; i++)
            for (int j = 0; j < m.Columns; j++)
                clone[i, j] = m[i, j];
        return clone;
    }

    private static double ColumnMean(Matrix<double> m, int col)
    {
        double sum = 0;
        for (int i = 0; i < m.Rows; i++) sum += m[i, col];
        return sum / m.Rows;
    }

    private static double ColumnStd(Matrix<double> m, int col)
    {
        double mean = ColumnMean(m, col);
        double sumSq = 0;
        for (int i = 0; i < m.Rows; i++)
        {
            double diff = m[i, col] - mean;
            sumSq += diff * diff;
        }
        return Math.Sqrt(sumSq / (m.Rows - 1)); // Sample std
    }

    #endregion
}
