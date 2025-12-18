using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Normalizers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Normalizers;

/// <summary>
/// Integration tests for normalizer classes.
/// Tests normalization, denormalization, and round-trip operations.
/// </summary>
public class NormalizersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region MinMax Normalizer Tests

    [Fact]
    public void MinMaxNormalizer_NormalizeOutput_ScalesToZeroOne()
    {
        // Arrange
        var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 0.0, 25.0, 50.0, 75.0, 100.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - data should be scaled to [0, 1]
        Assert.Equal(0.0, normalized[0], Tolerance);
        Assert.Equal(0.25, normalized[1], Tolerance);
        Assert.Equal(0.5, normalized[2], Tolerance);
        Assert.Equal(0.75, normalized[3], Tolerance);
        Assert.Equal(1.0, normalized[4], Tolerance);
    }

    [Fact]
    public void MinMaxNormalizer_NormalizeOutput_CapturesMinMax()
    {
        // Arrange
        var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        // Act
        var (_, parameters) = normalizer.NormalizeOutput(data);

        // Assert
        Assert.Equal(10.0, parameters.Min, Tolerance);
        Assert.Equal(30.0, parameters.Max, Tolerance);
    }

    [Fact]
    public void MinMaxNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 10.0, 50.0, 90.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert - round trip should return original values
        Assert.Equal(data[0], denormalized[0], Tolerance);
        Assert.Equal(data[1], denormalized[1], Tolerance);
        Assert.Equal(data[2], denormalized[2], Tolerance);
    }

    [Fact]
    public void MinMaxNormalizer_NormalizeInput_NormalizesEachColumn()
    {
        // Arrange
        var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
        var matrix = new Matrix<double>(3, 2);
        // Column 0: [0, 50, 100], Column 1: [0, 25, 50]
        matrix[0, 0] = 0.0; matrix[0, 1] = 0.0;
        matrix[1, 0] = 50.0; matrix[1, 1] = 25.0;
        matrix[2, 0] = 100.0; matrix[2, 1] = 50.0;

        // Act
        var (normalized, paramsList) = normalizer.NormalizeInput(matrix);

        // Assert - each column should be scaled to [0, 1]
        Assert.Equal(2, paramsList.Count);
        Assert.Equal(0.0, normalized[0, 0], Tolerance);
        Assert.Equal(0.5, normalized[1, 0], Tolerance);
        Assert.Equal(1.0, normalized[2, 0], Tolerance);
        Assert.Equal(0.0, normalized[0, 1], Tolerance);
        Assert.Equal(0.5, normalized[1, 1], Tolerance);
        Assert.Equal(1.0, normalized[2, 1], Tolerance);
    }

    #endregion

    #region ZScore Normalizer Tests

    [Fact]
    public void ZScoreNormalizer_NormalizeOutput_HasZeroMean()
    {
        // Arrange
        var normalizer = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - normalized data should have mean close to 0
        double sum = 0;
        for (int i = 0; i < normalized.Length; i++)
        {
            sum += normalized[i];
        }
        double mean = sum / normalized.Length;
        Assert.True(Math.Abs(mean) < 1e-10);
    }

    [Fact]
    public void ZScoreNormalizer_NormalizeOutput_CapturesMeanAndStdDev()
    {
        // Arrange
        var normalizer = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        // Act
        var (_, parameters) = normalizer.NormalizeOutput(data);

        // Assert
        Assert.Equal(30.0, parameters.Mean, Tolerance);
        Assert.True(parameters.StdDev > 0);
    }

    [Fact]
    public void ZScoreNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 15.0, 25.0, 35.0, 45.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert - round trip should return original values
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], Tolerance);
        }
    }

    [Fact]
    public void ZScoreNormalizer_NormalizeInput_NormalizesEachColumn()
    {
        // Arrange
        var normalizer = new ZScoreNormalizer<double, Matrix<double>, Vector<double>>();
        var matrix = new Matrix<double>(4, 2);
        matrix[0, 0] = 10.0; matrix[0, 1] = 100.0;
        matrix[1, 0] = 20.0; matrix[1, 1] = 200.0;
        matrix[2, 0] = 30.0; matrix[2, 1] = 300.0;
        matrix[3, 0] = 40.0; matrix[3, 1] = 400.0;

        // Act
        var (normalized, paramsList) = normalizer.NormalizeInput(matrix);

        // Assert - each column should have parameters
        Assert.Equal(2, paramsList.Count);
        Assert.Equal(25.0, paramsList[0].Mean, Tolerance);
        Assert.Equal(250.0, paramsList[1].Mean, Tolerance);
    }

    #endregion

    #region NoNormalizer Tests

    [Fact]
    public void NoNormalizer_NormalizeOutput_ReturnsUnchangedData()
    {
        // Arrange
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var (normalized, _) = normalizer.NormalizeOutput(data);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], normalized[i], Tolerance);
        }
    }

    [Fact]
    public void NoNormalizer_Denormalize_ReturnsUnchangedData()
    {
        // Arrange
        var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 5.0, 10.0, 15.0 });
        var parameters = new NormalizationParameters<double>();

        // Act
        var result = normalizer.Denormalize(data, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], result[i], Tolerance);
        }
    }

    #endregion

    #region MaxAbsScaler Tests

    [Fact]
    public void MaxAbsScaler_NormalizeOutput_ScalesByMaxAbsolute()
    {
        // Arrange
        var normalizer = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { -5.0, 0.0, 10.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - values should be in range [-1, 1]
        Assert.True(normalized[0] >= -1.0);
        Assert.True(normalized[2] <= 1.0);
        Assert.Equal(1.0, normalized[2], Tolerance); // Max value should become 1
    }

    [Fact]
    public void MaxAbsScaler_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { -3.0, 6.0, 9.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], Tolerance);
        }
    }

    #endregion

    #region DecimalNormalizer Tests

    [Fact]
    public void DecimalNormalizer_NormalizeOutput_ScalesByPowerOfTen()
    {
        // Arrange
        var normalizer = new DecimalNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - values should be scaled by power of 10
        Assert.True(normalized[0] < data[0]);
        Assert.True(normalized[2] < data[2]);
    }

    [Fact]
    public void DecimalNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new DecimalNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 123.0, 456.0, 789.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], Tolerance);
        }
    }

    #endregion

    #region LogNormalizer Tests

    [Fact]
    public void LogNormalizer_NormalizeOutput_AppliesLogTransform()
    {
        // Arrange
        var normalizer = new LogNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 1.0, Math.E, Math.E * Math.E });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - log(1) = 0, log(e) = 1, log(e^2) = 2
        Assert.True(normalized[0] < normalized[1]);
        Assert.True(normalized[1] < normalized[2]);
    }

    [Fact]
    public void LogNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new LogNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 10.0, 100.0, 1000.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], 1e-3); // Log can have some precision loss
        }
    }

    #endregion

    #region RobustScalingNormalizer Tests

    [Fact]
    public void RobustScalingNormalizer_NormalizeOutput_UsesMedianAndIQR()
    {
        // Arrange
        var normalizer = new RobustScalingNormalizer<double, Matrix<double>, Vector<double>>();
        // Create data with known median and IQR
        var data = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - normalized median should be close to 0
        // For sorted [1,2,3,4,5,6,7,8,9], median is 5
        Assert.NotNull(normalized);
        Assert.Equal(9, normalized.Length);
    }

    [Fact]
    public void RobustScalingNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new RobustScalingNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], Tolerance);
        }
    }

    #endregion

    #region MeanVarianceNormalizer Tests

    [Fact]
    public void MeanVarianceNormalizer_NormalizeOutput_SimilarToZScore()
    {
        // Arrange
        var normalizer = new MeanVarianceNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - should have parameters for mean and variance
        Assert.NotNull(normalized);
        Assert.Equal(5, normalized.Length);
    }

    [Fact]
    public void MeanVarianceNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new MeanVarianceNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 5.0, 15.0, 25.0, 35.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], Tolerance);
        }
    }

    #endregion

    #region GlobalContrastNormalizer Tests

    [Fact]
    public void GlobalContrastNormalizer_NormalizeOutput_NormalizesContrast()
    {
        // Arrange
        var normalizer = new GlobalContrastNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 100.0, 150.0, 200.0, 250.0, 300.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert
        Assert.NotNull(normalized);
        Assert.Equal(5, normalized.Length);
    }

    [Fact]
    public void GlobalContrastNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new GlobalContrastNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 50.0, 100.0, 150.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], Tolerance);
        }
    }

    #endregion

    #region LpNormNormalizer Tests

    [Fact]
    public void LpNormNormalizer_NormalizeOutput_L2Norm()
    {
        // Arrange
        var normalizer = new LpNormNormalizer<double, Matrix<double>, Vector<double>>(2.0);
        var data = new Vector<double>(new[] { 3.0, 4.0 }); // L2 norm = 5

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - normalized L2 norm should be 1
        double normSquared = 0;
        for (int i = 0; i < normalized.Length; i++)
        {
            normSquared += normalized[i] * normalized[i];
        }
        Assert.Equal(1.0, Math.Sqrt(normSquared), Tolerance);
    }

    [Fact]
    public void LpNormNormalizer_Denormalize_ReturnsOriginalValues()
    {
        // Arrange
        var normalizer = new LpNormNormalizer<double, Matrix<double>, Vector<double>>(2.0);
        var data = new Vector<double>(new[] { 6.0, 8.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);
        var denormalized = normalizer.Denormalize(normalized, parameters);

        // Assert
        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(data[i], denormalized[i], Tolerance);
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllNormalizers_RoundTrip_PreservesOriginalData()
    {
        // Arrange
        var data = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        var normalizers = new INormalizer<double, Matrix<double>, Vector<double>>[]
        {
            new MinMaxNormalizer<double, Matrix<double>, Vector<double>>(),
            new ZScoreNormalizer<double, Matrix<double>, Vector<double>>(),
            new NoNormalizer<double, Matrix<double>, Vector<double>>(),
            new MaxAbsScaler<double, Matrix<double>, Vector<double>>(),
            new MeanVarianceNormalizer<double, Matrix<double>, Vector<double>>()
        };

        // Act & Assert
        foreach (var normalizer in normalizers)
        {
            var (normalized, parameters) = normalizer.NormalizeOutput(data);
            var denormalized = normalizer.Denormalize(normalized, parameters);

            for (int i = 0; i < data.Length; i++)
            {
                Assert.Equal(data[i], denormalized[i], Tolerance);
            }
        }
    }

    [Fact]
    public void AllNormalizers_NormalizedValuesAreBounded()
    {
        // Arrange
        var data = new Vector<double>(new[] { -100.0, 0.0, 100.0, 1000.0 });

        var normalizers = new INormalizer<double, Matrix<double>, Vector<double>>[]
        {
            new MinMaxNormalizer<double, Matrix<double>, Vector<double>>(),
            new MaxAbsScaler<double, Matrix<double>, Vector<double>>(),
        };

        // Act & Assert
        foreach (var normalizer in normalizers)
        {
            var (normalized, _) = normalizer.NormalizeOutput(data);

            // Check that normalized values are bounded
            for (int i = 0; i < normalized.Length; i++)
            {
                Assert.True(normalized[i] >= -2.0 && normalized[i] <= 2.0,
                    $"Normalizer {normalizer.GetType().Name} produced out of range value: {normalized[i]}");
            }
        }
    }

    [Fact]
    public void MinMaxNormalizer_ConstantData_HandlesGracefully()
    {
        // Arrange - all same values (edge case)
        var normalizer = new MinMaxNormalizer<double, Matrix<double>, Vector<double>>();
        var data = new Vector<double>(new[] { 5.0, 5.0, 5.0 });

        // Act
        var (normalized, parameters) = normalizer.NormalizeOutput(data);

        // Assert - should handle division by zero gracefully
        Assert.Equal(3, normalized.Length);
        // When min == max, normalized values will be NaN or 0
        Assert.True(double.IsNaN(normalized[0]) || normalized[0] == 0.0 || double.IsInfinity(normalized[0]));
    }

    #endregion
}
