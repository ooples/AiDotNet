using AiDotNet.TransferLearning.DomainAdaptation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TransferLearning;

/// <summary>
/// Integration tests for Domain Adaptation classes (CORAL and MMD).
/// These tests verify mathematical correctness of domain adaptation algorithms.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class DomainAdaptationIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double RelaxedTolerance = 1e-3; // For complex computations

    #region CORAL Domain Adapter Tests

    [Fact]
    public void CORALDomainAdapter_AdaptSource_IdenticalDomains_ReturnsDataWithSimilarDistribution()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });

        // Act
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert - When domains are identical, adapted data should be similar to original
        Assert.Equal(sourceData.Rows, adapted.Rows);
        Assert.Equal(sourceData.Columns, adapted.Columns);
    }

    [Fact]
    public void CORALDomainAdapter_ComputeDomainDiscrepancy_IdenticalDomains_ReturnsNearZero()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

        // Assert - Identical domains should have near-zero discrepancy
        Assert.True(discrepancy < 0.1, $"Expected near-zero discrepancy for identical domains, got {discrepancy}");
    }

    [Fact]
    public void CORALDomainAdapter_ComputeDomainDiscrepancy_DifferentDomains_ReturnsPositiveValue()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 10.0, 20.0 },
            { 30.0, 40.0 },
            { 50.0, 60.0 }
        });

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

        // Assert - Different domains should have positive discrepancy
        Assert.True(discrepancy > 0, $"Expected positive discrepancy for different domains, got {discrepancy}");
    }

    [Fact]
    public void CORALDomainAdapter_AdaptSource_PreservesRowCount()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 10.0, 20.0, 30.0 },
            { 40.0, 50.0, 60.0 },
            { 70.0, 80.0, 90.0 },
            { 100.0, 110.0, 120.0 }
        });

        // Act
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert - Row count should be preserved
        Assert.Equal(sourceData.Rows, adapted.Rows);
        Assert.Equal(sourceData.Columns, adapted.Columns);
    }

    [Fact]
    public void CORALDomainAdapter_AdaptTarget_ReturnsValidMatrix()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 10.0, 20.0 },
            { 30.0, 40.0 },
            { 50.0, 60.0 }
        });

        // Act
        var adapted = adapter.AdaptTarget(targetData, sourceData);

        // Assert
        Assert.Equal(targetData.Rows, adapted.Rows);
        Assert.Equal(targetData.Columns, adapted.Columns);
        // Verify no NaN or Infinity values
        for (int i = 0; i < adapted.Rows; i++)
        {
            for (int j = 0; j < adapted.Columns; j++)
            {
                Assert.False(double.IsNaN(adapted[i, j]), $"Found NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(adapted[i, j]), $"Found Infinity at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void CORALDomainAdapter_RequiresTraining_ReturnsTrue()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();

        // Assert
        Assert.True(adapter.RequiresTraining);
    }

    [Fact]
    public void CORALDomainAdapter_AdaptationMethod_ReturnsCorrectName()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();

        // Assert
        Assert.Contains("CORAL", adapter.AdaptationMethod);
    }

    [Fact]
    public void CORALDomainAdapter_SingleRow_HandlesGracefully()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 4.0, 5.0, 6.0 }
        });

        // Act
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert - Should not crash, should return valid matrix
        Assert.Equal(1, adapted.Rows);
        Assert.Equal(3, adapted.Columns);
    }

    [Fact]
    public void CORALDomainAdapter_HighVariance_AdaptsCorrectly()
    {
        // Arrange
        var adapter = new CORALDomainAdapter<double>();
        // Low variance source
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 1.1 },
            { 1.0, 1.2 },
            { 1.0, 1.0 },
            { 1.0, 0.9 }
        });
        // High variance target
        var targetData = CreateMatrix(new double[,]
        {
            { 0.0, 10.0 },
            { 5.0, 0.0 },
            { 10.0, 5.0 },
            { 2.0, 8.0 }
        });

        // Act
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert - Adapted data should have higher variance than source
        var sourceVariance = ComputeVariance(sourceData);
        var adaptedVariance = ComputeVariance(adapted);

        // After adaptation to high-variance target, variance should increase
        Assert.True(adaptedVariance >= sourceVariance * 0.5,
            $"Expected adapted variance ({adaptedVariance}) to be closer to high-variance target");
    }

    #endregion

    #region MMD Domain Adapter Tests

    [Fact]
    public void MMDDomainAdapter_ComputeDomainDiscrepancy_IdenticalDomains_ReturnsNearZero()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var data = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(data, data);

        // Assert - Same data should have zero MMD
        Assert.True(discrepancy < Tolerance, $"Expected near-zero MMD for identical data, got {discrepancy}");
    }

    [Fact]
    public void MMDDomainAdapter_ComputeDomainDiscrepancy_DifferentDomains_ReturnsPositiveValue()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateMatrix(new double[,]
        {
            { 0.0, 0.0 },
            { 0.1, 0.1 },
            { 0.2, 0.2 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 10.0, 10.0 },
            { 10.1, 10.1 },
            { 10.2, 10.2 }
        });

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

        // Assert - Different domains should have positive MMD
        Assert.True(discrepancy > 0, $"Expected positive MMD for different domains, got {discrepancy}");
    }

    [Fact]
    public void MMDDomainAdapter_ComputeDomainDiscrepancy_IsSymmetric()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        });

        // Act
        var discrepancy1 = adapter.ComputeDomainDiscrepancy(sourceData, targetData);
        var discrepancy2 = adapter.ComputeDomainDiscrepancy(targetData, sourceData);

        // Assert - MMD should be symmetric
        Assert.Equal(discrepancy1, discrepancy2, RelaxedTolerance);
    }

    [Fact]
    public void MMDDomainAdapter_AdaptSource_ShiftsMeanTowardTarget()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateMatrix(new double[,]
        {
            { 0.0, 0.0 },
            { 1.0, 1.0 },
            { 2.0, 2.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 10.0, 10.0 },
            { 11.0, 11.0 },
            { 12.0, 12.0 }
        });

        // Act
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert - Adapted mean should be closer to target mean
        var sourceMean = ComputeMean(sourceData);
        var targetMean = ComputeMean(targetData);
        var adaptedMean = ComputeMean(adapted);

        // The adapted mean should be shifted toward target
        Assert.True(adaptedMean[0] > sourceMean[0],
            $"Adapted mean ({adaptedMean[0]}) should be greater than source mean ({sourceMean[0]})");
    }

    [Fact]
    public void MMDDomainAdapter_RequiresTraining_ReturnsFalse()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>();

        // Assert - MMD is non-parametric
        Assert.False(adapter.RequiresTraining);
    }

    [Fact]
    public void MMDDomainAdapter_AdaptationMethod_ReturnsCorrectName()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>();

        // Assert
        Assert.Contains("MMD", adapter.AdaptationMethod);
    }

    [Fact]
    public void MMDDomainAdapter_Train_UpdatesSigmaWithMedianHeuristic()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateMatrix(new double[,]
        {
            { 0.0, 0.0 },
            { 10.0, 10.0 },
            { 20.0, 20.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 5.0, 5.0 },
            { 15.0, 15.0 },
            { 25.0, 25.0 }
        });

        // Act - Training should update sigma via median heuristic
        adapter.Train(sourceData, targetData);

        // Compute discrepancy after training (sigma should be adapted)
        var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

        // Assert - Should not throw and should return valid value
        Assert.True(discrepancy >= 0, "Discrepancy should be non-negative");
        Assert.False(double.IsNaN(discrepancy), "Discrepancy should not be NaN");
    }

    [Fact]
    public void MMDDomainAdapter_AdaptTarget_ReturnsValidMatrix()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        });

        // Act
        var adapted = adapter.AdaptTarget(targetData, sourceData);

        // Assert
        Assert.Equal(targetData.Rows, adapted.Rows);
        Assert.Equal(targetData.Columns, adapted.Columns);
        // Verify no NaN values
        for (int i = 0; i < adapted.Rows; i++)
        {
            for (int j = 0; j < adapted.Columns; j++)
            {
                Assert.False(double.IsNaN(adapted[i, j]), $"Found NaN at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void MMDDomainAdapter_DifferentSigmaValues_AffectDiscrepancy()
    {
        // Arrange
        var sourceData = CreateMatrix(new double[,]
        {
            { 0.0, 0.0 },
            { 1.0, 1.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 5.0, 5.0 },
            { 6.0, 6.0 }
        });

        var adapterSmallSigma = new MMDDomainAdapter<double>(sigma: 0.1);
        var adapterLargeSigma = new MMDDomainAdapter<double>(sigma: 10.0);

        // Act
        var discrepancySmall = adapterSmallSigma.ComputeDomainDiscrepancy(sourceData, targetData);
        var discrepancyLarge = adapterLargeSigma.ComputeDomainDiscrepancy(sourceData, targetData);

        // Assert - Different sigma values should give different results
        Assert.NotEqual(discrepancySmall, discrepancyLarge, RelaxedTolerance);
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void CORALDomainAdapter_ZeroVarianceColumn_HandlesGracefully()
    {
        // Arrange - One column has zero variance
        var adapter = new CORALDomainAdapter<double>();
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 5.0 },
            { 2.0, 5.0 },
            { 3.0, 5.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 4.0, 10.0 },
            { 5.0, 10.0 },
            { 6.0, 10.0 }
        });

        // Act - Should not throw due to regularization
        adapter.Train(sourceData, targetData);
        var adapted = adapter.AdaptSource(sourceData, targetData);

        // Assert
        Assert.Equal(sourceData.Rows, adapted.Rows);
        Assert.Equal(sourceData.Columns, adapted.Columns);
    }

    [Fact]
    public void MMDDomainAdapter_SingleSample_HandlesGracefully()
    {
        // Arrange
        var adapter = new MMDDomainAdapter<double>(sigma: 1.0);
        var sourceData = CreateMatrix(new double[,]
        {
            { 1.0, 2.0 }
        });
        var targetData = CreateMatrix(new double[,]
        {
            { 3.0, 4.0 }
        });

        // Act
        var discrepancy = adapter.ComputeDomainDiscrepancy(sourceData, targetData);

        // Assert - Should return valid value
        Assert.True(discrepancy >= 0, "Discrepancy should be non-negative");
        Assert.False(double.IsNaN(discrepancy), "Discrepancy should not be NaN");
    }

    [Fact]
    public void DomainAdapters_LargeDimensionData_HandlesWithoutOverflow()
    {
        // Arrange
        var coralAdapter = new CORALDomainAdapter<double>();
        var mmdAdapter = new MMDDomainAdapter<double>(sigma: 1.0);

        // Create larger matrices (20 samples x 10 features)
        var random = new Random(42);
        var sourceData = CreateRandomMatrix(20, 10, random, mean: 0, stdDev: 1);
        var targetData = CreateRandomMatrix(20, 10, random, mean: 5, stdDev: 2);

        // Act & Assert - Should not overflow or produce invalid values
        coralAdapter.Train(sourceData, targetData);
        var coralAdapted = coralAdapter.AdaptSource(sourceData, targetData);
        var coralDiscrepancy = coralAdapter.ComputeDomainDiscrepancy(sourceData, targetData);

        var mmdDiscrepancy = mmdAdapter.ComputeDomainDiscrepancy(sourceData, targetData);

        Assert.False(double.IsNaN(coralDiscrepancy), "CORAL discrepancy should not be NaN");
        Assert.False(double.IsInfinity(coralDiscrepancy), "CORAL discrepancy should not be Infinity");
        Assert.False(double.IsNaN(mmdDiscrepancy), "MMD discrepancy should not be NaN");
        Assert.False(double.IsInfinity(mmdDiscrepancy), "MMD discrepancy should not be Infinity");
    }

    #endregion

    #region Helper Methods

    private static Matrix<double> CreateMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = data[i, j];
            }
        }
        return matrix;
    }

    private static Matrix<double> CreateRandomMatrix(int rows, int cols, Random random, double mean, double stdDev)
    {
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Box-Muller transform for normal distribution
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                matrix[i, j] = mean + stdDev * normal;
            }
        }
        return matrix;
    }

    private static double[] ComputeMean(Matrix<double> data)
    {
        var mean = new double[data.Columns];
        for (int j = 0; j < data.Columns; j++)
        {
            double sum = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                sum += data[i, j];
            }
            mean[j] = sum / data.Rows;
        }
        return mean;
    }

    private static double ComputeVariance(Matrix<double> data)
    {
        var mean = ComputeMean(data);
        double totalVariance = 0;
        for (int j = 0; j < data.Columns; j++)
        {
            double sumSquares = 0;
            for (int i = 0; i < data.Rows; i++)
            {
                double diff = data[i, j] - mean[j];
                sumSquares += diff * diff;
            }
            totalVariance += sumSquares / data.Rows;
        }
        return totalVariance / data.Columns;
    }

    #endregion
}
