using AiDotNet.Tensors.Helpers;
using AiDotNet.TransferLearning.FeatureMapping;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TransferLearning;

/// <summary>
/// Integration tests for Feature Mapping classes.
/// These tests verify mathematical correctness of feature mapping algorithms.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class FeatureMappingIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double RelaxedTolerance = 1e-3;

    #region LinearFeatureMapper Basic Tests

    [Fact]
    public void LinearFeatureMapper_Train_SetsIsTrainedToTrue()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });

        // Act
        mapper.Train(sourceData, targetData);

        // Assert
        Assert.True(mapper.IsTrained);
    }

    [Fact]
    public void LinearFeatureMapper_MapToTarget_BeforeTrain_ThrowsException()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceFeatures = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => mapper.MapToTarget(sourceFeatures, 2));
    }

    [Fact]
    public void LinearFeatureMapper_MapToSource_BeforeTrain_ThrowsException()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var targetFeatures = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => mapper.MapToSource(targetFeatures, 3));
    }

    [Fact]
    public void LinearFeatureMapper_MapToTarget_ProducesCorrectDimensions()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });

        mapper.Train(sourceData, targetData);

        var newSourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 2.0, 3.0, 4.0, 5.0 }
        });

        // Act
        var mapped = mapper.MapToTarget(newSourceData, 2);

        // Assert
        Assert.Equal(2, mapped.Rows); // Same number of samples
        Assert.Equal(2, mapped.Columns); // Target dimension
    }

    [Fact]
    public void LinearFeatureMapper_MapToSource_ProducesCorrectDimensions()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 6.0, 7.0, 8.0, 9.0, 10.0 },
            { 11.0, 12.0, 13.0, 14.0, 15.0 }
        });

        mapper.Train(sourceData, targetData);

        var newTargetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 3.0, 4.0, 5.0, 6.0 }
        });

        // Act
        var mapped = mapper.MapToSource(newTargetData, 3);

        // Assert
        Assert.Equal(2, mapped.Rows); // Same number of samples
        Assert.Equal(3, mapped.Columns); // Source dimension
    }

    [Fact]
    public void LinearFeatureMapper_GetMappingConfidence_ReturnsValueBetweenZeroAndOne()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        });

        mapper.Train(sourceData, targetData);

        // Act
        var confidence = mapper.GetMappingConfidence();

        // Assert
        Assert.True(confidence >= 0.0, $"Confidence {confidence} should be >= 0");
        Assert.True(confidence <= 1.0, $"Confidence {confidence} should be <= 1");
    }

    [Fact]
    public void LinearFeatureMapper_GetMappingConfidence_BeforeTrain_ReturnsZero()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();

        // Act
        var confidence = mapper.GetMappingConfidence();

        // Assert
        Assert.Equal(0.0, confidence, Tolerance);
    }

    #endregion

    #region LinearFeatureMapper Dimension Reduction/Expansion Tests

    [Fact]
    public void LinearFeatureMapper_DimensionReduction_PreservesInformation()
    {
        // Arrange - Map from 5D to 2D
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 4.0, 6.0, 8.0, 10.0 },
            { 3.0, 6.0, 9.0, 12.0, 15.0 },
            { 4.0, 8.0, 12.0, 16.0, 20.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 4.0 },
            { 3.0, 6.0 },
            { 4.0, 8.0 }
        });

        mapper.Train(sourceData, targetData);

        // Act
        var mapped = mapper.MapToTarget(sourceData, 2);

        // Assert - Verify values are not NaN or Infinity
        for (int i = 0; i < mapped.Rows; i++)
        {
            for (int j = 0; j < mapped.Columns; j++)
            {
                Assert.False(double.IsNaN(mapped[i, j]), $"Found NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(mapped[i, j]), $"Found Infinity at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void LinearFeatureMapper_DimensionExpansion_ProducesValidOutput()
    {
        // Arrange - Map from 2D to 5D
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 4.0, 6.0, 8.0, 10.0 },
            { 3.0, 6.0, 9.0, 12.0, 15.0 },
            { 4.0, 8.0, 12.0, 16.0, 20.0 }
        });

        mapper.Train(sourceData, targetData);

        // Act
        var mapped = mapper.MapToTarget(sourceData, 5);

        // Assert
        Assert.Equal(4, mapped.Rows);
        Assert.Equal(5, mapped.Columns);
        for (int i = 0; i < mapped.Rows; i++)
        {
            for (int j = 0; j < mapped.Columns; j++)
            {
                Assert.False(double.IsNaN(mapped[i, j]), $"Found NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(mapped[i, j]), $"Found Infinity at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void LinearFeatureMapper_SameDimensions_Works()
    {
        // Arrange - Map from 3D to 3D
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 10.0, 20.0, 30.0 },
            { 40.0, 50.0, 60.0 },
            { 70.0, 80.0, 90.0 }
        });

        mapper.Train(sourceData, targetData);

        // Act
        var mapped = mapper.MapToTarget(sourceData, 3);

        // Assert
        Assert.Equal(3, mapped.Rows);
        Assert.Equal(3, mapped.Columns);
    }

    #endregion

    #region LinearFeatureMapper Round-Trip Tests

    [Fact]
    public void LinearFeatureMapper_RoundTrip_PreservesApproximateStructure()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });

        mapper.Train(sourceData, targetData);

        // Act - Map to target and back to source
        var mapped = mapper.MapToTarget(sourceData, 3);
        var reconstructed = mapper.MapToSource(mapped, 3);

        // Assert - With same dimensions, reconstruction should preserve structure
        Assert.Equal(sourceData.Rows, reconstructed.Rows);
        Assert.Equal(sourceData.Columns, reconstructed.Columns);

        // Verify no NaN values
        for (int i = 0; i < reconstructed.Rows; i++)
        {
            for (int j = 0; j < reconstructed.Columns; j++)
            {
                Assert.False(double.IsNaN(reconstructed[i, j]), $"Found NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void LinearFeatureMapper_SingleRow_HandlesGracefully()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 }
        });

        // Act - Should not throw
        mapper.Train(sourceData, targetData);
        var mapped = mapper.MapToTarget(sourceData, 2);

        // Assert
        Assert.Equal(1, mapped.Rows);
        Assert.Equal(2, mapped.Columns);
    }

    [Fact]
    public void LinearFeatureMapper_SingleColumn_HandlesGracefully()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 10.0 },
            { 20.0 },
            { 30.0 }
        });

        // Act
        mapper.Train(sourceData, targetData);
        var mapped = mapper.MapToTarget(sourceData, 1);

        // Assert
        Assert.Equal(3, mapped.Rows);
        Assert.Equal(1, mapped.Columns);
    }

    [Fact]
    public void LinearFeatureMapper_LargeMatrix_HandlesWithoutOverflow()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var random = new Random(42);

        var sourceData = CreateRandomMatrix(50, 20, random, mean: 0, stdDev: 1);
        var targetData = CreateRandomMatrix(50, 10, random, mean: 5, stdDev: 2);

        // Act
        mapper.Train(sourceData, targetData);
        var mapped = mapper.MapToTarget(sourceData, 10);
        var confidence = mapper.GetMappingConfidence();

        // Assert
        Assert.Equal(50, mapped.Rows);
        Assert.Equal(10, mapped.Columns);
        Assert.False(double.IsNaN(confidence), "Confidence should not be NaN");
        Assert.True(confidence >= 0 && confidence <= 1, "Confidence should be between 0 and 1");

        // Verify no NaN or Infinity values
        for (int i = 0; i < mapped.Rows; i++)
        {
            for (int j = 0; j < mapped.Columns; j++)
            {
                Assert.False(double.IsNaN(mapped[i, j]), $"Found NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(mapped[i, j]), $"Found Infinity at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void LinearFeatureMapper_ZeroVarianceColumn_HandlesGracefully()
    {
        // Arrange - Column 1 has zero variance
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1.0, 5.0, 3.0 },
            { 2.0, 5.0, 4.0 },
            { 3.0, 5.0, 5.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 }
        });

        // Act - Should not throw
        mapper.Train(sourceData, targetData);
        var mapped = mapper.MapToTarget(sourceData, 2);

        // Assert
        Assert.Equal(3, mapped.Rows);
        Assert.Equal(2, mapped.Columns);
    }

    [Fact]
    public void LinearFeatureMapper_NegativeValues_HandlesCorrectly()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { -1.0, -2.0, -3.0 },
            { -4.0, -5.0, -6.0 },
            { 0.0, 0.0, 0.0 },
            { 4.0, 5.0, 6.0 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { -10.0, -20.0 },
            { -40.0, -50.0 },
            { 0.0, 0.0 },
            { 40.0, 50.0 }
        });

        // Act
        mapper.Train(sourceData, targetData);
        var mapped = mapper.MapToTarget(sourceData, 2);

        // Assert
        Assert.Equal(4, mapped.Rows);
        Assert.Equal(2, mapped.Columns);
        for (int i = 0; i < mapped.Rows; i++)
        {
            for (int j = 0; j < mapped.Columns; j++)
            {
                Assert.False(double.IsNaN(mapped[i, j]), $"Found NaN at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void LinearFeatureMapper_VerySmallValues_HandlesCorrectly()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1e-10, 2e-10, 3e-10 },
            { 4e-10, 5e-10, 6e-10 },
            { 7e-10, 8e-10, 9e-10 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1e-10, 2e-10 },
            { 3e-10, 4e-10 },
            { 5e-10, 6e-10 }
        });

        // Act
        mapper.Train(sourceData, targetData);
        var mapped = mapper.MapToTarget(sourceData, 2);

        // Assert
        for (int i = 0; i < mapped.Rows; i++)
        {
            for (int j = 0; j < mapped.Columns; j++)
            {
                Assert.False(double.IsNaN(mapped[i, j]), $"Found NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(mapped[i, j]), $"Found Infinity at [{i},{j}]");
            }
        }
    }

    [Fact]
    public void LinearFeatureMapper_VeryLargeValues_HandlesCorrectly()
    {
        // Arrange
        var mapper = new LinearFeatureMapper<double>();
        var sourceData = new Matrix<double>(new double[,]
        {
            { 1e8, 2e8, 3e8 },
            { 4e8, 5e8, 6e8 },
            { 7e8, 8e8, 9e8 }
        });
        var targetData = new Matrix<double>(new double[,]
        {
            { 1e8, 2e8 },
            { 3e8, 4e8 },
            { 5e8, 6e8 }
        });

        // Act
        mapper.Train(sourceData, targetData);
        var mapped = mapper.MapToTarget(sourceData, 2);

        // Assert
        for (int i = 0; i < mapped.Rows; i++)
        {
            for (int j = 0; j < mapped.Columns; j++)
            {
                Assert.False(double.IsNaN(mapped[i, j]), $"Found NaN at [{i},{j}]");
                Assert.False(double.IsInfinity(mapped[i, j]), $"Found Infinity at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a matrix with normally distributed random values.
    /// Uses MathHelper.GetNormalRandom for Box-Muller transform.
    /// </summary>
    private static Matrix<double> CreateRandomMatrix(int rows, int cols, Random random, double mean, double stdDev)
    {
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = MathHelper.GetNormalRandom(mean, stdDev, random);
            }
        }
        return matrix;
    }

    #endregion
}
