using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Unit tests for imputer classes: SimpleImputer, KNNImputer, and IterativeImputer.
/// </summary>
public class ImputerTests
{
    private const double Tolerance = 1e-6;

    #region SimpleImputer Tests

    [Fact]
    public void SimpleImputer_MeanStrategy_ImputesMissingValues()
    {
        // Arrange - Column 0: [1, NaN, 3], mean = 2
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(2.0, result[1, 0], Tolerance); // NaN replaced with mean
        Assert.Equal(20.0, result[1, 1], Tolerance); // Unchanged
    }

    [Fact]
    public void SimpleImputer_MedianStrategy_ImputesMissingValues()
    {
        // Arrange - Column 0: [1, NaN, 3, 5], median = 3
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { double.NaN },
            { 3.0 },
            { 5.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Median);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(3.0, result[1, 0], Tolerance); // NaN replaced with median
    }

    [Fact]
    public void SimpleImputer_MostFrequentStrategy_ImputesMissingValues()
    {
        // Arrange - Column 0: [1, NaN, 2, 2, 1, 2], most frequent = 2
        var data = new Matrix<double>(new double[,]
        {
            { 1.0 },
            { double.NaN },
            { 2.0 },
            { 2.0 },
            { 1.0 },
            { 2.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.MostFrequent);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(2.0, result[1, 0], Tolerance); // NaN replaced with most frequent
    }

    [Fact]
    public void SimpleImputer_ConstantStrategy_ImputesMissingValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { double.NaN, double.NaN },
            { 3.0, 30.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Constant, fillValue: -999.0);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(-999.0, result[1, 0], Tolerance);
        Assert.Equal(-999.0, result[1, 1], Tolerance);
    }

    [Fact]
    public void SimpleImputer_ConstantStrategy_WithDefaultFillValue_UsesNumOpsZero()
    {
        // Arrange - When no fill value is explicitly provided, the imputer uses NumOps.Zero
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        // When fillValue is not specified, it falls back to NumOps.Zero (0.0 for double)
        var imputer = new SimpleImputer<double>(ImputationStrategy.Constant);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert - NaN should be replaced with NumOps.Zero (0.0 for double)
        Assert.Equal(0.0, result[1, 0], Tolerance);
        Assert.Equal(20.0, result[1, 1], Tolerance);
    }

    [Fact]
    public void SimpleImputer_SpecificColumns_OnlyImputesSelectedColumns()
    {
        // Arrange - Only impute column 0
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, double.NaN },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean, columnIndices: new[] { 0 });

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(2.0, result[1, 0], Tolerance); // Imputed
        Assert.True(double.IsNaN(result[0, 1])); // Not imputed (column 1 not selected)
    }

    [Fact]
    public void SimpleImputer_NoMissingValues_ReturnsOriginalData()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[0, 1], Tolerance);
        Assert.Equal(3.0, result[1, 0], Tolerance);
        Assert.Equal(4.0, result[1, 1], Tolerance);
    }

    [Fact]
    public void SimpleImputer_AllMissingInColumn_ReturnsZero()
    {
        // Arrange - Column 0 has all NaN
        var data = new Matrix<double>(new double[,]
        {
            { double.NaN, 10.0 },
            { double.NaN, 20.0 },
            { double.NaN, 30.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert - All NaN replaced with 0 (default when no valid values)
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(0.0, result[1, 0], Tolerance);
        Assert.Equal(0.0, result[2, 0], Tolerance);
    }

    [Fact]
    public void SimpleImputer_Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => imputer.Transform(data));
    }

    [Fact]
    public void SimpleImputer_FitTransform_WorksCorrectly()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act
        var result = imputer.FitTransform(data);

        // Assert
        Assert.Equal(2.0, result[1, 0], Tolerance);
    }

    [Fact]
    public void SimpleImputer_Statistics_ReturnsComputedValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);

        // Act
        imputer.Fit(data);

        // Assert
        Assert.NotNull(imputer.Statistics);
        Assert.Equal(2.0, imputer.Statistics[0], Tolerance); // Mean of [1, 3]
        Assert.Equal(20.0, imputer.Statistics[1], Tolerance); // Mean of [10, 20, 30]
    }

    [Fact]
    public void SimpleImputer_InverseTransform_ThrowsNotSupported()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
        imputer.Fit(data);

        // Act & Assert
        Assert.False(imputer.SupportsInverseTransform);
        Assert.Throws<NotSupportedException>(() => imputer.InverseTransform(data));
    }

    #endregion

    #region KNNImputer Tests

    [Fact]
    public void KNNImputer_UniformWeights_ImputesMissingValues()
    {
        // Arrange - Simple case where nearest neighbors have clear values
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { double.NaN, 15.0 }, // Missing in column 0, similar to rows 0 and 1
            { 3.0, 30.0 }
        });

        var imputer = new KNNImputer<double>(nNeighbors: 2, weights: KNNWeights.Uniform);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert - Should be imputed based on nearest neighbors
        Assert.False(double.IsNaN(result[2, 0]));
    }

    [Fact]
    public void KNNImputer_DistanceWeights_ImputesMissingValues()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 20.0 },
            { double.NaN, 15.0 },
            { 3.0, 30.0 }
        });

        var imputer = new KNNImputer<double>(nNeighbors: 2, weights: KNNWeights.Distance);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.False(double.IsNaN(result[2, 0]));
    }

    [Fact]
    public void KNNImputer_InvalidNNeighbors_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new KNNImputer<double>(nNeighbors: 0));
        Assert.Throws<ArgumentException>(() => new KNNImputer<double>(nNeighbors: -1));
    }

    [Fact]
    public void KNNImputer_Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var imputer = new KNNImputer<double>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => imputer.Transform(data));
    }

    [Fact]
    public void KNNImputer_NoMissingValues_ReturnsOriginalData()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        var imputer = new KNNImputer<double>();

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[0, 1], Tolerance);
    }

    [Fact]
    public void KNNImputer_SpecificColumns_OnlyImputesSelectedColumns()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, double.NaN },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        var imputer = new KNNImputer<double>(nNeighbors: 2, columnIndices: new[] { 0 });

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.False(double.IsNaN(result[1, 0])); // Imputed
        Assert.True(double.IsNaN(result[0, 1])); // Not imputed
    }

    [Fact]
    public void KNNImputer_InverseTransform_ThrowsNotSupported()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var imputer = new KNNImputer<double>();
        imputer.Fit(data);

        // Act & Assert
        Assert.False(imputer.SupportsInverseTransform);
        Assert.Throws<NotSupportedException>(() => imputer.InverseTransform(data));
    }

    [Fact]
    public void KNNImputer_DefaultNNeighbors_IsFive()
    {
        // Arrange
        var imputer = new KNNImputer<double>();

        // Assert
        Assert.Equal(5, imputer.NNeighbors);
    }

    #endregion

    #region IterativeImputer Tests

    [Fact]
    public void IterativeImputer_ImputesMissingValues()
    {
        // Arrange - Data with clear linear relationship
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 4.0 },
            { 3.0, double.NaN }, // Missing: should be ~6
            { 4.0, 8.0 },
            { 5.0, 10.0 }
        });

        var imputer = new IterativeImputer<double>(maxIterations: 10);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert - Imputed value should be close to 6 (based on linear relationship)
        Assert.False(double.IsNaN(result[2, 1]));
        // The imputed value should be reasonably close to 6
        Assert.True(Math.Abs(result[2, 1] - 6.0) < 2.0);
    }

    [Fact]
    public void IterativeImputer_InvalidMaxIterations_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new IterativeImputer<double>(maxIterations: 0));
        Assert.Throws<ArgumentException>(() => new IterativeImputer<double>(maxIterations: -1));
    }

    [Fact]
    public void IterativeImputer_InvalidTolerance_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new IterativeImputer<double>(tolerance: -0.1));
    }

    [Fact]
    public void IterativeImputer_Transform_BeforeFit_ThrowsException()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var imputer = new IterativeImputer<double>();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => imputer.Transform(data));
    }

    [Fact]
    public void IterativeImputer_NoMissingValues_ReturnsOriginalData()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        var imputer = new IterativeImputer<double>();

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[0, 1], Tolerance);
    }

    [Fact]
    public void IterativeImputer_MeanInitialStrategy_Works()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        var imputer = new IterativeImputer<double>(
            initialStrategy: IterativeImputerInitialStrategy.Mean);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.False(double.IsNaN(result[1, 0]));
    }

    [Fact]
    public void IterativeImputer_MedianInitialStrategy_Works()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 10.0 },
            { double.NaN, 20.0 },
            { 3.0, 30.0 }
        });

        var imputer = new IterativeImputer<double>(
            initialStrategy: IterativeImputerInitialStrategy.Median);

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.False(double.IsNaN(result[1, 0]));
    }

    [Fact]
    public void IterativeImputer_InverseTransform_ThrowsNotSupported()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var imputer = new IterativeImputer<double>();
        imputer.Fit(data);

        // Act & Assert
        Assert.False(imputer.SupportsInverseTransform);
        Assert.Throws<NotSupportedException>(() => imputer.InverseTransform(data));
    }

    [Fact]
    public void IterativeImputer_MultipleMissingColumns_ImputesAll()
    {
        // Arrange
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { double.NaN, 4.0, double.NaN },
            { 3.0, 6.0, 9.0 }
        });

        var imputer = new IterativeImputer<double>();

        // Act
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.False(double.IsNaN(result[1, 0]));
        Assert.False(double.IsNaN(result[1, 2]));
    }

    [Fact]
    public void IterativeImputer_Convergence_StopsEarly()
    {
        // Arrange - Simple data that should converge quickly
        var data = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 2.0, 2.0 },
            { double.NaN, 3.0 },
            { 4.0, 4.0 }
        });

        var imputer = new IterativeImputer<double>(maxIterations: 100, tolerance: 1e-3);

        // Act - Should complete without timeout
        imputer.Fit(data);
        var result = imputer.Transform(data);

        // Assert
        Assert.False(double.IsNaN(result[2, 0]));
    }

    #endregion

    #region Properties and Configuration Tests

    [Fact]
    public void SimpleImputer_Properties_ReturnCorrectValues()
    {
        // Arrange
        var imputer = new SimpleImputer<double>(
            ImputationStrategy.Constant,
            fillValue: 42.0);

        // Assert
        Assert.Equal(ImputationStrategy.Constant, imputer.Strategy);
        Assert.Equal(42.0, imputer.FillValue);
    }

    [Fact]
    public void KNNImputer_Properties_ReturnCorrectValues()
    {
        // Arrange
        var imputer = new KNNImputer<double>(nNeighbors: 10, weights: KNNWeights.Distance);

        // Assert
        Assert.Equal(10, imputer.NNeighbors);
        Assert.Equal(KNNWeights.Distance, imputer.Weights);
    }

    [Fact]
    public void IterativeImputer_Properties_ReturnCorrectValues()
    {
        // Arrange
        var imputer = new IterativeImputer<double>(
            maxIterations: 20,
            tolerance: 1e-4,
            estimator: IterativeImputerEstimator.Ridge,
            initialStrategy: IterativeImputerInitialStrategy.Median);

        // Assert
        Assert.Equal(20, imputer.MaxIterations);
        Assert.Equal(1e-4, imputer.Tolerance);
        Assert.Equal(IterativeImputerEstimator.Ridge, imputer.Estimator);
        Assert.Equal(IterativeImputerInitialStrategy.Median, imputer.InitialStrategy);
    }

    #endregion

    #region GetFeatureNamesOut Tests

    [Fact]
    public void SimpleImputer_GetFeatureNamesOut_ReturnsInputNames()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var imputer = new SimpleImputer<double>();
        imputer.Fit(data);

        // Act
        var names = imputer.GetFeatureNamesOut(new[] { "feature1", "feature2" });

        // Assert
        Assert.Equal(new[] { "feature1", "feature2" }, names);
    }

    [Fact]
    public void KNNImputer_GetFeatureNamesOut_ReturnsInputNames()
    {
        // Arrange
        var data = new Matrix<double>(new double[,] { { 1.0, 2.0 } });
        var imputer = new KNNImputer<double>();
        imputer.Fit(data);

        // Act
        var names = imputer.GetFeatureNamesOut(new[] { "a", "b" });

        // Assert
        Assert.Equal(new[] { "a", "b" }, names);
    }

    #endregion
}
