using AiDotNet.Enums;
using AiDotNet.FeatureSelectors;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FeatureSelectors;

/// <summary>
/// Integration tests for FeatureSelectors module.
/// Tests feature selection accuracy, edge cases, and validates correct feature identification.
/// </summary>
public class FeatureSelectorsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(double[,] data)
    {
        return new Matrix<double>(data);
    }

    #endregion

    #region VarianceThresholdFeatureSelector Tests

    [Fact]
    public void VarianceThresholdFeatureSelector_RemovesConstantFeatures()
    {
        // Arrange - Column 0 is constant (variance = 0), column 1 varies
        var data = CreateTestMatrix(new double[,]
        {
            { 5.0, 1.0 },
            { 5.0, 2.0 },
            { 5.0, 3.0 },
            { 5.0, 4.0 }
        });
        // Use small positive threshold to remove constant features (variance=0)
        // threshold=0 would keep ALL features since variance >= 0 is always true
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1e-10);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Only column 1 should remain (column 0 has zero variance)
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[1, 0], Tolerance);
        Assert.Equal(3.0, result[2, 0], Tolerance);
        Assert.Equal(4.0, result[3, 0], Tolerance);
    }

    [Fact]
    public void VarianceThresholdFeatureSelector_RemovesLowVarianceFeatures()
    {
        // Arrange - Column 0 has low variance, column 1 has high variance
        // Column 0: [1, 1.1, 0.9, 1] - variance ~0.0067
        // Column 1: [1, 10, 20, 30] - variance ~142
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 1.0 },
            { 1.1, 10.0 },
            { 0.9, 20.0 },
            { 1.0, 30.0 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1.0);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Only column 1 should remain
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(10.0, result[1, 0], Tolerance);
    }

    [Fact]
    public void VarianceThresholdFeatureSelector_KeepsAllFeaturesAboveThreshold()
    {
        // Arrange - Both columns have variance above threshold
        var data = CreateTestMatrix(new double[,]
        {
            { 0.0, 0.0 },
            { 10.0, 100.0 },
            { 20.0, 200.0 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.1);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Both columns should remain
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void VarianceThresholdFeatureSelector_DefaultThreshold_RemovesNearConstantFeatures()
    {
        // Arrange - Default threshold is 0.1
        // Column 0 has tiny variance, column 1 has larger variance
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 0.0 },
            { 1.001, 10.0 },
            { 0.999, 20.0 },
            { 1.0, 30.0 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>();

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Column 0 variance is tiny (well below 0.1), only column 1 remains
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void VarianceThresholdFeatureSelector_AllConstant_ThrowsException()
    {
        // Arrange - All columns are constant
        var data = CreateTestMatrix(new double[,]
        {
            { 5.0, 3.0 },
            { 5.0, 3.0 },
            { 5.0, 3.0 }
        });
        // Use small positive threshold to require some variance (removes zero-variance features)
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1e-10);

        // Act & Assert - When all features are removed, Matrix creation throws since empty matrices aren't supported
        Assert.Throws<ArgumentException>(() => selector.SelectFeatures(data));
    }

    [Fact]
    public void VarianceThresholdFeatureSelector_SingleColumn_HighVariance_Keeps()
    {
        // Arrange - Single column with variance
        var data = CreateTestMatrix(new double[,]
        {
            { 0.0 },
            { 10.0 },
            { 20.0 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.0);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void VarianceThresholdFeatureSelector_NegativeValues_CalculatesVarianceCorrectly()
    {
        // Arrange - Data with negative values
        var data = CreateTestMatrix(new double[,]
        {
            { -10.0, 0.0 },
            { 0.0, 0.0 },
            { 10.0, 0.0 }
        });
        // Use small positive threshold to remove zero-variance features
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1e-10);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Column 0 has variance, column 1 is constant
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(-10.0, result[0, 0], Tolerance);
    }

    #endregion

    #region CorrelationFeatureSelector Tests

    [Fact]
    public void CorrelationFeatureSelector_RemovesPerfectlyCorrelatedFeatures()
    {
        // Arrange - Column 1 is perfectly correlated with column 0 (y = 2x)
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 100.0 },
            { 2.0, 4.0, 200.0 },
            { 3.0, 6.0, 300.0 },
            { 4.0, 8.0, 400.0 }
        });
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Column 0 and 2 should remain, column 1 removed (correlated with 0)
        // Column 2 is also correlated with 0 and 1, but selector adds first, then checks
        Assert.Equal(4, result.Rows);
        Assert.True(result.Columns < 3); // At least one correlated feature removed
    }

    [Fact]
    public void CorrelationFeatureSelector_KeepsIndependentFeatures()
    {
        // Arrange - Independent features (low correlation)
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 10.0 },
            { 2.0, 5.0 },
            { 3.0, 15.0 },
            { 4.0, 2.0 }
        });
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Both columns should remain if uncorrelated
        Assert.Equal(4, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void CorrelationFeatureSelector_HighThreshold_KeepsMoreFeatures()
    {
        // Arrange - Moderately correlated features
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 5.0 },
            { 4.0, 6.0 }
        });

        // With threshold 0.99, even moderately correlated features should be kept
        var selectorHigh = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.99);
        // With threshold 0.5, moderately correlated features should be removed
        var selectorLow = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.5);

        // Act
        var resultHigh = selectorHigh.SelectFeatures(data);
        var resultLow = selectorLow.SelectFeatures(data);

        // Assert
        Assert.True(resultHigh.Columns >= resultLow.Columns);
    }

    [Fact]
    public void CorrelationFeatureSelector_NegativeCorrelation_Detected()
    {
        // Arrange - Column 1 is negatively correlated with column 0 (y = -x)
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, -1.0 },
            { 2.0, -2.0 },
            { 3.0, -3.0 },
            { 4.0, -4.0 }
        });
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Negative correlation should be detected (abs correlation = 1.0)
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns); // One removed due to perfect negative correlation
    }

    [Fact]
    public void CorrelationFeatureSelector_SingleFeature_ReturnsFeature()
    {
        // Arrange - Single feature
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0 },
            { 2.0 },
            { 3.0 }
        });
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.5);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    #endregion

    #region UnivariateFeatureSelector Tests

    [Fact]
    public void UnivariateFeatureSelector_FValue_SelectsTopKFeatures()
    {
        // Arrange - Feature 0 is more predictive of target than feature 1
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0 },
            { 1.0, 6.0 },
            { 2.0, 5.0 },
            { 2.0, 6.0 },
            { 3.0, 5.0 },
            { 3.0, 6.0 }
        });
        // Target: classes match feature 0 pattern
        var target = new Vector<double>(new double[] { 0, 0, 1, 1, 2, 2 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.FValue,
            k: 1
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Should select 1 feature
        Assert.Equal(6, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void UnivariateFeatureSelector_DefaultK_SelectsHalfFeatures()
    {
        // Arrange - 4 features, default k should select 2
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 2.0, 3.0, 4.0, 5.0 },
            { 3.0, 4.0, 5.0, 6.0 }
        });
        var target = new Vector<double>(new double[] { 0, 1, 2 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.FValue
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Default should select 50% = 2 features
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void UnivariateFeatureSelector_InvalidK_ThrowsException()
    {
        // Arrange & Act & Assert
        var target = new Vector<double>(new double[] { 0, 1, 2 });
        Assert.Throws<ArgumentException>(() => new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            k: 0
        ));
        Assert.Throws<ArgumentException>(() => new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            k: -1
        ));
    }

    [Fact]
    public void UnivariateFeatureSelector_NullTarget_ThrowsException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentNullException>(() => new UnivariateFeatureSelector<double, Matrix<double>>(
            target: null!
        ));
    }

    [Fact]
    public void UnivariateFeatureSelector_TargetLengthMismatch_ThrowsException()
    {
        // Arrange - 3 samples but target has 2 elements
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var target = new Vector<double>(new double[] { 0, 1 }); // Mismatch!
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            k: 1
        );

        // Act & Assert
        Assert.Throws<ArgumentException>(() => selector.SelectFeatures(data));
    }

    [Fact]
    public void UnivariateFeatureSelector_MutualInformation_SelectsFeatures()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 10.0 },
            { 1.0, 20.0 },
            { 2.0, 10.0 },
            { 2.0, 20.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.MutualInformation,
            k: 1
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Should select 1 feature
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    #endregion

    #region NoFeatureSelector Tests

    [Fact]
    public void NoFeatureSelector_ReturnsAllFeatures()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });
        var selector = new NoFeatureSelector<double, Matrix<double>>();

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - All features should remain unchanged
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(5.0, result[1, 1], Tolerance);
        Assert.Equal(9.0, result[2, 2], Tolerance);
    }

    [Fact]
    public void NoFeatureSelector_EmptyMatrix_ThrowsException()
    {
        // Arrange & Act & Assert
        // The Matrix constructor doesn't allow empty matrices
        Assert.Throws<ArgumentException>(() => CreateTestMatrix(new double[,] { }));
    }

    [Fact]
    public void NoFeatureSelector_SingleElement_ReturnsSame()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,] { { 42.0 } });
        var selector = new NoFeatureSelector<double, Matrix<double>>();

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(1, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(42.0, result[0, 0], Tolerance);
    }

    #endregion

    #region Edge Cases and Integration Tests

    [Fact]
    public void VarianceThresholdFeatureSelector_LargeDataset_PerformsCorrectly()
    {
        // Arrange - Large dataset with mixed variance columns
        int rows = 1000;
        int cols = 10;
        var data = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (j % 2 == 0)
                {
                    // Even columns: constant
                    data[i, j] = 5.0;
                }
                else
                {
                    // Odd columns: varying
                    data[i, j] = i * (j + 1);
                }
            }
        }

        var matrix = CreateTestMatrix(data);
        // Use small positive threshold to remove constant columns (even columns)
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1e-10);

        // Act
        var result = selector.SelectFeatures(matrix);

        // Assert - Only odd columns (5 columns) should remain
        Assert.Equal(rows, result.Rows);
        Assert.Equal(5, result.Columns);
    }

    [Fact]
    public void FeatureSelector_MoreFeaturesThanSamples_HandlesCorrectly()
    {
        // Arrange - Wide data (more features than samples)
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 2.0, 4.0, 6.0, 8.0, 10.0 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.0);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - All columns have variance, all should remain
        Assert.Equal(2, result.Rows);
        Assert.Equal(5, result.Columns);
    }

    [Fact]
    public void CorrelationFeatureSelector_AllCorrelated_ReturnsFirstFeature()
    {
        // Arrange - All features are perfectly correlated
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 2.0, 4.0, 6.0 },
            { 3.0, 6.0, 9.0 },
            { 4.0, 8.0, 12.0 }
        });
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Only the first feature should remain
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[1, 0], Tolerance);
    }

    [Fact]
    public void VarianceThreshold_ZeroThreshold_KeepsAllFeatures()
    {
        // Arrange - Threshold=0 uses >= comparison, so variance >= 0 keeps ALL features
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0, 0.0 },
            { 1.0, 5.0, 0.001 },
            { 1.0, 5.0, -0.001 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.0);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - With threshold=0, ALL features are kept (variance >= 0 is always true)
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void Selectors_ChainedSelection_WorksCorrectly()
    {
        // Arrange - First remove low variance, then check correlation
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0, 10.0, 2.0 },  // Col 1 is constant, Col 0 and 3 are correlated
            { 2.0, 5.0, 20.0, 4.0 },
            { 3.0, 5.0, 30.0, 6.0 },
            { 4.0, 5.0, 40.0, 8.0 }
        });

        // Use small positive threshold to remove constant column (column 1)
        var varianceSelector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1e-10);
        var correlationSelector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);

        // Act - Chain selectors
        var afterVariance = varianceSelector.SelectFeatures(data);
        var afterCorrelation = correlationSelector.SelectFeatures(afterVariance);

        // Assert
        // After variance: columns 0, 2, 3 remain (column 1 is constant)
        Assert.Equal(4, afterVariance.Rows);
        Assert.Equal(3, afterVariance.Columns);

        // After correlation: correlated columns should be reduced
        Assert.Equal(4, afterCorrelation.Rows);
        Assert.True(afterCorrelation.Columns < 3);
    }

    [Fact]
    public void VarianceThreshold_VerySmallVariance_HandledCorrectly()
    {
        // Arrange - The 1e-15 variation results in variance ~1e-30 which is below 1e-20
        // Column 0: variance from 1e-15 changes is ~(1e-15)^2 / 2 ≈ 5e-31 (below 1e-20)
        // Column 1: variance from [1, 2, 3] is ~1.0 (above 1e-20)
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 1.0 },
            { 1.0 + 1e-15, 2.0 },
            { 1.0 - 1e-15, 3.0 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1e-20);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Only column 1 remains because column 0's variance (~1e-30) < 1e-20
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void CorrelationFeatureSelector_ZeroVarianceFeature_HandlesGracefully()
    {
        // Arrange - One feature has zero variance (constant)
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0 },
            { 2.0, 5.0 },
            { 3.0, 5.0 }
        });
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.5);

        // Act - Should not crash on zero-variance feature
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.True(result.Columns >= 1);
    }

    [Fact]
    public void UnivariateFeatureSelector_SingleClass_HandlesGracefully()
    {
        // Arrange - All samples have the same class
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 0 }); // Single class
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.FValue,
            k: 1
        );

        // Act - Should not crash, scores may be zero
        var result = selector.SelectFeatures(data);

        // Assert - Should still return at least 1 feature
        Assert.Equal(3, result.Rows);
        Assert.True(result.Columns >= 1);
    }

    [Fact]
    public void VarianceThreshold_FloatingPointPrecision_ConsistentResults()
    {
        // Arrange - Test floating point edge case
        var data = CreateTestMatrix(new double[,]
        {
            { 0.1, 0.2 },
            { 0.2, 0.3 },
            { 0.3, 0.4 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.0);

        // Act
        var result1 = selector.SelectFeatures(data);
        var result2 = selector.SelectFeatures(data);

        // Assert - Results should be identical
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
    }

    [Fact]
    public void UnivariateFeatureSelector_KGreaterThanFeatures_ReturnsAllFeatures()
    {
        // Arrange - k is larger than number of features
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });
        var target = new Vector<double>(new double[] { 0, 1, 2 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            k: 100 // Much larger than 2 features
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Should return all 2 features
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    #endregion

    #region Accuracy Verification Tests

    [Fact]
    public void VarianceThresholdFeatureSelector_CalculatesCorrectVariance()
    {
        // Arrange - Known variance: [0, 1, 2] has variance = (0-1)^2 + (1-1)^2 + (2-1)^2 / (3-1) = 1
        // Using sample variance (N-1 denominator)
        var data = CreateTestMatrix(new double[,]
        {
            { 0.0, 10.0 },
            { 1.0, 10.0 },
            { 2.0, 10.0 }
        });

        // Column 0: sample variance = 1
        // Column 1: variance = 0 (constant)
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.5);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Only column 0 should remain (variance = 1 > 0.5)
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(1.0, result[1, 0], Tolerance);
        Assert.Equal(2.0, result[2, 0], Tolerance);
    }

    [Fact]
    public void CorrelationFeatureSelector_CalculatesCorrectCorrelation()
    {
        // Arrange - Perfect positive correlation: y = x
        // Pearson correlation should be 1.0
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 1.0, 5.0 },
            { 2.0, 2.0, 10.0 },
            { 3.0, 3.0, 15.0 },
            { 4.0, 4.0, 20.0 }
        });

        // Col 0 and Col 1 are perfectly correlated (r = 1.0)
        // Col 2 is also perfectly correlated with Col 0 and Col 1
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.99);

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Only first feature should remain
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void UnivariateFeatureSelector_FValue_RanksCorrectly()
    {
        // Arrange - Feature 0 perfectly separates classes, feature 1 doesn't
        var data = CreateTestMatrix(new double[,]
        {
            { 0.0, 5.0 },
            { 0.0, 5.0 },
            { 10.0, 5.0 },
            { 10.0, 5.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.FValue,
            k: 1
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Feature 0 should be selected (perfectly separates classes)
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
        // The selected column should have values 0, 0, 10, 10
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(10.0, result[2, 0], Tolerance);
    }

    #endregion
}
