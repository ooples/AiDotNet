using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.FeatureSelectors;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
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
    public void VarianceThresholdFeatureSelector_AllConstant_ReturnsOneFeature()
    {
        // Arrange - All columns are constant (variance = 0 for both)
        var data = CreateTestMatrix(new double[,]
        {
            { 5.0, 3.0 },
            { 5.0, 3.0 },
            { 5.0, 3.0 }
        });
        // Use small positive threshold to require some variance (removes zero-variance features)
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1e-10);

        // Act - When no features pass threshold, selector keeps the one with highest variance (first one when tied)
        var result = selector.SelectFeatures(data);

        // Assert - At least one feature is always kept (safety behavior)
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
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

    #region RecursiveFeatureElimination (RFE) Tests

    [Fact]
    public void RecursiveFeatureElimination_SelectsTopFeatures()
    {
        // Arrange - Use model that assigns different weights to features
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 0.1, 0.01 },
            { 2.0, 0.2, 0.02 },
            { 3.0, 0.3, 0.03 },
            { 4.0, 0.4, 0.04 }
        });

        var model = new RFEMockModel(featureCount: 3);
        var selector = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
            model,
            createDummyTarget: numSamples => new Vector<double>(numSamples),
            numFeaturesToSelect: 2
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Should select 2 features
        Assert.Equal(4, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void RecursiveFeatureElimination_DefaultNumFeatures_SelectsHalf()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        });

        var model = new RFEMockModel(featureCount: 4);
        var selector = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
            model,
            createDummyTarget: numSamples => new Vector<double>(numSamples)
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Default should select 50% = 2 features
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void RecursiveFeatureElimination_SingleFeature_SelectsOne()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        var model = new RFEMockModel(featureCount: 3);
        var selector = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
            model,
            createDummyTarget: numSamples => new Vector<double>(numSamples),
            numFeaturesToSelect: 1
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void RecursiveFeatureElimination_NumFeaturesGreaterThanTotal_SelectsAll()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        var model = new RFEMockModel(featureCount: 2);
        var selector = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
            model,
            createDummyTarget: numSamples => new Vector<double>(numSamples),
            numFeaturesToSelect: 100
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Should select all features (2)
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    #endregion

    #region SelectFromModel Tests

    [Fact]
    public void SelectFromModel_MeanThreshold_SelectsAboveMean()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        });

        // Feature importances: 0.1, 0.2, 0.3, 0.4 (mean = 0.25)
        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.1 },
            { "Feature_1", 0.2 },
            { "Feature_2", 0.3 },
            { "Feature_3", 0.4 }
        };

        var model = new MockFeatureImportanceModel(importances);
        var selector = new SelectFromModel<double, Matrix<double>>(
            model,
            ImportanceThresholdStrategy.Mean
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Features 2 and 3 should be selected (>= mean of 0.25)
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void SelectFromModel_MedianThreshold_SelectsAboveMedian()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 }
        });

        // Feature importances: 0.1, 0.2, 0.3, 0.4 (median = 0.25)
        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.1 },
            { "Feature_1", 0.2 },
            { "Feature_2", 0.3 },
            { "Feature_3", 0.4 }
        };

        var model = new MockFeatureImportanceModel(importances);
        var selector = new SelectFromModel<double, Matrix<double>>(
            model,
            ImportanceThresholdStrategy.Median
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Top 50% features should be selected
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void SelectFromModel_CustomThreshold_SelectsAboveThreshold()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 }
        });

        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.1 },
            { "Feature_1", 0.2 },
            { "Feature_2", 0.3 },
            { "Feature_3", 0.4 }
        };

        var model = new MockFeatureImportanceModel(importances);
        var selector = new SelectFromModel<double, Matrix<double>>(
            model,
            threshold: 0.35
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Only feature 3 should be selected (>= 0.35)
        Assert.Equal(2, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void SelectFromModel_TopK_SelectsExactlyKFeatures()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0, 5.0 },
            { 6.0, 7.0, 8.0, 9.0, 10.0 }
        });

        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.1 },
            { "Feature_1", 0.5 },
            { "Feature_2", 0.2 },
            { "Feature_3", 0.4 },
            { "Feature_4", 0.3 }
        };

        var model = new MockFeatureImportanceModel(importances);
        var selector = new SelectFromModel<double, Matrix<double>>(
            model,
            k: 3
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Top 3 features by importance
        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void SelectFromModel_MaxFeaturesLimit_LimitsSelection()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 }
        });

        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.1 },
            { "Feature_1", 0.2 },
            { "Feature_2", 0.3 },
            { "Feature_3", 0.4 }
        };

        var model = new MockFeatureImportanceModel(importances);
        var selector = new SelectFromModel<double, Matrix<double>>(
            model,
            ImportanceThresholdStrategy.Mean,
            maxFeatures: 1
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Limited to 1 feature
        Assert.Equal(2, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void SelectFromModel_NoFeaturesAboveThreshold_SelectsBest()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.1 },
            { "Feature_1", 0.2 },
            { "Feature_2", 0.15 }
        };

        var model = new MockFeatureImportanceModel(importances);
        var selector = new SelectFromModel<double, Matrix<double>>(
            model,
            threshold: 1.0 // Very high threshold
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - At least one feature selected (safety behavior)
        Assert.Equal(2, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void SelectFromModel_ZeroImportances_SelectsAllFeatures()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.0 },
            { "Feature_1", 0.0 },
            { "Feature_2", 0.0 }
        };

        var model = new MockFeatureImportanceModel(importances);
        var selector = new SelectFromModel<double, Matrix<double>>(
            model,
            ImportanceThresholdStrategy.Mean
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - All features >= mean (0.0)
        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    #endregion

    #region SequentialFeatureSelector Tests

    [Fact]
    public void SequentialFeatureSelector_ForwardSelection_SelectsFeatures()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 0.1, 0.01 },
            { 2.0, 0.2, 0.02 },
            { 8.0, 3.0, 0.03 },
            { 9.0, 4.0, 0.04 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

        var model = new SequentialMockModel();
        var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
            model,
            target,
            CalculateAccuracy,
            SequentialFeatureSelectionDirection.Forward,
            numFeaturesToSelect: 2
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(4, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void SequentialFeatureSelector_BackwardElimination_SelectsFeatures()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 0.1, 0.01 },
            { 2.0, 0.2, 0.02 },
            { 8.0, 3.0, 0.03 },
            { 9.0, 4.0, 0.04 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

        var model = new SequentialMockModel();
        var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
            model,
            target,
            CalculateAccuracy,
            SequentialFeatureSelectionDirection.Backward,
            numFeaturesToSelect: 2
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(4, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void SequentialFeatureSelector_DefaultNumFeatures_SelectsHalf()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 0.1, 0.01, 0.001 },
            { 2.0, 0.2, 0.02, 0.002 },
            { 8.0, 3.0, 0.03, 0.003 },
            { 9.0, 4.0, 0.04, 0.004 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

        var model = new SequentialMockModel();
        var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
            model,
            target,
            CalculateAccuracy,
            SequentialFeatureSelectionDirection.Forward
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Default should select 50% = 2 features
        Assert.Equal(4, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void SequentialFeatureSelector_SingleFeature_SelectsOne()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 0.1, 0.01 },
            { 2.0, 0.2, 0.02 },
            { 8.0, 3.0, 0.03 },
            { 9.0, 4.0, 0.04 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

        var model = new SequentialMockModel();
        var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
            model,
            target,
            CalculateAccuracy,
            SequentialFeatureSelectionDirection.Forward,
            numFeaturesToSelect: 1
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void SequentialFeatureSelector_NumFeaturesGreaterThanTotal_SelectsAll()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 0.1 },
            { 2.0, 0.2 },
            { 8.0, 3.0 },
            { 9.0, 4.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

        var model = new SequentialMockModel();
        var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
            model,
            target,
            CalculateAccuracy,
            SequentialFeatureSelectionDirection.Forward,
            numFeaturesToSelect: 100
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - All features
        Assert.Equal(4, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    /// <summary>
    /// Simple accuracy scorer for testing.
    /// </summary>
    private static double CalculateAccuracy(Vector<double> predictions, Vector<double> actual)
    {
        int correct = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            if (Math.Abs(predictions[i] - actual[i]) < 0.5)
            {
                correct++;
            }
        }
        return (double)correct / predictions.Length;
    }

    #endregion

    #region UnivariateFeatureSelector Chi-Squared Tests

    [Fact]
    public void UnivariateFeatureSelector_ChiSquared_SelectsFeatures()
    {
        // Arrange - Feature 0 is correlated with target (categorical), feature 1 is not
        var data = CreateTestMatrix(new double[,]
        {
            { 0.0, 5.0 },
            { 0.0, 6.0 },
            { 1.0, 5.0 },
            { 1.0, 6.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.ChiSquared,
            k: 1
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Should select 1 feature
        Assert.Equal(4, result.Rows);
        Assert.Equal(1, result.Columns);
    }

    [Fact]
    public void UnivariateFeatureSelector_ChiSquared_WithMultipleClasses()
    {
        // Arrange - 3 classes
        var data = CreateTestMatrix(new double[,]
        {
            { 0.0, 5.0, 1.0 },
            { 0.0, 6.0, 2.0 },
            { 1.0, 5.0, 1.0 },
            { 1.0, 6.0, 2.0 },
            { 2.0, 5.0, 1.0 },
            { 2.0, 6.0, 2.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1, 2, 2 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.ChiSquared,
            k: 2
        );

        // Act
        var result = selector.SelectFeatures(data);

        // Assert - Should select 2 features
        Assert.Equal(6, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void UnivariateFeatureSelector_ChiSquared_AllSameClass_HandlesGracefully()
    {
        // Arrange - All samples have the same class
        var data = CreateTestMatrix(new double[,]
        {
            { 0.0, 5.0 },
            { 1.0, 6.0 },
            { 2.0, 7.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 0 }); // Single class
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
            target: target,
            scoringFunction: UnivariateScoringFunction.ChiSquared,
            k: 1
        );

        // Act - Should not crash
        var result = selector.SelectFeatures(data);

        // Assert - Should still return at least 1 feature
        Assert.Equal(3, result.Rows);
        Assert.True(result.Columns >= 1);
    }

    #endregion

    #region Consistency and Reproducibility Tests

    [Fact]
    public void VarianceThresholdFeatureSelector_MultipleCallsProduceSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0, 2.0 },
            { 2.0, 5.0, 3.0 },
            { 3.0, 5.0, 4.0 },
            { 4.0, 5.0, 5.0 }
        });
        var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.5);

        // Act
        var result1 = selector.SelectFeatures(data);
        var result2 = selector.SelectFeatures(data);

        // Assert - results should be identical
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j]);
            }
        }
    }

    [Fact]
    public void CorrelationFeatureSelector_MultipleCallsProduceSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 10.0 },
            { 2.0, 4.0, 20.0 },
            { 3.0, 6.0, 30.0 },
            { 4.0, 8.0, 40.0 }
        });
        var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);

        // Act
        var result1 = selector.SelectFeatures(data);
        var result2 = selector.SelectFeatures(data);

        // Assert - results should be identical
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j]);
            }
        }
    }

    [Fact]
    public void UnivariateFeatureSelector_MultipleCallsProduceSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0, 2.0, 8.0 },
            { 2.0, 5.0, 3.0, 7.0 },
            { 3.0, 5.0, 4.0, 6.0 },
            { 4.0, 5.0, 5.0, 5.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });
        var selector = new UnivariateFeatureSelector<double, Matrix<double>>(target, UnivariateScoringFunction.FValue, k: 2);

        // Act
        var result1 = selector.SelectFeatures(data);
        var result2 = selector.SelectFeatures(data);

        // Assert - results should be identical
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j]);
            }
        }
    }

    [Fact]
    public void VarianceThresholdFeatureSelector_NewInstanceWithSameConfigProducesSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0, 2.0 },
            { 2.0, 5.0, 3.0 },
            { 3.0, 5.0, 4.0 },
            { 4.0, 5.0, 5.0 }
        });
        var selector1 = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.5);
        var selector2 = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.5);

        // Act
        var result1 = selector1.SelectFeatures(data);
        var result2 = selector2.SelectFeatures(data);

        // Assert - different instances with same config produce identical results
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j]);
            }
        }
    }

    [Fact]
    public void CorrelationFeatureSelector_NewInstanceWithSameConfigProducesSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 10.0 },
            { 2.0, 4.0, 20.0 },
            { 3.0, 6.0, 30.0 },
            { 4.0, 8.0, 40.0 }
        });
        var selector1 = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);
        var selector2 = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);

        // Act
        var result1 = selector1.SelectFeatures(data);
        var result2 = selector2.SelectFeatures(data);

        // Assert - different instances with same config produce identical results
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j]);
            }
        }
    }

    [Fact]
    public void UnivariateFeatureSelector_NewInstanceWithSameConfigProducesSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 5.0, 2.0, 8.0 },
            { 2.0, 5.0, 3.0, 7.0 },
            { 3.0, 5.0, 4.0, 6.0 },
            { 4.0, 5.0, 5.0, 5.0 }
        });
        var target = new Vector<double>(new double[] { 0, 0, 1, 1 });
        var selector1 = new UnivariateFeatureSelector<double, Matrix<double>>(target, UnivariateScoringFunction.FValue, k: 2);
        var selector2 = new UnivariateFeatureSelector<double, Matrix<double>>(target, UnivariateScoringFunction.FValue, k: 2);

        // Act
        var result1 = selector1.SelectFeatures(data);
        var result2 = selector2.SelectFeatures(data);

        // Assert - different instances with same config produce identical results
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j]);
            }
        }
    }

    [Fact]
    public void RFE_NewInstanceWithSameConfigProducesSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 2.0, 3.0, 4.0, 5.0 },
            { 3.0, 4.0, 5.0, 6.0 },
            { 4.0, 5.0, 6.0, 7.0 }
        });
        var model1 = new RFEMockModel(featureCount: 4);
        var model2 = new RFEMockModel(featureCount: 4);
        var selector1 = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
            model1,
            createDummyTarget: numSamples => new Vector<double>(numSamples),
            numFeaturesToSelect: 2
        );
        var selector2 = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
            model2,
            createDummyTarget: numSamples => new Vector<double>(numSamples),
            numFeaturesToSelect: 2
        );

        // Act
        var result1 = selector1.SelectFeatures(data);
        var result2 = selector2.SelectFeatures(data);

        // Assert - same number of features selected
        Assert.Equal(result1.Columns, result2.Columns);
    }

    [Fact]
    public void SelectFromModel_NewInstanceWithSameConfigProducesSameResult()
    {
        // Arrange
        var data = CreateTestMatrix(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 2.0, 3.0, 4.0, 5.0 },
            { 3.0, 4.0, 5.0, 6.0 },
            { 4.0, 5.0, 6.0, 7.0 }
        });
        var importances = new Dictionary<string, double>
        {
            { "Feature_0", 0.1 },
            { "Feature_1", 0.5 },
            { "Feature_2", 0.3 },
            { "Feature_3", 0.8 }
        };
        var model1 = new MockFeatureImportanceModel(importances);
        var model2 = new MockFeatureImportanceModel(importances);
        var selector1 = new SelectFromModel<double, Matrix<double>>(model1, ImportanceThresholdStrategy.Mean);
        var selector2 = new SelectFromModel<double, Matrix<double>>(model2, ImportanceThresholdStrategy.Mean);

        // Act
        var result1 = selector1.SelectFeatures(data);
        var result2 = selector2.SelectFeatures(data);

        // Assert - same features selected
        Assert.Equal(result1.Rows, result2.Rows);
        Assert.Equal(result1.Columns, result2.Columns);
        for (int i = 0; i < result1.Rows; i++)
        {
            for (int j = 0; j < result1.Columns; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j]);
            }
        }
    }

    #endregion
}

#region Mock Models for Feature Selector Tests

/// <summary>
/// Mock model for RFE testing that returns predictable parameters.
/// </summary>
internal class RFEMockModel : IFullModel<double, Matrix<double>, Vector<double>>
{
    private int _currentFeatureCount;
    private Vector<double> _parameters;

    public RFEMockModel(int featureCount)
    {
        _currentFeatureCount = featureCount;
        _parameters = CreateParameters(featureCount);
    }

    private static Vector<double> CreateParameters(int count)
    {
        // Parameters decrease in importance: [1.0, 0.5, 0.25, 0.125, ...]
        var parameters = new Vector<double>(count);
        for (int i = 0; i < count; i++)
        {
            parameters[i] = 1.0 / Math.Pow(2, i);
        }
        return parameters;
    }

    public void Train(Matrix<double> input, Vector<double> expectedOutput)
    {
        // Update current feature count based on input
        _currentFeatureCount = input.Columns;
        _parameters = CreateParameters(_currentFeatureCount);
    }

    public Vector<double> Predict(Matrix<double> input)
    {
        var predictions = new Vector<double>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = 0;
        }
        return predictions;
    }

    public Vector<double> GetParameters() => _parameters.Clone();
    public void SetParameters(Vector<double> parameters) => _parameters = parameters.Clone();
    public int ParameterCount => _parameters.Length;

    public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters)
    {
        var newModel = new RFEMockModel(parameters.Length);
        newModel.SetParameters(parameters);
        return newModel;
    }

    public ModelMetadata<double> GetModelMetadata() => new() { Name = "RFEMockModel" };
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }
    public void SaveState(Stream stream) { }
    public void LoadState(Stream stream) { }
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _parameters.Length);
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => featureIndex >= 0 && featureIndex < _parameters.Length;
    public Dictionary<string, double> GetFeatureImportance() => new();
    public IFullModel<double, Matrix<double>, Vector<double>> Clone() => new RFEMockModel(_currentFeatureCount);
    public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy() => Clone();
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
    public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(ParameterCount);
    public void ApplyGradients(Vector<double> gradients, double learningRate) { }
    public bool SupportsJitCompilation => false;
    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes) => throw new NotSupportedException();
}

/// <summary>
/// Mock model that provides feature importances for SelectFromModel testing.
/// </summary>
internal class MockFeatureImportanceModel : IFeatureImportance<double>
{
    private readonly Dictionary<string, double> _importances;

    public MockFeatureImportanceModel(Dictionary<string, double> importances)
    {
        _importances = importances;
    }

    public Dictionary<string, double> GetFeatureImportance() => _importances;
}

/// <summary>
/// Mock model for SequentialFeatureSelector testing.
/// </summary>
internal class SequentialMockModel : IFullModel<double, Matrix<double>, Vector<double>>
{
    public void Train(Matrix<double> input, Vector<double> expectedOutput) { }

    public Vector<double> Predict(Matrix<double> input)
    {
        // Simple prediction based on sum of features
        var predictions = new Vector<double>(input.Rows);
        for (int i = 0; i < input.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < input.Columns; j++)
            {
                sum += input[i, j];
            }
            predictions[i] = sum > 5 ? 1.0 : 0.0;
        }
        return predictions;
    }

    public Vector<double> GetParameters() => new Vector<double>(0);
    public void SetParameters(Vector<double> parameters) { }
    public int ParameterCount => 0;
    public IFullModel<double, Matrix<double>, Vector<double>> WithParameters(Vector<double> parameters) => new SequentialMockModel();
    public ModelMetadata<double> GetModelMetadata() => new() { Name = "SequentialMockModel" };
    public byte[] Serialize() => Array.Empty<byte>();
    public void Deserialize(byte[] data) { }
    public void SaveModel(string filePath) { }
    public void LoadModel(string filePath) { }
    public void SaveState(Stream stream) { }
    public void LoadState(Stream stream) { }
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();
    public void SetActiveFeatureIndices(IEnumerable<int> indices) { }
    public bool IsFeatureUsed(int featureIndex) => true;
    public Dictionary<string, double> GetFeatureImportance() => new();
    public IFullModel<double, Matrix<double>, Vector<double>> Clone() => new SequentialMockModel();
    public IFullModel<double, Matrix<double>, Vector<double>> DeepCopy() => new SequentialMockModel();
    public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
    public Vector<double> ComputeGradients(Matrix<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(ParameterCount);
    public void ApplyGradients(Vector<double> gradients, double learningRate) { }
    public bool SupportsJitCompilation => false;
    public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes) => throw new NotSupportedException();
}

#endregion
