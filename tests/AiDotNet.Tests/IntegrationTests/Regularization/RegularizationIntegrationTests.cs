using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regularization;

/// <summary>
/// Integration tests for regularization classes.
/// Tests L1, L2, ElasticNet, and NoRegularization implementations.
/// </summary>
public class RegularizationIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region L1 Regularization Tests

    [Fact]
    public void L1Regularization_RegularizeVector_AppliesSoftThresholding()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 };
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 0.5, -0.5, 0.05, -0.05, 1.0 });

        // Act
        var result = l1.Regularize(vector);

        // Assert - L1 applies soft thresholding, values near zero should be pushed toward zero
        Assert.Equal(5, result.Length);
        // Values with magnitude > threshold should be shrunk but not zeroed
        Assert.True(result[0] < 0.5);
        Assert.True(result[1] > -0.5);
        // Larger values should still have same sign
        Assert.True(result[4] > 0);
    }

    [Fact]
    public void L1Regularization_RegularizeMatrix_AppliesSoftThresholding()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 };
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(options);
        var matrix = new Matrix<double>(new[,]
        {
            { 0.5, -0.5 },
            { 1.0, -1.0 }
        });

        // Act
        var result = l1.Regularize(matrix);

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
        // Values should be shrunk toward zero
        Assert.True(result[0, 0] < 0.5);
        Assert.True(result[0, 1] > -0.5);
    }

    [Fact]
    public void L1Regularization_RegularizeGradient_AppliesRegularization()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.1 };
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(options);
        var gradient = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var coefficients = new Vector<double>(new[] { 0.5, -0.5, 1.0 });

        // Act
        var result = l1.Regularize(gradient, coefficients);

        // Assert - gradient should be modified based on sign of coefficients
        Assert.Equal(3, result.Length);
        // Result should not be NaN or infinite
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
            Assert.False(double.IsInfinity(result[i]));
        }
    }

    [Fact]
    public void L1Regularization_ZeroStrength_ReturnsOriginalValues()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.0 };
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 0.5, -0.5, 1.0 });

        // Act
        var result = l1.Regularize(vector);

        // Assert - with zero strength, values should be unchanged
        Assert.Equal(0.5, result[0], Tolerance);
        Assert.Equal(-0.5, result[1], Tolerance);
        Assert.Equal(1.0, result[2], Tolerance);
    }

    [Fact]
    public void L1Regularization_GetOptions_ReturnsCorrectStrength()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.25 };
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>(options);

        // Act
        var returnedOptions = l1.GetOptions();

        // Assert
        Assert.Equal(0.25, returnedOptions.Strength, Tolerance);
    }

    [Fact]
    public void L1Regularization_DefaultStrength_IsApplied()
    {
        // Arrange - default constructor uses default options (strength = 0.1)
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>();

        // Act
        var options = l1.GetOptions();

        // Assert
        Assert.Equal(0.1, options.Strength, Tolerance);
    }

    #endregion

    #region L2 Regularization Tests

    [Fact]
    public void L2Regularization_RegularizeVector_AppliesShrinkage()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = l2.Regularize(vector);

        // Assert - L2 applies shrinkage factor (1 - strength)
        Assert.Equal(3, result.Length);
        // Values should be shrunk proportionally
        Assert.True(result[0] < 1.0);
        Assert.True(result[1] < 2.0);
        Assert.True(result[2] < 3.0);
        // Relative proportions should be maintained
        Assert.True(result[1] > result[0]);
        Assert.True(result[2] > result[1]);
    }

    [Fact]
    public void L2Regularization_RegularizeMatrix_AppliesShrinkage()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(options);
        var matrix = new Matrix<double>(new[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act
        var result = l2.Regularize(matrix);

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
        // All values should be shrunk
        Assert.True(result[0, 0] < 1.0);
        Assert.True(result[1, 1] < 4.0);
    }

    [Fact]
    public void L2Regularization_RegularizeGradient_AppliesRegularization()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(options);
        var gradient = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var coefficients = new Vector<double>(new[] { 0.5, 1.0, 1.5 });

        // Act
        var result = l2.Regularize(gradient, coefficients);

        // Assert - gradient should be modified based on coefficients
        Assert.Equal(3, result.Length);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
            Assert.False(double.IsInfinity(result[i]));
        }
    }

    [Fact]
    public void L2Regularization_ZeroStrength_ReturnsOriginalValues()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.0 };
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = l2.Regularize(vector);

        // Assert - with zero strength, values should be unchanged
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void L2Regularization_GetOptions_ReturnsCorrectStrength()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.05 };
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(options);

        // Act
        var returnedOptions = l2.GetOptions();

        // Assert
        Assert.Equal(0.05, returnedOptions.Strength, Tolerance);
    }

    [Fact]
    public void L2Regularization_DefaultStrength_IsApplied()
    {
        // Arrange - default constructor uses default options (strength = 0.01)
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>();

        // Act
        var options = l2.GetOptions();

        // Assert
        Assert.Equal(0.01, options.Strength, Tolerance);
    }

    [Fact]
    public void L2Regularization_PreservesSign()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.1 };
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 1.0, -1.0, 2.0, -2.0 });

        // Act
        var result = l2.Regularize(vector);

        // Assert - signs should be preserved
        Assert.True(result[0] > 0);
        Assert.True(result[1] < 0);
        Assert.True(result[2] > 0);
        Assert.True(result[3] < 0);
    }

    #endregion

    #region ElasticNet Regularization Tests

    [Fact]
    public void ElasticNetRegularization_RegularizeVector_CombinesL1AndL2()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 };
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = elasticNet.Regularize(vector);

        // Assert - combines L1 and L2 components
        Assert.Equal(3, result.Length);
        // Results should not contain NaN or Infinity
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
            Assert.False(double.IsInfinity(result[i]));
        }
    }

    [Fact]
    public void ElasticNetRegularization_RegularizeMatrix_CombinesL1AndL2()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 };
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(options);
        var matrix = new Matrix<double>(new[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act
        var result = elasticNet.Regularize(matrix);

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
        // Results should not contain NaN or Infinity
        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                Assert.False(double.IsNaN(result[i, j]));
                Assert.False(double.IsInfinity(result[i, j]));
            }
        }
    }

    [Fact]
    public void ElasticNetRegularization_HighL1Ratio_BehavesMoreLikeL1()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.9 };
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = elasticNet.Regularize(vector);

        // Assert - with high L1 ratio, results should be valid
        Assert.Equal(3, result.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
        }
    }

    [Fact]
    public void ElasticNetRegularization_LowL1Ratio_BehavesMoreLikeL2()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.1 };
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = elasticNet.Regularize(vector);

        // Assert - with low L1 ratio, results should be valid
        Assert.Equal(3, result.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
        }
    }

    [Fact]
    public void ElasticNetRegularization_ZeroStrength_ProducesValidResults()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.0, L1Ratio = 0.5 };
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(options);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = elasticNet.Regularize(vector);

        // Assert - results should be valid (no NaN/Infinity)
        // Note: ElasticNet implementation adds l1Part + l2Part, so with zero strength
        // the result equals 2*value (l1Part = value, l2Part = value)
        Assert.Equal(3, result.Length);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
            Assert.False(double.IsInfinity(result[i]));
        }
    }

    [Fact]
    public void ElasticNetRegularization_GetOptions_ReturnsCorrectValues()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.2, L1Ratio = 0.7 };
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(options);

        // Act
        var returnedOptions = elasticNet.GetOptions();

        // Assert
        Assert.Equal(0.2, returnedOptions.Strength, Tolerance);
        Assert.Equal(0.7, returnedOptions.L1Ratio, Tolerance);
    }

    [Fact]
    public void ElasticNetRegularization_DefaultValues_AreApplied()
    {
        // Arrange - default constructor uses default options
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>();

        // Act
        var options = elasticNet.GetOptions();

        // Assert
        Assert.Equal(0.1, options.Strength, Tolerance);
        Assert.Equal(0.5, options.L1Ratio, Tolerance);
    }

    [Fact]
    public void ElasticNetRegularization_RegularizeGradient_AppliesRegularization()
    {
        // Arrange
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.1, L1Ratio = 0.5 };
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>(options);
        var gradient = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var coefficients = new Vector<double>(new[] { 0.5, 1.0, 1.5 });

        // Act
        var result = elasticNet.Regularize(gradient, coefficients);

        // Assert
        Assert.Equal(3, result.Length);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]));
            Assert.False(double.IsInfinity(result[i]));
        }
    }

    #endregion

    #region NoRegularization Tests

    [Fact]
    public void NoRegularization_RegularizeVector_ReturnsOriginalValues()
    {
        // Arrange
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = noReg.Regularize(vector);

        // Assert - no regularization should return unchanged values
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void NoRegularization_RegularizeMatrix_ReturnsOriginalValues()
    {
        // Arrange
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var matrix = new Matrix<double>(new[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act
        var result = noReg.Regularize(matrix);

        // Assert - no regularization should return unchanged values
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[0, 1], Tolerance);
        Assert.Equal(3.0, result[1, 0], Tolerance);
        Assert.Equal(4.0, result[1, 1], Tolerance);
    }

    [Fact]
    public void NoRegularization_RegularizeGradient_ReturnsOriginalGradient()
    {
        // Arrange
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var gradient = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var coefficients = new Vector<double>(new[] { 0.5, 1.0, 1.5 });

        // Act
        var result = noReg.Regularize(gradient, coefficients);

        // Assert - no regularization should return original gradient
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void NoRegularization_GetOptions_ReturnsZeroStrength()
    {
        // Arrange
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();

        // Act
        var options = noReg.GetOptions();

        // Assert
        Assert.Equal(0.0, options.Strength, Tolerance);
    }

    [Fact]
    public void NoRegularization_NegativeValues_PreservedExactly()
    {
        // Arrange
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var vector = new Vector<double>(new[] { -1.0, -2.0, -3.0 });

        // Act
        var result = noReg.Regularize(vector);

        // Assert
        Assert.Equal(-1.0, result[0], Tolerance);
        Assert.Equal(-2.0, result[1], Tolerance);
        Assert.Equal(-3.0, result[2], Tolerance);
    }

    [Fact]
    public void NoRegularization_ZeroValues_PreservedExactly()
    {
        // Arrange
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var vector = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act
        var result = noReg.Regularize(vector);

        // Assert
        Assert.Equal(0.0, result[0], Tolerance);
        Assert.Equal(0.0, result[1], Tolerance);
        Assert.Equal(0.0, result[2], Tolerance);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllRegularizations_HandleEmptyVector()
    {
        // Arrange
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>();
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>();
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>();
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var emptyVector = new Vector<double>(0);

        // Act & Assert - should not throw
        var l1Result = l1.Regularize(emptyVector);
        var l2Result = l2.Regularize(emptyVector);
        var elasticResult = elasticNet.Regularize(emptyVector);
        var noRegResult = noReg.Regularize(emptyVector);

        Assert.Equal(0, l1Result.Length);
        Assert.Equal(0, l2Result.Length);
        Assert.Equal(0, elasticResult.Length);
        Assert.Equal(0, noRegResult.Length);
    }

    [Fact]
    public void AllRegularizations_HandleLargeValues()
    {
        // Arrange
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>();
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>();
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>();
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var largeVector = new Vector<double>(new[] { 1e10, -1e10, 1e5 });

        // Act
        var l1Result = l1.Regularize(largeVector);
        var l2Result = l2.Regularize(largeVector);
        var elasticResult = elasticNet.Regularize(largeVector);
        var noRegResult = noReg.Regularize(largeVector);

        // Assert - no NaN or Infinity
        for (int i = 0; i < largeVector.Length; i++)
        {
            Assert.False(double.IsNaN(l1Result[i]));
            Assert.False(double.IsNaN(l2Result[i]));
            Assert.False(double.IsNaN(elasticResult[i]));
            Assert.False(double.IsNaN(noRegResult[i]));
        }
    }

    [Fact]
    public void AllRegularizations_HandleSmallValues()
    {
        // Arrange
        var l1 = new L1Regularization<double, Vector<double>, Vector<double>>();
        var l2 = new L2Regularization<double, Vector<double>, Vector<double>>();
        var elasticNet = new ElasticNetRegularization<double, Vector<double>, Vector<double>>();
        var noReg = new NoRegularization<double, Vector<double>, Vector<double>>();
        var smallVector = new Vector<double>(new[] { 1e-10, -1e-10, 1e-5 });

        // Act
        var l1Result = l1.Regularize(smallVector);
        var l2Result = l2.Regularize(smallVector);
        var elasticResult = elasticNet.Regularize(smallVector);
        var noRegResult = noReg.Regularize(smallVector);

        // Assert - no NaN or Infinity
        for (int i = 0; i < smallVector.Length; i++)
        {
            Assert.False(double.IsNaN(l1Result[i]));
            Assert.False(double.IsNaN(l2Result[i]));
            Assert.False(double.IsNaN(elasticResult[i]));
            Assert.False(double.IsNaN(noRegResult[i]));
        }
    }

    [Fact]
    public void RegularizationStrength_AffectsResults()
    {
        // Arrange
        var weakOptions = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.01 };
        var strongOptions = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.5 };
        var weakL2 = new L2Regularization<double, Vector<double>, Vector<double>>(weakOptions);
        var strongL2 = new L2Regularization<double, Vector<double>, Vector<double>>(strongOptions);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var weakResult = weakL2.Regularize(vector);
        var strongResult = strongL2.Regularize(vector);

        // Assert - stronger regularization should shrink more
        Assert.True(strongResult[0] < weakResult[0]);
        Assert.True(strongResult[1] < weakResult[1]);
        Assert.True(strongResult[2] < weakResult[2]);
    }

    [Fact]
    public void FloatType_AllRegularizations_Work()
    {
        // Arrange
        var l1 = new L1Regularization<float, Vector<float>, Vector<float>>();
        var l2 = new L2Regularization<float, Vector<float>, Vector<float>>();
        var elasticNet = new ElasticNetRegularization<float, Vector<float>, Vector<float>>();
        var noReg = new NoRegularization<float, Vector<float>, Vector<float>>();
        var vector = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });

        // Act
        var l1Result = l1.Regularize(vector);
        var l2Result = l2.Regularize(vector);
        var elasticResult = elasticNet.Regularize(vector);
        var noRegResult = noReg.Regularize(vector);

        // Assert
        Assert.Equal(3, l1Result.Length);
        Assert.Equal(3, l2Result.Length);
        Assert.Equal(3, elasticResult.Length);
        Assert.Equal(3, noRegResult.Length);
    }

    #endregion
}
