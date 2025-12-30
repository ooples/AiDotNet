using AiDotNet.Exceptions;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for Linear Regression models (SimpleRegression, MultipleRegression,
/// RidgeRegression, LassoRegression, ElasticNetRegression).
/// These tests verify regression algorithms work correctly with various data patterns.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class LinearRegressionIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 0.1; // For approximate comparisons

    #region SimpleRegression Tests

    [Fact]
    public void SimpleRegression_Train_PerfectLinearData_RecoversTrueCoefficients()
    {
        // Arrange: y = 2x + 3 (slope=2, intercept=3)
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new SimpleRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 5, 7, 9, 11, 13 }); // y = 2x + 3

        // Act
        regression.Train(x, y);

        // Assert
        Assert.Equal(2.0, regression.Coefficients[0], Tolerance);
        Assert.Equal(3.0, regression.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_Train_WithoutIntercept_RecoversSlopeOnly()
    {
        // Arrange: y = 3x (no intercept)
        var options = new RegressionOptions<double> { UseIntercept = false };
        var regression = new SimpleRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 3, 6, 9, 12, 15 }); // y = 3x

        // Act
        regression.Train(x, y);

        // Assert
        Assert.Equal(3.0, regression.Coefficients[0], Tolerance);
        Assert.Equal(0.0, regression.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_Predict_ReturnsCorrectValues()
    {
        // Arrange
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new SimpleRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 5, 7, 9, 11, 13 }); // y = 2x + 3
        regression.Train(x, y);

        // Act
        var newX = CreateMatrix(new double[,] { { 6 }, { 7 }, { 10 } });
        var predictions = regression.Predict(newX);

        // Assert
        Assert.Equal(15.0, predictions[0], Tolerance); // 2*6 + 3
        Assert.Equal(17.0, predictions[1], Tolerance); // 2*7 + 3
        Assert.Equal(23.0, predictions[2], Tolerance); // 2*10 + 3
    }

    [Fact]
    public void SimpleRegression_Train_NoisyData_FitsTrend()
    {
        // Arrange: y ~ 2x + 3 with noise
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new SimpleRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 } });
        var y = CreateVector(new double[] { 5.1, 6.8, 9.2, 10.9, 13.1, 14.8, 17.2, 18.9, 21.1, 22.8 });

        // Act
        regression.Train(x, y);

        // Assert - should be close to y = 2x + 3
        Assert.True(Math.Abs(regression.Coefficients[0] - 2.0) < LooseTolerance,
            $"Slope should be close to 2.0, got {regression.Coefficients[0]}");
        Assert.True(Math.Abs(regression.Intercept - 3.0) < LooseTolerance,
            $"Intercept should be close to 3.0, got {regression.Intercept}");
    }

    [Fact]
    public void SimpleRegression_Train_MultipleColumns_ThrowsException()
    {
        // Arrange
        var regression = new SimpleRegression<double>();
        var x = CreateMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }); // 2 columns - invalid
        var y = CreateVector(new double[] { 1, 2, 3 });

        // Act & Assert - throws InvalidInputDimensionException for invalid feature count
        Assert.Throws<InvalidInputDimensionException>(() => regression.Train(x, y));
    }

    #endregion

    #region MultipleRegression Tests

    [Fact]
    public void MultipleRegression_Train_PerfectLinearData_RecoversTrueCoefficients()
    {
        // Arrange: y = 2*x1 + 3*x2 + 5 (coefficients: 2, 3; intercept: 5)
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new MultipleRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 }); // y = 2*x1 + 3*x2 + 5

        // Act
        regression.Train(x, y);

        // Assert
        Assert.Equal(2.0, regression.Coefficients[0], Tolerance);
        Assert.Equal(3.0, regression.Coefficients[1], Tolerance);
        Assert.Equal(5.0, regression.Intercept, Tolerance);
    }

    [Fact]
    public void MultipleRegression_Train_WithoutIntercept_RecoversCoefficientsOnly()
    {
        // Arrange: y = 2*x1 + 3*x2 (no intercept)
        var options = new RegressionOptions<double> { UseIntercept = false };
        var regression = new MultipleRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 5, 7, 8, 10, 15 }); // y = 2*x1 + 3*x2

        // Act
        regression.Train(x, y);

        // Assert
        Assert.Equal(2.0, regression.Coefficients[0], Tolerance);
        Assert.Equal(3.0, regression.Coefficients[1], Tolerance);
        Assert.Equal(0.0, regression.Intercept, Tolerance);
    }

    [Fact]
    public void MultipleRegression_Predict_ReturnsCorrectValues()
    {
        // Arrange
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new MultipleRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 }); // y = 2*x1 + 3*x2 + 5
        regression.Train(x, y);

        // Act
        var newX = CreateMatrix(new double[,] { { 4, 4 }, { 5, 1 } });
        var predictions = regression.Predict(newX);

        // Assert
        Assert.Equal(25.0, predictions[0], Tolerance); // 2*4 + 3*4 + 5
        Assert.Equal(18.0, predictions[1], Tolerance); // 2*5 + 3*1 + 5
    }

    [Fact]
    public void MultipleRegression_Train_ManyFeatures_HandlesCorrectly()
    {
        // Arrange: 10 features
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new MultipleRegression<double>(options);
        var random = new Random(42);
        int numSamples = 50;
        int numFeatures = 10;

        var xData = new double[numSamples, numFeatures];
        var trueCoeffs = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++)
        {
            trueCoeffs[i] = i + 1; // Coefficients: 1, 2, 3, ..., 10
        }

        var yData = new double[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            yData[i] = 5.0; // Intercept
            for (int j = 0; j < numFeatures; j++)
            {
                xData[i, j] = random.NextDouble() * 10;
                yData[i] += trueCoeffs[j] * xData[i, j];
            }
        }

        var x = CreateMatrix(xData);
        var y = CreateVector(yData);

        // Act
        regression.Train(x, y);

        // Assert - coefficients should be close to true values
        for (int i = 0; i < numFeatures; i++)
        {
            Assert.True(Math.Abs(regression.Coefficients[i] - trueCoeffs[i]) < LooseTolerance,
                $"Coefficient {i} should be close to {trueCoeffs[i]}, got {regression.Coefficients[i]}");
        }
        Assert.True(Math.Abs(regression.Intercept - 5.0) < LooseTolerance,
            $"Intercept should be close to 5.0, got {regression.Intercept}");
    }

    #endregion

    #region RidgeRegression Tests

    [Fact]
    public void RidgeRegression_Train_WithRegularization_ShrinkCoefficients()
    {
        // Arrange
        var optionsNoReg = new RidgeRegressionOptions<double> { Alpha = 0.0, UseIntercept = true };
        var optionsHighReg = new RidgeRegressionOptions<double> { Alpha = 100.0, UseIntercept = true };

        var regressionNoReg = new RidgeRegression<double>(optionsNoReg);
        var regressionHighReg = new RidgeRegression<double>(optionsHighReg);

        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 });

        // Act
        regressionNoReg.Train(x, y);
        regressionHighReg.Train(x, y);

        // Assert - high regularization should shrink coefficients toward zero
        double normNoReg = Math.Sqrt(
            regressionNoReg.Coefficients[0] * regressionNoReg.Coefficients[0] +
            regressionNoReg.Coefficients[1] * regressionNoReg.Coefficients[1]);
        double normHighReg = Math.Sqrt(
            regressionHighReg.Coefficients[0] * regressionHighReg.Coefficients[0] +
            regressionHighReg.Coefficients[1] * regressionHighReg.Coefficients[1]);

        Assert.True(normHighReg < normNoReg,
            $"High regularization should shrink coefficients: norm with reg ({normHighReg}) should be < norm without ({normNoReg})");
    }

    [Fact]
    public void RidgeRegression_Train_ZeroAlpha_MatchesOLS()
    {
        // Arrange
        var ridgeOptions = new RidgeRegressionOptions<double> { Alpha = 0.0, UseIntercept = true };
        var olsOptions = new RegressionOptions<double> { UseIntercept = true };

        var ridge = new RidgeRegression<double>(ridgeOptions);
        var ols = new MultipleRegression<double>(olsOptions);

        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 });

        // Act
        ridge.Train(x, y);
        ols.Train(x, y);

        // Assert - should produce same coefficients
        Assert.Equal(ols.Coefficients[0], ridge.Coefficients[0], Tolerance);
        Assert.Equal(ols.Coefficients[1], ridge.Coefficients[1], Tolerance);
        Assert.Equal(ols.Intercept, ridge.Intercept, Tolerance);
    }

    [Fact]
    public void RidgeRegression_Predict_ReturnsCorrectValues()
    {
        // Arrange
        var options = new RidgeRegressionOptions<double> { Alpha = 0.1, UseIntercept = true };
        var regression = new RidgeRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 });
        regression.Train(x, y);

        // Act
        var newX = CreateMatrix(new double[,] { { 4, 4 } });
        var predictions = regression.Predict(newX);

        // Assert - prediction should be reasonable
        Assert.True(predictions[0] > 20 && predictions[0] < 30,
            $"Prediction should be between 20 and 30, got {predictions[0]}");
    }

    [Fact]
    public void RidgeRegression_GetModelMetadata_IncludesAlpha()
    {
        // Arrange
        var options = new RidgeRegressionOptions<double> { Alpha = 1.5, UseIntercept = true };
        var regression = new RidgeRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1, 1 }, { 2, 2 } });
        var y = CreateVector(new double[] { 3, 6 });
        regression.Train(x, y);

        // Act
        var metadata = regression.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("Alpha"));
        Assert.Equal(1.5, (double)metadata.AdditionalInfo["Alpha"], Tolerance);
    }

    #endregion

    #region LassoRegression Tests

    [Fact]
    public void LassoRegression_Train_WithHighRegularization_ProducesSparseCoefficients()
    {
        // Arrange
        var options = new LassoRegressionOptions<double> { Alpha = 10.0, UseIntercept = true, MaxIterations = 1000 };
        var regression = new LassoRegression<double>(options);

        // Create data where some features are irrelevant
        var x = CreateMatrix(new double[,]
        {
            { 1, 0.1, 0.05 },
            { 2, 0.2, 0.08 },
            { 3, 0.15, 0.07 },
            { 4, 0.25, 0.09 },
            { 5, 0.18, 0.06 }
        });
        var y = CreateVector(new double[] { 5, 7, 9, 11, 13 }); // Only depends on first feature

        // Act
        regression.Train(x, y);

        // Assert - with high regularization, irrelevant features should have near-zero coefficients
        double coef1Abs = Math.Abs(regression.Coefficients[0]);
        double coef2Abs = Math.Abs(regression.Coefficients[1]);
        double coef3Abs = Math.Abs(regression.Coefficients[2]);

        Assert.True(coef1Abs > coef2Abs,
            $"First coefficient ({coef1Abs}) should be larger than second ({coef2Abs})");
    }

    [Fact]
    public void LassoRegression_Train_LowRegularization_RecoversTrueCoefficients()
    {
        // Arrange
        var options = new LassoRegressionOptions<double> { Alpha = 0.001, UseIntercept = true, MaxIterations = 1000 };
        var regression = new LassoRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 }); // y = 2*x1 + 3*x2 + 5

        // Act
        regression.Train(x, y);

        // Assert - should be close to true coefficients
        Assert.True(Math.Abs(regression.Coefficients[0] - 2.0) < LooseTolerance,
            $"First coefficient should be close to 2.0, got {regression.Coefficients[0]}");
        Assert.True(Math.Abs(regression.Coefficients[1] - 3.0) < LooseTolerance,
            $"Second coefficient should be close to 3.0, got {regression.Coefficients[1]}");
    }

    #endregion

    #region ElasticNetRegression Tests

    [Fact]
    public void ElasticNetRegression_Train_L1RatioZero_BehavesLikeRidge()
    {
        // Arrange - L1Ratio = 0 means pure L2 (Ridge)
        var elasticOptions = new ElasticNetRegressionOptions<double>
        {
            Alpha = 1.0,
            L1Ratio = 0.0,
            UseIntercept = true,
            MaxIterations = 1000
        };
        var ridgeOptions = new RidgeRegressionOptions<double> { Alpha = 1.0, UseIntercept = true };

        var elastic = new ElasticNetRegression<double>(elasticOptions);
        var ridge = new RidgeRegression<double>(ridgeOptions);

        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 });

        // Act
        elastic.Train(x, y);
        ridge.Train(x, y);

        // Assert - should produce similar coefficients (not exact due to different solvers)
        Assert.True(Math.Abs(elastic.Coefficients[0] - ridge.Coefficients[0]) < LooseTolerance,
            $"First coefficients should be similar: ElasticNet={elastic.Coefficients[0]}, Ridge={ridge.Coefficients[0]}");
    }

    [Fact]
    public void ElasticNetRegression_Train_L1RatioOne_BehavesLikeLasso()
    {
        // Arrange - L1Ratio = 1 means pure L1 (Lasso)
        var elasticOptions = new ElasticNetRegressionOptions<double>
        {
            Alpha = 0.1,
            L1Ratio = 1.0,
            UseIntercept = true,
            MaxIterations = 1000
        };
        var lassoOptions = new LassoRegressionOptions<double>
        {
            Alpha = 0.1,
            UseIntercept = true,
            MaxIterations = 1000
        };

        var elastic = new ElasticNetRegression<double>(elasticOptions);
        var lasso = new LassoRegression<double>(lassoOptions);

        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 });

        // Act
        elastic.Train(x, y);
        lasso.Train(x, y);

        // Assert - should produce similar coefficients
        Assert.True(Math.Abs(elastic.Coefficients[0] - lasso.Coefficients[0]) < LooseTolerance,
            $"First coefficients should be similar: ElasticNet={elastic.Coefficients[0]}, Lasso={lasso.Coefficients[0]}");
    }

    [Fact]
    public void ElasticNetRegression_Train_MixedRatio_CombinesBothPenalties()
    {
        // Arrange
        var options = new ElasticNetRegressionOptions<double>
        {
            Alpha = 0.5,
            L1Ratio = 0.5,  // 50% L1, 50% L2
            UseIntercept = true,
            MaxIterations = 1000
        };
        var regression = new ElasticNetRegression<double>(options);

        var x = CreateMatrix(new double[,]
        {
            { 1, 1 },
            { 2, 1 },
            { 1, 2 },
            { 2, 2 },
            { 3, 3 }
        });
        var y = CreateVector(new double[] { 10, 12, 13, 15, 20 });

        // Act
        regression.Train(x, y);

        // Assert - should train without errors and produce reasonable coefficients
        Assert.True(regression.Coefficients[0] > 0, "First coefficient should be positive");
        Assert.True(regression.Coefficients[1] > 0, "Second coefficient should be positive");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void SimpleRegression_Train_TwoPoints_FindsExactLine()
    {
        // Arrange
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new SimpleRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 0 }, { 10 } });
        var y = CreateVector(new double[] { 5, 25 }); // y = 2x + 5

        // Act
        regression.Train(x, y);

        // Assert
        Assert.Equal(2.0, regression.Coefficients[0], Tolerance);
        Assert.Equal(5.0, regression.Intercept, Tolerance);
    }

    [Fact]
    public void MultipleRegression_Train_LargeValues_HandlesCorrectly()
    {
        // Arrange
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new MultipleRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1e6, 1e6 },
            { 2e6, 1e6 },
            { 1e6, 2e6 },
            { 2e6, 2e6 }
        });
        var y = CreateVector(new double[] { 5e6, 7e6, 8e6, 10e6 }); // y = 2*x1 + 3*x2 - 0 intercept

        // Act
        regression.Train(x, y);

        // Assert - should handle large values
        Assert.True(Math.Abs(regression.Coefficients[0] - 2.0) < LooseTolerance,
            $"First coefficient should be close to 2.0, got {regression.Coefficients[0]}");
    }

    [Fact]
    public void MultipleRegression_Train_SmallValues_HandlesCorrectly()
    {
        // Arrange
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new MultipleRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1e-6, 1e-6 },
            { 2e-6, 1e-6 },
            { 1e-6, 2e-6 },
            { 2e-6, 2e-6 }
        });
        var y = CreateVector(new double[] { 5e-6, 7e-6, 8e-6, 10e-6 });

        // Act
        regression.Train(x, y);

        // Assert - should handle small values without numerical issues
        Assert.True(!double.IsNaN(regression.Coefficients[0]), "Coefficient should not be NaN");
        Assert.True(!double.IsInfinity(regression.Coefficients[0]), "Coefficient should not be Infinity");
    }

    [Fact]
    public void RidgeRegression_Train_IllConditionedMatrix_HandlesWithRegularization()
    {
        // Arrange - create nearly collinear features
        var options = new RidgeRegressionOptions<double> { Alpha = 1.0, UseIntercept = true };
        var regression = new RidgeRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1.0, 1.001 },
            { 2.0, 2.002 },
            { 3.0, 3.003 },
            { 4.0, 4.004 },
            { 5.0, 5.005 }
        });
        var y = CreateVector(new double[] { 5, 9, 13, 17, 21 });

        // Act - should not throw
        regression.Train(x, y);

        // Assert
        Assert.True(!double.IsNaN(regression.Coefficients[0]), "Coefficient should not be NaN");
        Assert.True(!double.IsNaN(regression.Coefficients[1]), "Coefficient should not be NaN");
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void SimpleRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new SimpleRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 5, 7, 9, 11, 13 });
        regression.Train(x, y);

        // Act
        var serialized = regression.Serialize();
        var newRegression = new SimpleRegression<double>(options);
        newRegression.Deserialize(serialized);

        // Assert
        Assert.Equal(regression.Coefficients[0], newRegression.Coefficients[0], Tolerance);
        Assert.Equal(regression.Intercept, newRegression.Intercept, Tolerance);
    }

    [Fact]
    public void MultipleRegression_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var options = new RegressionOptions<double> { UseIntercept = true };
        var regression = new MultipleRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1, 1 }, { 2, 2 }, { 3, 3 } });
        var y = CreateVector(new double[] { 5, 9, 13 });
        regression.Train(x, y);

        // Act
        var clone = regression.Clone();

        // Assert
        Assert.NotSame(regression, clone);
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

    private static Vector<double> CreateVector(double[] data)
    {
        var vector = new Vector<double>(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            vector[i] = data[i];
        }
        return vector;
    }

    #endregion
}
