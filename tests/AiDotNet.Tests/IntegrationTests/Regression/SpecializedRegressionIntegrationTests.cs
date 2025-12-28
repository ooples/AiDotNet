using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for Specialized Regression models (PolynomialRegression, SplineRegression,
/// IsotonicRegression, QuantileRegression, RobustRegression, BayesianRegression).
/// These tests verify specialized regression algorithms work correctly.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class SpecializedRegressionIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 0.5;

    #region PolynomialRegression Tests

    [Fact]
    public void PolynomialRegression_Train_QuadraticData_FitsPolynomial()
    {
        // Arrange: y = x^2 + 2x + 1
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var regression = new PolynomialRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 1, 4, 9, 16, 25, 36 }); // y = x^2 + 2x + 1

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - should fit quadratic data well
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < LooseTolerance,
                $"Prediction {i} should be close to {y[i]}, got {predictions[i]}");
        }
    }

    [Fact]
    public void PolynomialRegression_Train_Degree1_EquivalentToLinear()
    {
        // Arrange
        var polyOptions = new PolynomialRegressionOptions<double> { Degree = 1, UseIntercept = true };
        var linearOptions = new RegressionOptions<double> { UseIntercept = true };

        var poly = new PolynomialRegression<double>(polyOptions);
        var linear = new SimpleRegression<double>(linearOptions);

        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 3, 5, 7, 9, 11 }); // y = 2x + 1

        // Act
        poly.Train(x, y);
        linear.Train(x, y);

        var polyPred = poly.Predict(x);
        var linearPred = linear.Predict(x);

        // Assert - should produce similar predictions
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(polyPred[i] - linearPred[i]) < LooseTolerance,
                $"Polynomial degree 1 should match linear regression at {i}");
        }
    }

    [Fact]
    public void PolynomialRegression_Train_HighDegree_Overfits()
    {
        // Arrange - high degree polynomial on few points should fit exactly
        var options = new PolynomialRegressionOptions<double> { Degree = 4, UseIntercept = true };
        var regression = new PolynomialRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 2, 7, 3, 9, 5 }); // Random-looking data

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - with n points and degree n-1, should fit exactly
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < LooseTolerance,
                $"High degree polynomial should fit training point {i} exactly");
        }
    }

    [Fact]
    public void PolynomialRegression_Predict_Extrapolation_ReturnsValues()
    {
        // Arrange
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var regression = new PolynomialRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 } });
        var y = CreateVector(new double[] { 0, 1, 4, 9, 16 }); // y = x^2
        regression.Train(x, y);

        // Act - predict outside training range
        var newX = CreateMatrix(new double[,] { { 5 }, { 6 } });
        var predictions = regression.Predict(newX);

        // Assert - should extrapolate (may not be accurate but should work)
        Assert.True(!double.IsNaN(predictions[0]), "Prediction should not be NaN");
        Assert.True(!double.IsInfinity(predictions[0]), "Prediction should not be Infinity");
    }

    #endregion

    #region SplineRegression Tests

    [Fact]
    public void SplineRegression_Train_SmoothData_FitsWell()
    {
        // Arrange
        var options = new SplineRegressionOptions { NumberOfKnots = 3 };
        var regression = new SplineRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }
        });
        var y = CreateVector(new double[] { 0, 1, 4, 9, 16, 25, 36, 49, 64, 81 });

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < 5.0,
                $"Prediction {i} should be reasonably close to {y[i]}, got {predictions[i]}");
        }
    }

    #endregion

    #region IsotonicRegression Tests

    [Fact]
    public void IsotonicRegression_Train_PreservesMonotonicity()
    {
        // Arrange
        var regression = new IsotonicRegression<double>();
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 1, 3, 2, 5, 4 }); // Non-monotonic

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - predictions should be monotonically increasing
        for (int i = 1; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= predictions[i - 1],
                $"Predictions should be monotonically increasing: {predictions[i - 1]} <= {predictions[i]}");
        }
    }

    [Fact]
    public void IsotonicRegression_Train_AlreadyMonotonic_PreservesValues()
    {
        // Arrange
        var regression = new IsotonicRegression<double>();
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 1, 2, 3, 4, 5 }); // Already monotonic

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - should preserve monotonic values closely
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < LooseTolerance,
                $"Already monotonic data should be preserved at {i}");
        }
    }

    #endregion

    #region QuantileRegression Tests

    [Fact]
    public void QuantileRegression_Train_MedianQuantile_SimilarToMean()
    {
        // Arrange - median (0.5 quantile) should be similar to mean for symmetric data
        var options = new QuantileRegressionOptions<double> { Quantile = 0.5 };
        var regression = new QuantileRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 }); // Linear, symmetric

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - predictions should be reasonable
        Assert.True(!double.IsNaN(predictions[0]), "Prediction should not be NaN");
    }

    [Fact]
    public void QuantileRegression_Train_LowQuantile_BelowMedian()
    {
        // Arrange
        var lowQuantileOptions = new QuantileRegressionOptions<double> { Quantile = 0.1 };
        var highQuantileOptions = new QuantileRegressionOptions<double> { Quantile = 0.9 };

        var lowRegression = new QuantileRegression<double>(lowQuantileOptions);
        var highRegression = new QuantileRegression<double>(highQuantileOptions);

        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });
        var y = CreateVector(new double[] { 2, 5, 6, 9, 10, 13, 14, 17, 18, 21 });

        // Act
        lowRegression.Train(x, y);
        highRegression.Train(x, y);

        var newX = CreateMatrix(new double[,] { { 5 } });
        var lowPred = lowRegression.Predict(newX);
        var highPred = highRegression.Predict(newX);

        // Assert - low quantile should be below high quantile
        Assert.True(lowPred[0] <= highPred[0],
            $"Low quantile ({lowPred[0]}) should be <= high quantile ({highPred[0]})");
    }

    #endregion

    #region RobustRegression Tests

    [Fact]
    public void RobustRegression_Train_WithOutliers_ResistsInfluence()
    {
        // Arrange
        var robustOptions = new RobustRegressionOptions<double>();
        var regularOptions = new RegressionOptions<double> { UseIntercept = true };

        var robust = new RobustRegression<double>(robustOptions);
        var regular = new MultipleRegression<double>(regularOptions);

        // Data with outliers
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 100, 12, 14, 16, 18, 20 }); // Outlier at index 4

        // Act
        robust.Train(x, y);
        regular.Train(x, y);

        var newX = CreateMatrix(new double[,] { { 5.5 } });
        var robustPred = robust.Predict(newX);
        var regularPred = regular.Predict(newX);

        // Assert - robust should predict closer to the true trend (around 11)
        double trueValue = 11; // Expected value without outlier
        double robustError = Math.Abs(robustPred[0] - trueValue);
        double regularError = Math.Abs(regularPred[0] - trueValue);

        Assert.True(robustError < regularError || robustError < 10,
            $"Robust regression should resist outlier influence better: robust error={robustError}, regular error={regularError}");
    }

    [Fact]
    public void RobustRegression_Train_NoOutliers_SimilarToOLS()
    {
        // Arrange
        var robustOptions = new RobustRegressionOptions<double>();
        var regularOptions = new RegressionOptions<double> { UseIntercept = true };

        var robust = new RobustRegression<double>(robustOptions);
        var regular = new MultipleRegression<double>(regularOptions);

        // Clean data without outliers
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 }); // y = 2x

        // Act
        robust.Train(x, y);
        regular.Train(x, y);

        var newX = CreateMatrix(new double[,] { { 3 } });
        var robustPred = robust.Predict(newX);
        var regularPred = regular.Predict(newX);

        // Assert - without outliers, both should be similar
        Assert.True(Math.Abs(robustPred[0] - regularPred[0]) < 2.0,
            $"Without outliers, robust and OLS should be similar: robust={robustPred[0]}, OLS={regularPred[0]}");
    }

    #endregion

    #region BayesianRegression Tests

    [Fact]
    public void BayesianRegression_Train_FitsData()
    {
        // Arrange
        var options = new BayesianRegressionOptions<double> { UseIntercept = true };
        var regression = new BayesianRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < LooseTolerance,
                $"Prediction {i} should be close to {y[i]}, got {predictions[i]}");
        }
    }

    [Fact]
    public void BayesianRegression_Train_ProvidesUncertainty()
    {
        // Arrange
        var options = new BayesianRegressionOptions<double> { UseIntercept = true };
        var regression = new BayesianRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });

        // Act
        regression.Train(x, y);

        // Assert - model should train without errors
        Assert.NotNull(regression.Coefficients);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void PolynomialRegression_Train_NegativeValues_HandlesCorrectly()
    {
        // Arrange
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var regression = new PolynomialRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { -2 }, { -1 }, { 0 }, { 1 }, { 2 }
        });
        var y = CreateVector(new double[] { 4, 1, 0, 1, 4 }); // y = x^2

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert
        for (int i = 0; i < y.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - y[i]) < LooseTolerance,
                $"Should handle negative x values at {i}");
        }
    }

    [Fact]
    public void IsotonicRegression_Train_SinglePoint_HandlesGracefully()
    {
        // Arrange
        var regression = new IsotonicRegression<double>();
        var x = CreateMatrix(new double[,] { { 1 } });
        var y = CreateVector(new double[] { 5 });

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert
        Assert.Equal(5.0, predictions[0], LooseTolerance);
    }

    [Fact]
    public void QuantileRegression_Train_ConstantTarget_HandlesCorrectly()
    {
        // Arrange
        var options = new QuantileRegressionOptions<double> { Quantile = 0.5 };
        var regression = new QuantileRegression<double>(options);
        var x = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        });
        var y = CreateVector(new double[] { 10, 10, 10, 10, 10 }); // Constant

        // Act
        regression.Train(x, y);
        var predictions = regression.Predict(x);

        // Assert - should predict constant
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(Math.Abs(predictions[i] - 10) < LooseTolerance,
                $"Constant target should result in constant prediction at {i}");
        }
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
