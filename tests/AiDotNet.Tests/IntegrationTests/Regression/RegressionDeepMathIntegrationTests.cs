using AiDotNet.Exceptions;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Deep math-correctness integration tests for regression models.
/// Each test verifies a specific mathematical property or hand-calculated result.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class RegressionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double MediumTolerance = 1e-4;
    private const double LooseTolerance = 1e-2;

    // ========================================================================
    // SIMPLE REGRESSION: OLS closed-form verification
    // ========================================================================

    #region SimpleRegression OLS Closed-Form

    [Fact]
    public void SimpleRegression_PerfectLine_Y_Equals_2X_Plus_3()
    {
        // y = 2x + 3 for x = {1,2,3,4,5}
        // Hand-calculated OLS for perfect data: slope=2, intercept=3
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 5.0, 7, 9, 11, 13 }));

        Assert.Equal(2.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(3.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_NoIntercept_Y_Equals_4X()
    {
        // y = 4x, no intercept
        // OLS without intercept: slope = sum(x*y) / sum(x^2) = (4+16+36+64+100)/(1+4+9+16+25) = 220/55 = 4
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = false });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 4.0, 8, 12, 16, 20 }));

        Assert.Equal(4.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(0.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_HandCalculated_OLS_Formulas()
    {
        // x = {1,2,3,4,5}, y = {2,4,5,4,5}
        // n=5, sum_x=15, sum_y=20, sum_xy=66, sum_x2=55
        // xbar = 3, ybar = 4
        // slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        //       = (5*66 - 15*20) / (5*55 - 225)
        //       = (330 - 300) / (275 - 225) = 30/50 = 0.6
        // intercept = ybar - slope*xbar = 4 - 0.6*3 = 4 - 1.8 = 2.2
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 2.0, 4, 5, 4, 5 }));

        Assert.Equal(0.6, reg.Coefficients[0], Tolerance);
        Assert.Equal(2.2, reg.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_NegativeSlope_Y_Equals_Minus3X_Plus_10()
    {
        // y = -3x + 10
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 4 } }),
                  CreateVector(new[] { 10.0, 7, 4, 1, -2 }));

        Assert.Equal(-3.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(10.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_ConstantY_SlopeIsZero()
    {
        // y = 5 for all x → slope = 0, intercept = 5
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 5.0, 5, 5, 5, 5 }));

        Assert.Equal(0.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(5.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_Prediction_Matches_Manual_Computation()
    {
        // Train on y = 2x + 3, predict at x = {10, 20, 100}
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 5.0, 7, 9, 11, 13 }));
        var pred = reg.Predict(CreateMatrix(new double[,] { { 10 }, { 20 }, { 100 } }));

        Assert.Equal(23.0, pred[0], Tolerance);  // 2*10 + 3
        Assert.Equal(43.0, pred[1], Tolerance);  // 2*20 + 3
        Assert.Equal(203.0, pred[2], Tolerance); // 2*100 + 3
    }

    [Fact]
    public void SimpleRegression_TrainingResiduals_SumToZero()
    {
        // For OLS with intercept, residuals always sum to zero
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 2, 4, 5, 4, 5 };
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(yData);
        reg.Train(x, y);
        var pred = reg.Predict(x);

        double residualSum = 0;
        for (int i = 0; i < yData.Length; i++)
            residualSum += yData[i] - pred[i];

        Assert.Equal(0.0, residualSum, Tolerance);
    }

    [Fact]
    public void SimpleRegression_NoIntercept_HandCalculated()
    {
        // x = {1,2,3}, y = {3,5,8}
        // Without intercept: slope = sum(x*y)/sum(x^2) = (3+10+24)/(1+4+9) = 37/14 ≈ 2.642857
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = false });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } }),
                  CreateVector(new[] { 3.0, 5, 8 }));

        Assert.Equal(37.0 / 14.0, reg.Coefficients[0], Tolerance);
    }

    [Fact]
    public void SimpleRegression_TwoPoints_ExactLine()
    {
        // Two points: (1,5) and (3,11) → slope = (11-5)/(3-1) = 3, intercept = 5-3 = 2
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 3 } }),
                  CreateVector(new[] { 5.0, 11 }));

        Assert.Equal(3.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(2.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void SimpleRegression_LargeSlope_Y_Equals_1000X_Plus_1()
    {
        // y = 1000x + 1
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 1001.0, 2001, 3001, 4001, 5001 }));

        Assert.Equal(1000.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(1.0, reg.Intercept, Tolerance);
    }

    #endregion

    // ========================================================================
    // MULTIPLE REGRESSION: OLS closed-form verification
    // ========================================================================

    #region MultipleRegression OLS Closed-Form

    [Fact]
    public void MultipleRegression_TwoFeatures_PerfectFit()
    {
        // y = 2*x1 + 3*x2 + 1
        // (x1,x2,y): (1,1,6), (2,1,8), (1,2,9), (2,2,11), (3,1,10)
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(
            CreateMatrix(new double[,] { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 1 } }),
            CreateVector(new[] { 6.0, 8, 9, 11, 10 }));

        Assert.Equal(2.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(3.0, reg.Coefficients[1], Tolerance);
        Assert.Equal(1.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void MultipleRegression_NoIntercept_ThreeFeatures()
    {
        // y = 1*x1 + 2*x2 + 3*x3 (no intercept)
        var x = CreateMatrix(new double[,] {
            { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 },
            { 1, 1, 0 }, { 1, 0, 1 }, { 0, 1, 1 },
            { 1, 1, 1 }
        });
        var y = CreateVector(new[] { 1.0, 2, 3, 3, 4, 5, 6 });
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = false });
        reg.Train(x, y);

        Assert.Equal(1.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(2.0, reg.Coefficients[1], Tolerance);
        Assert.Equal(3.0, reg.Coefficients[2], Tolerance);
    }

    [Fact]
    public void MultipleRegression_Prediction_Equals_XBeta_Plus_Intercept()
    {
        // Verify that Predict(X) = X * beta + intercept
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var x = CreateMatrix(new double[,] {
            { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }
        });
        var y = CreateVector(new[] { 8.0, 18, 28, 38, 48 }); // y = 2*x1 + 3*x2 - 1? Let's just verify the identity
        reg.Train(x, y);

        var pred = reg.Predict(x);
        // Manual: pred[i] = coeff[0]*x[i,0] + coeff[1]*x[i,1] + intercept
        for (int i = 0; i < x.Rows; i++)
        {
            double expected = reg.Coefficients[0] * x[i, 0] + reg.Coefficients[1] * x[i, 1] + reg.Intercept;
            Assert.Equal(expected, pred[i], Tolerance);
        }
    }

    [Fact]
    public void MultipleRegression_Training_Residuals_SumToZero_WithIntercept()
    {
        // For OLS with intercept: sum of residuals = 0
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 3, 7, 5, 9, 11 };
        var x = CreateMatrix(new double[,] { { 1, 2 }, { 3, 1 }, { 2, 3 }, { 4, 2 }, { 5, 3 } });
        var y = CreateVector(yData);
        reg.Train(x, y);
        var pred = reg.Predict(x);

        double residualSum = 0;
        for (int i = 0; i < yData.Length; i++)
            residualSum += yData[i] - pred[i];

        Assert.Equal(0.0, residualSum, Tolerance);
    }

    [Fact]
    public void MultipleRegression_Training_Residuals_Orthogonal_To_Features()
    {
        // For OLS: X^T * (y - Xb) = 0 (residuals orthogonal to feature space)
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 3, 7, 5, 9, 11 };
        var x = CreateMatrix(new double[,] { { 1, 2 }, { 3, 1 }, { 2, 3 }, { 4, 2 }, { 5, 3 } });
        var y = CreateVector(yData);
        reg.Train(x, y);
        var pred = reg.Predict(x);

        // Compute X^T * residual for each feature
        double[] residuals = new double[yData.Length];
        for (int i = 0; i < yData.Length; i++)
            residuals[i] = yData[i] - pred[i];

        // X^T * residuals should be zero for each feature column
        for (int j = 0; j < x.Columns; j++)
        {
            double dotProduct = 0;
            for (int i = 0; i < x.Rows; i++)
                dotProduct += x[i, j] * residuals[i];
            Assert.Equal(0.0, dotProduct, Tolerance);
        }

        // Also orthogonal to constant column (for intercept) = sum of residuals = 0
        double sumResiduals = residuals.Sum();
        Assert.Equal(0.0, sumResiduals, Tolerance);
    }

    [Fact]
    public void MultipleRegression_IdentityDesign_RecoversMean()
    {
        // X = Identity, y = {2,4,6}: with intercept, the OLS solution is different
        // Without intercept: beta = (I'I)^-1 I' y = I^-1 y = y, so coefficients = y
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = false });
        reg.Train(
            CreateMatrix(new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }),
            CreateVector(new[] { 2.0, 4, 6 }));

        Assert.Equal(2.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(4.0, reg.Coefficients[1], Tolerance);
        Assert.Equal(6.0, reg.Coefficients[2], Tolerance);
    }

    [Fact]
    public void MultipleRegression_SimpleVsMultiple_SingleFeature_SameResult()
    {
        // When MultipleRegression has one feature, results should match SimpleRegression
        var simpleReg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var multiReg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new[] { 2.0, 4, 5, 4, 5 });

        simpleReg.Train(x, y);
        multiReg.Train(x, y);

        Assert.Equal(simpleReg.Coefficients[0], multiReg.Coefficients[0], Tolerance);
        Assert.Equal(simpleReg.Intercept, multiReg.Intercept, Tolerance);
    }

    [Fact]
    public void MultipleRegression_MeanOfY_Equals_Intercept_Plus_Coeffs_Dot_MeanX()
    {
        // OLS property: ybar = intercept + sum(coeff_j * xbar_j)
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 10, 20, 15, 25, 30 };
        var x = CreateMatrix(new double[,] { { 1, 5 }, { 2, 4 }, { 3, 3 }, { 4, 2 }, { 5, 1 } });
        var y = CreateVector(yData);
        reg.Train(x, y);

        double yBar = yData.Average();
        double predicted = reg.Intercept;
        for (int j = 0; j < x.Columns; j++)
        {
            double xBarJ = 0;
            for (int i = 0; i < x.Rows; i++) xBarJ += x[i, j];
            xBarJ /= x.Rows;
            predicted += reg.Coefficients[j] * xBarJ;
        }

        Assert.Equal(yBar, predicted, Tolerance);
    }

    #endregion

    // ========================================================================
    // POLYNOMIAL REGRESSION
    // ========================================================================

    #region PolynomialRegression

    [Fact]
    public void PolynomialRegression_Degree2_PerfectQuadratic()
    {
        // y = x^2, degree=2
        // With intercept: y = a*x + b*x^2 + c
        // For y = x^2: a=0, b=1, c=0
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);
        var x = CreateMatrix(new double[,] { { -2 }, { -1 }, { 0 }, { 1 }, { 2 } });
        var y = CreateVector(new[] { 4.0, 1, 0, 1, 4 }); // x^2

        reg.Train(x, y);

        // Polynomial features for degree 2: [x, x^2]
        // Coefficients should be [0, 1] (coefficient for x=0, coefficient for x^2=1)
        // Intercept should be 0
        Assert.Equal(0.0, reg.Coefficients[0], Tolerance); // x coefficient
        Assert.Equal(1.0, reg.Coefficients[1], Tolerance); // x^2 coefficient
        Assert.Equal(0.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void PolynomialRegression_Degree2_Y_Equals_3XSquared_Plus_2X_Plus_1()
    {
        // y = 3x^2 + 2x + 1
        // Polynomial features [x, x^2] with intercept
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);

        double[] xVals = { -3, -2, -1, 0, 1, 2, 3 };
        var x = CreateMatrix(xVals.Select(v => new[] { v }).ToArray());
        var y = CreateVector(xVals.Select(v => 3 * v * v + 2 * v + 1).ToArray());

        reg.Train(x, y);

        // Coefficients: [x_coeff, x^2_coeff], intercept
        Assert.Equal(2.0, reg.Coefficients[0], Tolerance); // x
        Assert.Equal(3.0, reg.Coefficients[1], Tolerance); // x^2
        Assert.Equal(1.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void PolynomialRegression_Degree1_EquivalentToLinear()
    {
        // Polynomial degree 1 should behave like linear regression
        var polyOptions = new PolynomialRegressionOptions<double> { Degree = 1, UseIntercept = true };
        var polyReg = new PolynomialRegression<double>(polyOptions);
        var linReg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });

        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new[] { 3.0, 5, 7, 6, 10 });

        polyReg.Train(x, y);
        linReg.Train(x, y);

        Assert.Equal(linReg.Coefficients[0], polyReg.Coefficients[0], Tolerance);
        Assert.Equal(linReg.Intercept, polyReg.Intercept, Tolerance);
    }

    [Fact]
    public void PolynomialRegression_Degree2_Prediction_Matches_Formula()
    {
        // Verify predictions use the polynomial expansion correctly
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);

        double[] xVals = { -2, -1, 0, 1, 2, 3 };
        var x = CreateMatrix(xVals.Select(v => new[] { v }).ToArray());
        var y = CreateVector(xVals.Select(v => 2 * v * v - v + 3).ToArray()); // y = 2x^2 - x + 3

        reg.Train(x, y);

        // Predict at new points
        var newX = CreateMatrix(new double[,] { { 5 }, { -5 }, { 10 } });
        var pred = reg.Predict(newX);

        Assert.Equal(2.0 * 25 - 5 + 3, pred[0], Tolerance);   // x=5: 48
        Assert.Equal(2.0 * 25 + 5 + 3, pred[1], Tolerance);   // x=-5: 58
        Assert.Equal(2.0 * 100 - 10 + 3, pred[2], Tolerance);  // x=10: 193
    }

    [Fact]
    public void PolynomialRegression_Degree3_PerfectCubic()
    {
        // y = x^3 (need degree 3 polynomial features: x, x^2, x^3)
        var options = new PolynomialRegressionOptions<double> { Degree = 3, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);

        double[] xVals = { -3, -2, -1, 0, 1, 2, 3, 4 };
        var x = CreateMatrix(xVals.Select(v => new[] { v }).ToArray());
        var y = CreateVector(xVals.Select(v => v * v * v).ToArray());

        reg.Train(x, y);

        // Coefficients: [x, x^2, x^3], intercept
        Assert.Equal(0.0, reg.Coefficients[0], MediumTolerance); // x
        Assert.Equal(0.0, reg.Coefficients[1], MediumTolerance); // x^2
        Assert.Equal(1.0, reg.Coefficients[2], MediumTolerance); // x^3
        Assert.Equal(0.0, reg.Intercept, MediumTolerance);
    }

    [Fact]
    public void PolynomialRegression_NoIntercept_Y_Equals_XSquared()
    {
        // y = x^2 without intercept
        // Features: [x, x^2], no intercept column
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = false };
        var reg = new PolynomialRegression<double>(options);

        // Use symmetric data around 0 so x^2 dominates
        double[] xVals = { -3, -2, -1, 1, 2, 3 }; // skip 0 for no-intercept
        var x = CreateMatrix(xVals.Select(v => new[] { v }).ToArray());
        var y = CreateVector(xVals.Select(v => v * v).ToArray());

        reg.Train(x, y);

        // Since data is symmetric, the x coefficient should be ~0 and x^2 should be ~1
        Assert.Equal(0.0, reg.Coefficients[0], Tolerance); // x
        Assert.Equal(1.0, reg.Coefficients[1], Tolerance); // x^2
    }

    #endregion

    // ========================================================================
    // RIDGE REGRESSION: L2 regularization verification
    // ========================================================================

    #region RidgeRegression

    [Fact]
    public void RidgeRegression_Alpha0_Equals_OLS()
    {
        // With alpha=0, Ridge should give same result as OLS
        // Use non-collinear features to ensure X'X is positive definite
        var olsReg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var ridgeReg = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 0, UseIntercept = true });

        var x = CreateMatrix(new double[,] { { 1, 3 }, { 3, 1 }, { 5, 4 }, { 7, 2 }, { 9, 5 } });
        var y = CreateVector(new[] { 10.0, 8, 21, 19, 32 }); // y = 2*x1 + 3*x2 - 1

        olsReg.Train(x, y);
        ridgeReg.Train(x, y);

        Assert.Equal(olsReg.Coefficients[0], ridgeReg.Coefficients[0], MediumTolerance);
        Assert.Equal(olsReg.Coefficients[1], ridgeReg.Coefficients[1], MediumTolerance);
        Assert.Equal(olsReg.Intercept, ridgeReg.Intercept, MediumTolerance);
    }

    [Fact]
    public void RidgeRegression_LargeAlpha_Shrinks_Coefficients_Toward_Zero()
    {
        // With increasing alpha, coefficients should shrink toward zero
        // Use non-collinear features
        var x = CreateMatrix(new double[,] {
            { 1, 3 }, { 3, 1 }, { 5, 4 }, { 7, 2 }, { 9, 5 }
        });
        var y = CreateVector(new[] { 10.0, 8, 21, 19, 32 });

        var small = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 0.01, UseIntercept = true });
        var medium = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 1.0, UseIntercept = true });
        var large = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 100.0, UseIntercept = true });

        small.Train(x, y);
        medium.Train(x, y);
        large.Train(x, y);

        // L2 norm of coefficients should decrease with increasing alpha
        double normSmall = Math.Sqrt(small.Coefficients[0] * small.Coefficients[0] +
                                     small.Coefficients[1] * small.Coefficients[1]);
        double normMedium = Math.Sqrt(medium.Coefficients[0] * medium.Coefficients[0] +
                                      medium.Coefficients[1] * medium.Coefficients[1]);
        double normLarge = Math.Sqrt(large.Coefficients[0] * large.Coefficients[0] +
                                     large.Coefficients[1] * large.Coefficients[1]);

        Assert.True(normSmall > normMedium, $"Small alpha ({normSmall}) should have larger norm than medium ({normMedium})");
        Assert.True(normMedium > normLarge, $"Medium alpha ({normMedium}) should have larger norm than large ({normLarge})");
    }

    [Fact]
    public void RidgeRegression_VeryLargeAlpha_Coefficients_NearZero()
    {
        // With very large alpha, coefficients should be nearly zero
        // Use non-collinear features
        var x = CreateMatrix(new double[,] {
            { 1, 3 }, { 3, 1 }, { 5, 4 }, { 7, 2 }, { 9, 5 }
        });
        var y = CreateVector(new[] { 10.0, 8, 21, 19, 32 });

        var reg = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 1e6, UseIntercept = true });
        reg.Train(x, y);

        Assert.True(Math.Abs(reg.Coefficients[0]) < 0.1, $"Coefficient[0] should be near zero: {reg.Coefficients[0]}");
        Assert.True(Math.Abs(reg.Coefficients[1]) < 0.1, $"Coefficient[1] should be near zero: {reg.Coefficients[1]}");
    }

    [Fact]
    public void RidgeRegression_HandCalculated_SingleFeature()
    {
        // Single feature, no intercept for simpler hand calculation
        // X = [1; 2; 3], y = [2; 4; 6]
        // X'X = 1+4+9 = 14
        // X'y = 2+8+18 = 28
        // Ridge solution: beta = (X'X + alpha*I)^-1 * X'y
        // For alpha=1: beta = 28 / (14+1) = 28/15 ≈ 1.8667
        // For alpha=0: beta = 28/14 = 2.0 (OLS)
        var reg = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 1.0, UseIntercept = false });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } }),
                  CreateVector(new[] { 2.0, 4, 6 }));

        Assert.Equal(28.0 / 15.0, reg.Coefficients[0], MediumTolerance);
    }

    [Fact]
    public void RidgeRegression_DoesNotRegularize_Intercept()
    {
        // Ridge should NOT regularize the intercept term.
        // With very large alpha and data centered at mean, intercept should still be ~ybar.
        var x = CreateMatrix(new double[,] {
            { -2 }, { -1 }, { 0 }, { 1 }, { 2 }
        });
        var y = CreateVector(new[] { 8.0, 9, 10, 11, 12 }); // y = x + 10

        var reg = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 1000, UseIntercept = true });
        reg.Train(x, y);

        // Intercept should still be near the mean of y = 10
        // (slope will be shrunk toward 0 but intercept should be preserved)
        Assert.Equal(10.0, reg.Intercept, LooseTolerance);
    }

    [Fact]
    public void RidgeRegression_PerfectData_Prediction_Still_Reasonable()
    {
        // Even with regularization, predictions should be reasonable
        // Ridge introduces bias, so predictions won't be exact - use wider tolerance
        var options = new RidgeRegressionOptions<double> { Alpha = 0.1, UseIntercept = true };
        var reg = new RidgeRegression<double>(options);
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new[] { 3.0, 5, 7, 9, 11 }); // y = 2x + 1
        reg.Train(x, y);

        var pred = reg.Predict(x);
        // Ridge with alpha=0.1 introduces ~2-5% prediction bias on this data
        double ridgeTolerance = 0.1;
        for (int i = 0; i < y.Length; i++)
        {
            Assert.Equal(y[i], pred[i], ridgeTolerance);
        }
    }

    #endregion

    // ========================================================================
    // CROSS-MODEL CONSISTENCY: mathematical identities between models
    // ========================================================================

    #region Cross-Model Consistency

    [Fact]
    public void AllLinearModels_PerfectData_SameCoefficients()
    {
        // On perfect linear data with tiny alpha, all should agree
        // Use non-collinear features to ensure well-conditioned X'X
        var x = CreateMatrix(new double[,] { { 1, 3 }, { 3, 1 }, { 5, 4 }, { 7, 2 }, { 9, 5 } });
        var y = CreateVector(new[] { 10.0, 8, 21, 19, 32 }); // y = 2*x1 + 3*x2 - 1

        var multiReg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var ridgeReg = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 1e-10, UseIntercept = true });

        multiReg.Train(x, y);
        ridgeReg.Train(x, y);

        Assert.Equal(multiReg.Coefficients[0], ridgeReg.Coefficients[0], LooseTolerance);
        Assert.Equal(multiReg.Coefficients[1], ridgeReg.Coefficients[1], LooseTolerance);
        Assert.Equal(multiReg.Intercept, ridgeReg.Intercept, LooseTolerance);
    }

    [Fact]
    public void OLS_R_Squared_Equals_1_For_PerfectFit()
    {
        // For perfect linear data, R^2 should be exactly 1
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 5.0, 7, 9, 11, 13 }; // y = 2x + 3
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(yData);
        reg.Train(x, y);
        var pred = reg.Predict(x);

        // R^2 = 1 - SS_res/SS_tot
        double yBar = yData.Average();
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < yData.Length; i++)
        {
            ssRes += (yData[i] - pred[i]) * (yData[i] - pred[i]);
            ssTot += (yData[i] - yBar) * (yData[i] - yBar);
        }
        double rSquared = 1 - ssRes / ssTot;

        Assert.Equal(1.0, rSquared, Tolerance);
    }

    [Fact]
    public void OLS_TSS_Equals_ESS_Plus_RSS()
    {
        // Total Sum of Squares = Explained SS + Residual SS
        // TSS = sum((y_i - ybar)^2)
        // ESS = sum((yhat_i - ybar)^2)
        // RSS = sum((y_i - yhat_i)^2)
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 3, 7, 5, 9, 11 };
        var x = CreateMatrix(new double[,] { { 1, 2 }, { 3, 1 }, { 2, 3 }, { 4, 2 }, { 5, 3 } });
        var y = CreateVector(yData);
        reg.Train(x, y);
        var pred = reg.Predict(x);

        double yBar = yData.Average();
        double tss = 0, ess = 0, rss = 0;
        for (int i = 0; i < yData.Length; i++)
        {
            tss += (yData[i] - yBar) * (yData[i] - yBar);
            ess += (pred[i] - yBar) * (pred[i] - yBar);
            rss += (yData[i] - pred[i]) * (yData[i] - pred[i]);
        }

        Assert.Equal(tss, ess + rss, Tolerance);
    }

    [Fact]
    public void SimpleRegression_R2_Between_0_And_1()
    {
        // For typical data with intercept, R^2 is between 0 and 1
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 2, 4, 5, 4, 5, 7, 8 };
        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } });
        var y = CreateVector(yData);
        reg.Train(x, y);
        var pred = reg.Predict(x);

        double yBar = yData.Average();
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < yData.Length; i++)
        {
            ssRes += (yData[i] - pred[i]) * (yData[i] - pred[i]);
            ssTot += (yData[i] - yBar) * (yData[i] - yBar);
        }
        double rSquared = 1 - ssRes / ssTot;

        Assert.True(rSquared >= 0.0 && rSquared <= 1.0, $"R^2 should be in [0,1], got {rSquared}");
    }

    [Fact]
    public void MultipleRegression_Adding_Feature_Cannot_Decrease_R2()
    {
        // Adding a feature can only increase (or maintain) R^2
        double[] yData = { 2, 5, 3, 8, 7, 10, 6 };
        var x1 = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 } });
        var x2 = CreateMatrix(new double[,] {
            { 1, 3 }, { 2, 1 }, { 3, 4 }, { 4, 2 }, { 5, 5 }, { 6, 3 }, { 7, 6 }
        });
        var y = CreateVector(yData);

        var reg1 = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var reg2 = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg1.Train(x1, y);
        reg2.Train(x2, y);

        double r2_1 = ComputeR2(yData, reg1.Predict(x1));
        double r2_2 = ComputeR2(yData, reg2.Predict(x2));

        Assert.True(r2_2 >= r2_1 - Tolerance,
            $"R^2 with 2 features ({r2_2}) should be >= R^2 with 1 feature ({r2_1})");
    }

    #endregion

    // ========================================================================
    // EDGE CASES AND NUMERICAL STABILITY
    // ========================================================================

    #region Edge Cases

    [Fact]
    public void SimpleRegression_ZeroSlope_AllSameY()
    {
        // When all y values are the same, slope=0
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 42.0, 42, 42, 42, 42 }));

        Assert.Equal(0.0, reg.Coefficients[0], Tolerance);
        Assert.Equal(42.0, reg.Intercept, Tolerance);
    }

    [Fact]
    public void MultipleRegression_OneFeature_AllZeros_Coefficient_Is_Zero()
    {
        // If one feature is always zero, the matrix is rank-deficient.
        // The solver should handle this gracefully via SVD fallback (not return NaN).
        // y = 3*x1 + 0*x2 + 1
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(
            CreateMatrix(new double[,] { { 1, 0 }, { 2, 0 }, { 3, 0 }, { 4, 0 }, { 5, 0 } }),
            CreateVector(new[] { 4.0, 7, 10, 13, 16 }));

        // Coefficients should not be NaN
        Assert.False(double.IsNaN(reg.Coefficients[0]), "Coefficient[0] should not be NaN");
        Assert.False(double.IsNaN(reg.Intercept), "Intercept should not be NaN");

        // The active feature's coefficient should recover the slope
        Assert.Equal(3.0, reg.Coefficients[0], MediumTolerance);

        // Predictions should be correct
        var pred = reg.Predict(CreateMatrix(new double[,] { { 6, 0 } }));
        Assert.Equal(19.0, pred[0], MediumTolerance); // 3*6 + 1
    }

    [Fact]
    public void SimpleRegression_LargeValues_StillAccurate()
    {
        // Test with large feature values
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(
            CreateMatrix(new double[,] { { 1e6 }, { 2e6 }, { 3e6 }, { 4e6 }, { 5e6 } }),
            CreateVector(new[] { 2e6 + 100, 4e6 + 100, 6e6 + 100, 8e6 + 100, 10e6 + 100 }));

        Assert.Equal(2.0, reg.Coefficients[0], MediumTolerance);
        Assert.Equal(100.0, reg.Intercept, MediumTolerance);
    }

    [Fact]
    public void SimpleRegression_SmallValues_StillAccurate()
    {
        // Test with small feature values
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(
            CreateMatrix(new double[,] { { 1e-6 }, { 2e-6 }, { 3e-6 }, { 4e-6 }, { 5e-6 } }),
            CreateVector(new[] { 2e-6 + 1e-7, 4e-6 + 1e-7, 6e-6 + 1e-7, 8e-6 + 1e-7, 10e-6 + 1e-7 }));

        Assert.Equal(2.0, reg.Coefficients[0], LooseTolerance);
        Assert.Equal(1e-7, reg.Intercept, LooseTolerance);
    }

    [Fact]
    public void MultipleRegression_HighlyCorrelated_Features_Handles_Gracefully()
    {
        // Collinear features: x2 = 2*x1
        // y = 3*x1 + 1 (since x2 is a multiple of x1, the coefficient split is non-unique)
        // The solver should still produce valid predictions (not NaN) via SVD fallback
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var x = CreateMatrix(new double[,] {
            { 1, 2 }, { 2, 4 }, { 3, 6 }, { 4, 8 }, { 5, 10 }
        });
        var y = CreateVector(new[] { 4.0, 7, 10, 13, 16 }); // y = 3x1 + 1

        reg.Train(x, y);

        // Coefficients should not contain NaN (SVD should give minimum-norm solution)
        for (int i = 0; i < reg.Coefficients.Length; i++)
        {
            Assert.False(double.IsNaN(reg.Coefficients[i]),
                $"Coefficient[{i}] should not be NaN for collinear data");
        }
        Assert.False(double.IsNaN(reg.Intercept), "Intercept should not be NaN");

        // Predictions should still be correct even if individual coefficients are unstable
        var pred = reg.Predict(x);
        for (int i = 0; i < y.Length; i++)
        {
            Assert.Equal(y[i], pred[i], MediumTolerance);
        }
    }

    [Fact]
    public void RidgeRegression_Collinear_StabilizesCoefficients()
    {
        // Ridge should handle near-collinear features better than OLS
        // Use features with moderate correlation (not perfectly collinear)
        var x = CreateMatrix(new double[,] {
            { 1, 3 }, { 2, 5 }, { 3, 8 }, { 4, 10 }, { 5, 14 },
            { 6, 17 }, { 7, 20 }
        });
        var y = CreateVector(new[] { 4.0, 7, 10, 13, 16, 19, 22 });

        var olsReg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var ridgeReg = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 1.0, UseIntercept = true });

        olsReg.Train(x, y);
        ridgeReg.Train(x, y);

        // Ridge coefficients should have smaller L2 norm
        double olsNorm = Math.Sqrt(olsReg.Coefficients[0] * olsReg.Coefficients[0] +
                                   olsReg.Coefficients[1] * olsReg.Coefficients[1]);
        double ridgeNorm = Math.Sqrt(ridgeReg.Coefficients[0] * ridgeReg.Coefficients[0] +
                                     ridgeReg.Coefficients[1] * ridgeReg.Coefficients[1]);

        Assert.True(ridgeNorm <= olsNorm + MediumTolerance,
            $"Ridge norm ({ridgeNorm}) should be <= OLS norm ({olsNorm})");
    }

    #endregion

    // ========================================================================
    // COEFFICIENT COUNT AND METADATA
    // ========================================================================

    #region Metadata and Structure

    [Fact]
    public void SimpleRegression_HasExactlyOneCoefficient()
    {
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } }),
                  CreateVector(new[] { 2.0, 4, 6 }));

        Assert.Equal(1, reg.Coefficients.Length);
    }

    [Fact]
    public void MultipleRegression_CoefficientCount_Matches_FeatureCount()
    {
        int numFeatures = 4;
        var x = CreateMatrix(new double[,] {
            { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 },
            { 13, 14, 15, 16 }, { 17, 18, 19, 20 }
        });
        var y = CreateVector(new[] { 10.0, 26, 42, 58, 74 });

        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(x, y);

        Assert.Equal(numFeatures, reg.Coefficients.Length);
    }

    [Fact]
    public void PolynomialRegression_CoefficientCount_Equals_OriginalFeatures_Times_Degree()
    {
        // With 1 original feature and degree=3, should have 3 polynomial features
        var options = new PolynomialRegressionOptions<double> { Degree = 3, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 1.0, 8, 27, 64, 125 }));

        Assert.Equal(3, reg.Coefficients.Length); // x, x^2, x^3
    }

    [Fact]
    public void SimpleRegression_HasIntercept_ReflectsOptions()
    {
        var withIntercept = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var noIntercept = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = false });

        Assert.True(withIntercept.HasIntercept);
        Assert.False(noIntercept.HasIntercept);
    }

    [Fact]
    public void RidgeRegression_Metadata_Contains_Alpha()
    {
        var options = new RidgeRegressionOptions<double> { Alpha = 2.5, UseIntercept = true };
        var reg = new RidgeRegression<double>(options);
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } }),
                  CreateVector(new[] { 2.0, 4, 6 }));

        var metadata = reg.GetModelMetadata();
        Assert.True(metadata.AdditionalInfo.ContainsKey("Alpha"), "Metadata should contain Alpha");
        Assert.Equal(2.5, (double)metadata.AdditionalInfo["Alpha"], Tolerance);
    }

    #endregion

    // ========================================================================
    // POLYNOMIAL REGRESSION: Feature expansion correctness
    // ========================================================================

    #region Polynomial Feature Expansion

    [Fact]
    public void PolynomialRegression_TwoFeatures_Degree2_CorrectExpansion()
    {
        // With 2 original features and degree=2, polynomial features should be:
        // [x1, x1^2, x2, x2^2] → 4 coefficients
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);

        var x = CreateMatrix(new double[,] {
            { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 }
        });
        // y = x1^2 + x2^2 for verification
        var y = CreateVector(new[] { 2.0, 5, 5, 8, 18 });

        reg.Train(x, y);

        // Should have 4 coefficients (2 features * degree 2)
        Assert.Equal(4, reg.Coefficients.Length);
    }

    [Fact]
    public void PolynomialRegression_Degree2_Symmetric_PredictionSymmetric()
    {
        // y = x^2 is symmetric around 0: f(a) = f(-a)
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);

        double[] xVals = { -3, -2, -1, 0, 1, 2, 3 };
        var x = CreateMatrix(xVals.Select(v => new[] { v }).ToArray());
        var y = CreateVector(xVals.Select(v => v * v).ToArray());
        reg.Train(x, y);

        var pred = reg.Predict(CreateMatrix(new double[,] { { 4 }, { -4 } }));

        Assert.Equal(pred[0], pred[1], Tolerance); // f(4) should equal f(-4)
    }

    #endregion

    // ========================================================================
    // REGRESSION VALIDATION: input validation
    // ========================================================================

    #region Input Validation

    [Fact]
    public void SimpleRegression_MultipleColumns_ThrowsInvalidInputDimensionException()
    {
        // SimpleRegression requires exactly 1 feature column
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var x = CreateMatrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var y = CreateVector(new[] { 1.0, 2 });

        Assert.Throws<InvalidInputDimensionException>(() => reg.Train(x, y));
    }

    [Fact]
    public void RidgeRegression_NegativeAlpha_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            var options = new RidgeRegressionOptions<double> { Alpha = -1.0 };
        });
    }

    #endregion

    // ========================================================================
    // NUMERICAL IDENTITY: Gauss-Markov theorem implications
    // ========================================================================

    #region Gauss-Markov Properties

    [Fact]
    public void OLS_IsUnbiased_MeanPrediction_Equals_MeanActual()
    {
        // For OLS with intercept: mean(yhat) = mean(y)
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 3, 7, 2, 9, 5, 8, 4 };
        var x = CreateMatrix(new double[,] {
            { 1, 3 }, { 2, 1 }, { 3, 4 }, { 4, 2 }, { 5, 5 }, { 6, 3 }, { 7, 6 }
        });
        var y = CreateVector(yData);
        reg.Train(x, y);
        var pred = reg.Predict(x);

        double meanActual = yData.Average();
        double meanPredicted = 0;
        for (int i = 0; i < pred.Length; i++) meanPredicted += pred[i];
        meanPredicted /= pred.Length;

        Assert.Equal(meanActual, meanPredicted, Tolerance);
    }

    [Fact]
    public void OLS_Normal_Equations_Satisfied()
    {
        // The normal equations: X'X * beta = X'y (when no regularization, no intercept trick)
        // With intercept prepended, we check that [1|X]'[1|X]*[intercept;beta] = [1|X]'y
        var reg = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        double[] yData = { 3, 7, 5, 9, 11 };
        var x = CreateMatrix(new double[,] { { 1, 2 }, { 3, 1 }, { 2, 3 }, { 4, 2 }, { 5, 3 } });
        var y = CreateVector(yData);
        reg.Train(x, y);

        // Build augmented X = [1 | X]
        int n = x.Rows, p = x.Columns;
        double[,] augX = new double[n, p + 1];
        for (int i = 0; i < n; i++)
        {
            augX[i, 0] = 1.0; // intercept column
            for (int j = 0; j < p; j++)
                augX[i, j + 1] = x[i, j];
        }

        // Build beta_full = [intercept, coeff0, coeff1, ...]
        double[] betaFull = new double[p + 1];
        betaFull[0] = reg.Intercept;
        for (int j = 0; j < p; j++)
            betaFull[j + 1] = reg.Coefficients[j];

        // Check X'X * beta = X'y
        for (int col = 0; col < p + 1; col++)
        {
            double xTxBeta = 0;
            double xTy = 0;
            for (int i = 0; i < n; i++)
            {
                xTy += augX[i, col] * yData[i];
                double xBeta = 0;
                for (int k = 0; k < p + 1; k++)
                    xBeta += augX[i, k] * betaFull[k];
                xTxBeta += augX[i, col] * xBeta;
            }
            Assert.Equal(xTy, xTxBeta, Tolerance);
        }
    }

    [Fact]
    public void Polynomial_PerfectFit_Residuals_All_Zero()
    {
        // Polynomial of degree d should perfectly fit d+1 points
        // 3 points → degree 2 should give 0 residuals
        var options = new PolynomialRegressionOptions<double> { Degree = 2, UseIntercept = true };
        var reg = new PolynomialRegression<double>(options);

        var x = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var y = CreateVector(new[] { 10.0, 20, 35 }); // Arbitrary 3 points

        reg.Train(x, y);
        var pred = reg.Predict(x);

        for (int i = 0; i < y.Length; i++)
        {
            Assert.Equal(y[i], pred[i], Tolerance);
        }
    }

    #endregion

    // ========================================================================
    // SERIALIZATION ROUND-TRIP
    // ========================================================================

    #region Serialization

    [Fact]
    public void SimpleRegression_Serialize_Deserialize_PreservesModel()
    {
        var reg = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg.Train(CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } }),
                  CreateVector(new[] { 5.0, 7, 9, 11, 13 }));

        byte[] serialized = reg.Serialize();
        var reg2 = new SimpleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        reg2.Deserialize(serialized);

        Assert.Equal(reg.Coefficients[0], reg2.Coefficients[0], Tolerance);
        Assert.Equal(reg.Intercept, reg2.Intercept, Tolerance);

        // Predictions should match
        var testX = CreateMatrix(new double[,] { { 10 }, { 20 } });
        var pred1 = reg.Predict(testX);
        var pred2 = reg2.Predict(testX);
        Assert.Equal(pred1[0], pred2[0], Tolerance);
        Assert.Equal(pred1[1], pred2[1], Tolerance);
    }

    [Fact]
    public void RidgeRegression_Serialize_Deserialize_PreservesModel()
    {
        var options = new RidgeRegressionOptions<double> { Alpha = 2.0, UseIntercept = true };
        var reg = new RidgeRegression<double>(options);
        reg.Train(CreateMatrix(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 } }),
                  CreateVector(new[] { 3.0, 7, 11, 15, 19 }));

        byte[] serialized = reg.Serialize();
        var reg2 = new RidgeRegression<double>(new RidgeRegressionOptions<double>());
        reg2.Deserialize(serialized);

        for (int i = 0; i < reg.Coefficients.Length; i++)
            Assert.Equal(reg.Coefficients[i], reg2.Coefficients[i], Tolerance);
        Assert.Equal(reg.Intercept, reg2.Intercept, Tolerance);
    }

    #endregion

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    #region Helpers

    private static Matrix<double> CreateMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = data[i, j];
        return matrix;
    }

    private static Matrix<double> CreateMatrix(double[][] data)
    {
        int rows = data.Length;
        int cols = data[0].Length;
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = data[i][j];
        return matrix;
    }

    private static Vector<double> CreateVector(double[] data)
    {
        var vector = new Vector<double>(data.Length);
        for (int i = 0; i < data.Length; i++)
            vector[i] = data[i];
        return vector;
    }

    private static double ComputeR2(double[] actual, Vector<double> predicted)
    {
        double yBar = actual.Average();
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            ssRes += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
            ssTot += (actual[i] - yBar) * (actual[i] - yBar);
        }
        return 1 - ssRes / ssTot;
    }

    #endregion
}
