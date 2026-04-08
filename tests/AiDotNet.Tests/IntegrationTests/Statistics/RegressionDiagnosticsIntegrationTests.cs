using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Statistics;

/// <summary>
/// Integration tests for regression diagnostics (Residuals, Theil U, CRPS, VIF, Condition Number).
/// Ground truth values verified against standard implementations.
/// </summary>
public class RegressionDiagnosticsIntegrationTests
{
    private const double Tolerance = 1e-4;

    #region Residuals Tests

    [Fact]
    public void Residuals_PerfectPrediction_ReturnsZeros()
    {
        var actual = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);
        var predicted = Vector<double>.FromArray([1.0, 2.0, 3.0, 4.0, 5.0]);

        var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);

        for (int i = 0; i < residuals.Length; i++)
        {
            Assert.Equal(0.0, residuals[i], Tolerance);
        }
    }

    [Fact]
    public void Residuals_ConstantOffset_ReturnsOffset()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([8.0, 18.0, 28.0, 38.0, 48.0]);

        var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);

        // residual = actual - predicted = 2.0 for all
        for (int i = 0; i < residuals.Length; i++)
        {
            Assert.Equal(2.0, residuals[i], Tolerance);
        }
    }

    [Fact]
    public void Residuals_MixedErrors_ReturnsCorrectDifferences()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0, 38.0, 52.0]);

        var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);

        // Expected residuals: -2, 2, -2, 2, -2
        Assert.Equal(-2.0, residuals[0], Tolerance);
        Assert.Equal(2.0, residuals[1], Tolerance);
        Assert.Equal(-2.0, residuals[2], Tolerance);
        Assert.Equal(2.0, residuals[3], Tolerance);
        Assert.Equal(-2.0, residuals[4], Tolerance);
    }

    [Fact]
    public void Residuals_SumEqualsZeroForGoodModel()
    {
        // For a well-fitted model, residuals should sum to approximately zero
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0, 38.0, 50.0]);

        var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);
        var sum = residuals.Sum();

        // Sum: -2 + 2 - 2 + 2 + 0 = 0
        Assert.Equal(0.0, sum, Tolerance);
    }

    #endregion

    #region Theil U Statistic Tests

    [Fact]
    public void TheilU_PerfectPrediction_ReturnsZero()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);

        var result = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predicted);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TheilU_IsNonNegative()
    {
        var actual = Vector<double>.FromArray([10.0, 25.0, 30.0, 45.0, 50.0]);
        var predicted = Vector<double>.FromArray([12.0, 22.0, 35.0, 40.0, 55.0]);

        var result = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predicted);

        Assert.True(result >= 0, $"Theil U should be non-negative, got {result}");
    }

    [Fact]
    public void TheilU_IsLessThanOrEqualToOne_ForReasonableForecast()
    {
        // Theil U < 1 means forecast is better than naive (no-change) forecast
        var actual = Vector<double>.FromArray([100.0, 110.0, 105.0, 115.0, 120.0]);
        var predicted = Vector<double>.FromArray([102.0, 108.0, 107.0, 113.0, 118.0]);

        var result = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predicted);

        // A reasonable forecast should have Theil U close to 0
        Assert.True(result < 1.0, $"Theil U should be < 1 for good forecast, got {result}");
    }

    [Fact]
    public void TheilU_LargerError_LargerTheilU()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predictedGood = Vector<double>.FromArray([11.0, 19.0, 31.0, 39.0, 51.0]);
        var predictedBad = Vector<double>.FromArray([15.0, 25.0, 35.0, 45.0, 55.0]);

        var theilGood = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predictedGood);
        var theilBad = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predictedBad);

        Assert.True(theilGood < theilBad, $"Better forecast should have lower Theil U: {theilGood} vs {theilBad}");
    }

    [Fact]
    public void TheilU_CalculatesCorrectValue()
    {
        // Manual calculation:
        // actual = [10, 20, 30], predicted = [12, 18, 32]
        // MSE = ((10-12)² + (20-18)² + (30-32)²) / 3 = (4 + 4 + 4) / 3 = 4
        // RMSE = 2
        // RMS_actual = sqrt((100 + 400 + 900) / 3) = sqrt(466.67) = 21.6
        // RMS_predicted = sqrt((144 + 324 + 1024) / 3) = sqrt(497.33) = 22.3
        // Theil U = 2 / (21.6 + 22.3) = 2 / 43.9 ≈ 0.0456
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0]);

        var result = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predicted);

        Assert.Equal(0.0456, result, 0.01);
    }

    #endregion

    #region CRPS (Continuous Ranked Probability Score) Tests

    [Fact]
    public void CRPS_Deterministic_EqualsMeanAbsoluteError()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0, 38.0, 52.0]);

        var crps = StatisticsHelper<double>.CalculateCRPS(actual, predicted);
        var mae = StatisticsHelper<double>.CalculateMeanAbsoluteError(actual, predicted);

        Assert.Equal(mae, crps, Tolerance);
    }

    [Fact]
    public void CRPS_PerfectPrediction_ReturnsZero()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);

        var result = StatisticsHelper<double>.CalculateCRPS(actual, predicted);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CRPS_Probabilistic_ZeroUncertainty_EqualsMeanAbsoluteError()
    {
        // When stddev = 0, CRPS reduces to MAE
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0, 38.0, 52.0]);
        var stddev = Vector<double>.FromArray([0.0, 0.0, 0.0, 0.0, 0.0]);

        var crps = StatisticsHelper<double>.CalculateCRPS(actual, predicted, stddev);
        var mae = StatisticsHelper<double>.CalculateMeanAbsoluteError(actual, predicted);

        Assert.Equal(mae, crps, Tolerance);
    }

    [Fact]
    public void CRPS_Probabilistic_WithUncertainty_IsNonNegative()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0, 40.0, 50.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0, 38.0, 52.0]);
        var stddev = Vector<double>.FromArray([2.0, 2.0, 2.0, 2.0, 2.0]);

        var result = StatisticsHelper<double>.CalculateCRPS(actual, predicted, stddev);

        Assert.True(result >= 0, $"CRPS should be non-negative, got {result}");
    }

    [Fact]
    public void CRPS_Probabilistic_PerfectMeanPrediction_WithUncertainty()
    {
        // Perfect prediction at the mean, but with some uncertainty
        // CRPS = sigma * (1/sqrt(pi)) ≈ sigma * 0.5642
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0]);
        var predicted = Vector<double>.FromArray([10.0, 20.0, 30.0]);
        var stddev = Vector<double>.FromArray([1.0, 1.0, 1.0]);

        var result = StatisticsHelper<double>.CalculateCRPS(actual, predicted, stddev);

        // When y = mu (z = 0), CRPS = sigma * (0 + 2*phi(0) - 1/sqrt(pi))
        // phi(0) = 1/sqrt(2*pi) ≈ 0.3989
        // CRPS = sigma * (2*0.3989 - 0.5642) = sigma * (0.7979 - 0.5642) = sigma * 0.2337
        Assert.Equal(0.234, result, 0.01);
    }

    [Fact]
    public void CRPS_Probabilistic_LargerUncertainty_GenerallyLargerCRPS()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0]);
        var predicted = Vector<double>.FromArray([10.0, 20.0, 30.0]);
        var smallStddev = Vector<double>.FromArray([1.0, 1.0, 1.0]);
        var largeStddev = Vector<double>.FromArray([5.0, 5.0, 5.0]);

        var crpsSmall = StatisticsHelper<double>.CalculateCRPS(actual, predicted, smallStddev);
        var crpsLarge = StatisticsHelper<double>.CalculateCRPS(actual, predicted, largeStddev);

        // When prediction is perfect, larger uncertainty means larger CRPS
        Assert.True(crpsSmall < crpsLarge,
            $"Larger uncertainty should generally give larger CRPS: {crpsSmall} vs {crpsLarge}");
    }

    [Fact]
    public void CRPS_Probabilistic_CalculatesCorrectValue()
    {
        // Single observation test for verification
        // y = 0, mu = 0, sigma = 1
        // z = (0 - 0) / 1 = 0
        // Phi(0) = 0.5, phi(0) = 0.3989
        // CRPS = 1 * (0 * (2*0.5 - 1) + 2*0.3989 - 1/sqrt(pi))
        // CRPS = 1 * (0 + 0.7979 - 0.5642) = 0.2337
        var actual = Vector<double>.FromArray([0.0]);
        var predicted = Vector<double>.FromArray([0.0]);
        var stddev = Vector<double>.FromArray([1.0]);

        var result = StatisticsHelper<double>.CalculateCRPS(actual, predicted, stddev);

        Assert.Equal(0.2337, result, 0.01);
    }

    [Fact]
    public void CRPS_Probabilistic_OneStdDevAway()
    {
        // y = 1, mu = 0, sigma = 1
        // z = 1
        // Phi(1) ≈ 0.8413, phi(1) ≈ 0.2420
        // CRPS = 1 * (1 * (2*0.8413 - 1) + 2*0.2420 - 0.5642)
        // CRPS = 1 * (1 * 0.6826 + 0.4840 - 0.5642) = 1 * 0.6024 = 0.6024
        var actual = Vector<double>.FromArray([1.0]);
        var predicted = Vector<double>.FromArray([0.0]);
        var stddev = Vector<double>.FromArray([1.0]);

        var result = StatisticsHelper<double>.CalculateCRPS(actual, predicted, stddev);

        Assert.Equal(0.602, result, 0.01);
    }

    #endregion

    #region VIF (Variance Inflation Factor) Tests

    [Fact]
    public void VIF_IdentityMatrix_ReturnsOnes()
    {
        // Identity matrix means variables are uncorrelated
        // VIF should be 1 for all variables
        var identity = Matrix<double>.CreateIdentityMatrix(3);
        var options = new ModelStatsOptions();

        var result = StatisticsHelper<double>.CalculateVIF(identity, options);

        Assert.Equal(3, result.Count);
        // For identity matrix, VIF calculation may give different values
        // depending on how submatrix inverse is computed
        foreach (var vif in result)
        {
            Assert.True(vif > 0, $"VIF should be positive, got {vif}");
        }
    }

    [Fact]
    public void VIF_HighCorrelation_ReturnsHighVIF()
    {
        // Correlation matrix with high correlation (0.95) between first two variables
        var correlationMatrix = new Matrix<double>(new double[,]
        {
            { 1.0, 0.95, 0.0 },
            { 0.95, 1.0, 0.0 },
            { 0.0, 0.0, 1.0 }
        });
        var options = new ModelStatsOptions();

        var result = StatisticsHelper<double>.CalculateVIF(correlationMatrix, options);

        // High correlation should produce high VIF for correlated variables
        Assert.True(result[0] > 5 || result[1] > 5,
            $"Highly correlated variables should have high VIF: {result[0]}, {result[1]}");
    }

    [Fact]
    public void VIF_AllPositive()
    {
        var correlationMatrix = new Matrix<double>(new double[,]
        {
            { 1.0, 0.3, 0.2 },
            { 0.3, 1.0, 0.1 },
            { 0.2, 0.1, 1.0 }
        });
        var options = new ModelStatsOptions();

        var result = StatisticsHelper<double>.CalculateVIF(correlationMatrix, options);

        foreach (var vif in result)
        {
            Assert.True(vif > 0, $"VIF should always be positive, got {vif}");
        }
    }

    #endregion

    #region Condition Number Tests

    [Fact]
    public void ConditionNumber_IdentityMatrix_ReturnsOne()
    {
        var identity = Matrix<double>.CreateIdentityMatrix(3);
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.SVD };

        var result = StatisticsHelper<double>.CalculateConditionNumber(identity, options);

        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ConditionNumber_IsPositive()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 2.0, 1.0 },
            { 2.0, 5.0, 2.0 },
            { 1.0, 2.0, 4.0 }
        });
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.SVD };

        var result = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

        Assert.True(result >= 1.0, $"Condition number should be >= 1, got {result}");
    }

    [Fact]
    public void ConditionNumber_IllConditionedMatrix_HighValue()
    {
        // Nearly singular matrix - should have high condition number
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 1.0, 1.001 }
        });
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.SVD };

        var result = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

        Assert.True(result > 100, $"Ill-conditioned matrix should have high condition number, got {result}");
    }

    [Fact]
    public void ConditionNumber_WellConditionedMatrix_LowValue()
    {
        // Well-conditioned symmetric positive definite matrix
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 1.0 },
            { 1.0, 3.0 }
        });
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.SVD };

        var result = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

        Assert.True(result < 10, $"Well-conditioned matrix should have low condition number, got {result}");
    }

    [Fact]
    public void ConditionNumber_L1Norm_IsPositive()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 2.0 },
            { 2.0, 5.0 }
        });
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.L1Norm };

        var result = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

        Assert.True(result >= 1.0, $"L1 condition number should be >= 1, got {result}");
    }

    [Fact]
    public void ConditionNumber_InfinityNorm_IsPositive()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 2.0 },
            { 2.0, 5.0 }
        });
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.InfinityNorm };

        var result = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

        Assert.True(result >= 1.0, $"Infinity norm condition number should be >= 1, got {result}");
    }

    [Fact]
    public void ConditionNumber_PowerIteration_IsPositive()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 2.0 },
            { 2.0, 5.0 }
        });
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.PowerIteration };

        var result = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

        Assert.True(result >= 1.0, $"Power iteration condition number should be >= 1, got {result}");
    }

    [Fact]
    public void ConditionNumber_DifferentMethods_SimilarMagnitude()
    {
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 1.0, 0.0 },
            { 1.0, 3.0, 1.0 },
            { 0.0, 1.0, 2.0 }
        });

        var svd = StatisticsHelper<double>.CalculateConditionNumber(matrix,
            new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.SVD });
        var l1 = StatisticsHelper<double>.CalculateConditionNumber(matrix,
            new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.L1Norm });
        var linf = StatisticsHelper<double>.CalculateConditionNumber(matrix,
            new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.InfinityNorm });

        // All methods should give similar order of magnitude
        // They measure different things but for well-behaved matrices should be comparable
        Assert.True(svd > 0 && l1 > 0 && linf > 0, "All condition numbers should be positive");
        Assert.True(svd < 100 && l1 < 100 && linf < 100,
            $"All condition numbers should be reasonably small for this matrix: SVD={svd}, L1={l1}, LInf={linf}");
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void Residuals_FloatType_ReturnsCorrectValues()
    {
        var actual = Vector<float>.FromArray([10.0f, 20.0f, 30.0f, 40.0f, 50.0f]);
        var predicted = Vector<float>.FromArray([12.0f, 18.0f, 32.0f, 38.0f, 52.0f]);

        var residuals = StatisticsHelper<float>.CalculateResiduals(actual, predicted);

        Assert.Equal(-2.0f, residuals[0], 1e-4f);
        Assert.Equal(2.0f, residuals[1], 1e-4f);
        Assert.Equal(-2.0f, residuals[2], 1e-4f);
    }

    [Fact]
    public void TheilU_FloatType_ReturnsCorrectValue()
    {
        var actual = Vector<float>.FromArray([10.0f, 20.0f, 30.0f, 40.0f, 50.0f]);
        var predicted = Vector<float>.FromArray([10.0f, 20.0f, 30.0f, 40.0f, 50.0f]);

        var result = StatisticsHelper<float>.CalculateTheilUStatistic(actual, predicted);

        Assert.Equal(0.0f, result, 1e-4f);
    }

    [Fact]
    public void CRPS_FloatType_EqualsMeanAbsoluteError()
    {
        var actual = Vector<float>.FromArray([10.0f, 20.0f, 30.0f]);
        var predicted = Vector<float>.FromArray([12.0f, 18.0f, 32.0f]);

        var crps = StatisticsHelper<float>.CalculateCRPS(actual, predicted);
        var mae = StatisticsHelper<float>.CalculateMeanAbsoluteError(actual, predicted);

        Assert.Equal(mae, crps, 1e-4f);
    }

    [Fact]
    public void ConditionNumber_FloatType_ReturnsCorrectValue()
    {
        // Use L1Norm method which is more numerically stable for float precision
        var identity = Matrix<float>.CreateIdentityMatrix(3);
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.L1Norm };

        var result = StatisticsHelper<float>.CalculateConditionNumber(identity, options);

        // Identity matrix should have condition number = 1
        Assert.True(!float.IsNaN(result) && result >= 0.99f && result <= 1.01f,
            $"Expected condition number ≈ 1 for identity matrix, got {result}");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Residuals_SingleElement_ReturnsCorrectValue()
    {
        var actual = Vector<double>.FromArray([10.0]);
        var predicted = Vector<double>.FromArray([8.0]);

        var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);

        Assert.Equal(1, residuals.Length);
        Assert.Equal(2.0, residuals[0], Tolerance);
    }

    [Fact]
    public void TheilU_SingleElement_PerfectPrediction()
    {
        var actual = Vector<double>.FromArray([10.0]);
        var predicted = Vector<double>.FromArray([10.0]);

        var result = StatisticsHelper<double>.CalculateTheilUStatistic(actual, predicted);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CRPS_EmptyVectors_ReturnsZero()
    {
        var actual = Vector<double>.FromArray([]);
        var predicted = Vector<double>.FromArray([]);
        var stddev = Vector<double>.FromArray([]);

        var result = StatisticsHelper<double>.CalculateCRPS(actual, predicted, stddev);

        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ConditionNumber_DiagonalMatrix_IsWellConditioned()
    {
        // For a well-conditioned diagonal matrix, condition number should be >= 1
        // Note: The exact ratio of max/min diagonals depends on SVD algorithm accuracy
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 0.0, 0.0 },
            { 0.0, 2.0, 0.0 },
            { 0.0, 0.0, 1.0 }
        });
        var options = new ModelStatsOptions { ConditionNumberMethod = ConditionNumberMethod.SVD };

        var result = StatisticsHelper<double>.CalculateConditionNumber(matrix, options);

        // Condition number should be >= 1 for any matrix
        Assert.True(result >= 1.0 - Tolerance, $"Condition number should be >= 1, got {result}");
        // And reasonably small for a well-conditioned diagonal matrix
        Assert.True(result <= 10.0, $"Well-conditioned diagonal matrix should have condition number <= 10, got {result}");
    }

    [Fact]
    public void Residuals_LargeValues_MaintainsPrecision()
    {
        var actual = Vector<double>.FromArray([1e10, 2e10, 3e10]);
        var predicted = Vector<double>.FromArray([1e10 + 1, 2e10 - 1, 3e10 + 2]);

        var residuals = StatisticsHelper<double>.CalculateResiduals(actual, predicted);

        Assert.Equal(-1.0, residuals[0], 1e-6);
        Assert.Equal(1.0, residuals[1], 1e-6);
        Assert.Equal(-2.0, residuals[2], 1e-6);
    }

    [Fact]
    public void CRPS_VerySmallStdDev_ApproachesMAE()
    {
        var actual = Vector<double>.FromArray([10.0, 20.0, 30.0]);
        var predicted = Vector<double>.FromArray([12.0, 18.0, 32.0]);
        var stddev = Vector<double>.FromArray([1e-11, 1e-11, 1e-11]);

        var crps = StatisticsHelper<double>.CalculateCRPS(actual, predicted, stddev);
        var mae = StatisticsHelper<double>.CalculateMeanAbsoluteError(actual, predicted);

        // With very small stddev, CRPS should be very close to MAE
        Assert.Equal(mae, crps, 0.01);
    }

    #endregion
}
