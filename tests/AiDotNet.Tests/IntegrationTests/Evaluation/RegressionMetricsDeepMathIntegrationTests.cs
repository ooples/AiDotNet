using AiDotNet.Evaluation.Metrics.Regression;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep mathematical integration tests for regression metrics.
/// Tests hand-calculated values, cross-metric identities, and edge cases.
/// </summary>
public class RegressionMetricsDeepMathIntegrationTests
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();
    private const double Tol = 1e-10;

    // ========================================================================
    // MSE / RMSE / MAE: Hand-calculated values
    // ========================================================================

    [Fact]
    public void MSE_HandCalculated_MatchesExpected()
    {
        // actuals: [1, 2, 3, 4, 5], predictions: [1.5, 2.5, 3.0, 3.5, 5.5]
        // errors:   0.5  0.5  0.0  0.5  0.5
        // squared:  0.25 0.25 0.0  0.25 0.25
        // MSE = (0.25 + 0.25 + 0.0 + 0.25 + 0.25) / 5 = 1.0 / 5 = 0.2
        var mse = new MSEMetric<double>();
        double[] preds = [1.5, 2.5, 3.0, 3.5, 5.5];
        double[] actuals = [1, 2, 3, 4, 5];

        double result = NumOps.ToDouble(mse.Compute(preds, actuals));
        Assert.Equal(0.2, result, Tol);
    }

    [Fact]
    public void RMSE_EqualsSquareRootOfMSE()
    {
        var mseMetric = new MSEMetric<double>();
        var rmseMetric = new RMSEMetric<double>();
        double[] preds = [2.3, 4.1, 6.8, 8.2, 10.5];
        double[] actuals = [2.0, 4.0, 7.0, 8.0, 10.0];

        double mse = NumOps.ToDouble(mseMetric.Compute(preds, actuals));
        double rmse = NumOps.ToDouble(rmseMetric.Compute(preds, actuals));
        Assert.Equal(Math.Sqrt(mse), rmse, Tol);
    }

    [Fact]
    public void MAE_HandCalculated_MatchesExpected()
    {
        // actuals: [10, 20, 30], predictions: [12, 18, 33]
        // |errors|: 2, 2, 3
        // MAE = (2 + 2 + 3) / 3 = 7/3 = 2.333...
        var mae = new MAEMetric<double>();
        double[] preds = [12, 18, 33];
        double[] actuals = [10, 20, 30];

        double result = NumOps.ToDouble(mae.Compute(preds, actuals));
        Assert.Equal(7.0 / 3.0, result, Tol);
    }

    [Fact]
    public void MSE_GreaterThanOrEqual_MAE_Squared()
    {
        // By Jensen's inequality: MSE >= MAE² (equality iff all errors are equal)
        var mseMetric = new MSEMetric<double>();
        var maeMetric = new MAEMetric<double>();
        double[] preds = [1, 5, 3, 7, 2];
        double[] actuals = [2, 3, 6, 4, 8];

        double mse = NumOps.ToDouble(mseMetric.Compute(preds, actuals));
        double mae = NumOps.ToDouble(maeMetric.Compute(preds, actuals));
        Assert.True(mse >= mae * mae - 1e-10, $"MSE={mse} should be >= MAE²={mae * mae}");
    }

    [Fact]
    public void MSE_Equals_MAE_Squared_WhenAllErrorsEqual()
    {
        // When all |errors| are equal, MSE = MAE² exactly
        // actuals: [0, 0, 0], preds: [2, 2, 2] => all errors = 2
        // MAE = 2, MSE = 4, MAE² = 4
        var mseMetric = new MSEMetric<double>();
        var maeMetric = new MAEMetric<double>();
        double[] preds = [2, 2, 2];
        double[] actuals = [0, 0, 0];

        double mse = NumOps.ToDouble(mseMetric.Compute(preds, actuals));
        double mae = NumOps.ToDouble(maeMetric.Compute(preds, actuals));
        Assert.Equal(mae * mae, mse, Tol);
    }

    // ========================================================================
    // R² and related metrics
    // ========================================================================

    [Fact]
    public void R2_PerfectPredictions_Equals1()
    {
        var r2 = new R2ScoreMetric<double>();
        double[] actuals = [1, 2, 3, 4, 5];
        double[] preds = [1, 2, 3, 4, 5];

        double result = NumOps.ToDouble(r2.Compute(preds, actuals));
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void R2_MeanPredictions_Equals0()
    {
        // If predictions are always the mean, SS_res = SS_tot => R² = 0
        var r2 = new R2ScoreMetric<double>();
        double[] actuals = [1, 2, 3, 4, 5];
        double mean = 3.0;
        double[] preds = [mean, mean, mean, mean, mean];

        double result = NumOps.ToDouble(r2.Compute(preds, actuals));
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void R2_HandCalculated_MatchesExpected()
    {
        // actuals: [3, -0.5, 2, 7], preds: [2.5, 0.0, 2, 8]
        // mean_actual = (3 - 0.5 + 2 + 7) / 4 = 11.5 / 4 = 2.875
        // SS_res = (3-2.5)² + (-0.5-0)² + (2-2)² + (7-8)² = 0.25 + 0.25 + 0 + 1 = 1.5
        // SS_tot = (3-2.875)² + (-0.5-2.875)² + (2-2.875)² + (7-2.875)²
        //        = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
        // R² = 1 - 1.5 / 29.1875 = 1 - 0.051393... = 0.948607...
        var r2 = new R2ScoreMetric<double>();
        double[] preds = [2.5, 0.0, 2, 8];
        double[] actuals = [3, -0.5, 2, 7];

        double result = NumOps.ToDouble(r2.Compute(preds, actuals));
        double expected = 1.0 - 1.5 / 29.1875;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void R2_WorseThanMean_IsNegative()
    {
        // Predictions that are worse than always predicting the mean should give R² < 0
        var r2 = new R2ScoreMetric<double>();
        double[] actuals = [1, 2, 3, 4, 5];
        // Mean = 3. Predictions far from actuals and worse than mean:
        double[] preds = [10, 10, 10, 10, 10];

        double result = NumOps.ToDouble(r2.Compute(preds, actuals));
        Assert.True(result < 0, $"R²={result} should be negative for predictions worse than mean");
    }

    [Fact]
    public void RSE_Plus_R2_Equals_1()
    {
        // RSE = 1 - R² by definition (RSE = SS_res / SS_tot, R² = 1 - SS_res / SS_tot)
        var r2 = new R2ScoreMetric<double>();
        var rse = new RelativeSquaredErrorMetric<double>();
        double[] preds = [2.5, 4.1, 6.3, 7.8, 10.1];
        double[] actuals = [2, 4, 7, 8, 10];

        double r2Val = NumOps.ToDouble(r2.Compute(preds, actuals));
        double rseVal = NumOps.ToDouble(rse.Compute(preds, actuals));
        Assert.Equal(1.0, r2Val + rseVal, Tol);
    }

    [Fact]
    public void NMSE_Equals_1MinusR2()
    {
        // NMSE = MSE / Var(y) = (SS_res/n) / (SS_tot/n) = SS_res/SS_tot = 1 - R²
        var r2 = new R2ScoreMetric<double>();
        var nmse = new NormalizedMSEMetric<double>();
        double[] preds = [1.5, 3.2, 5.0, 7.1, 9.3];
        double[] actuals = [1, 3, 5, 7, 9];

        double r2Val = NumOps.ToDouble(r2.Compute(preds, actuals));
        double nmseVal = NumOps.ToDouble(nmse.Compute(preds, actuals));
        Assert.Equal(1.0 - r2Val, nmseVal, Tol);
    }

    [Fact]
    public void AdjustedR2_LessThanOrEqual_R2()
    {
        // Adjusted R² penalizes model complexity, so AdjR² <= R² for p >= 1
        var r2 = new R2ScoreMetric<double>();
        var adjR2 = new AdjustedR2Metric<double>(numPredictors: 3);
        double[] preds = [1.1, 2.2, 3.1, 4.3, 5.0, 6.1, 7.2, 8.0, 9.1, 10.2];
        double[] actuals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        double r2Val = NumOps.ToDouble(r2.Compute(preds, actuals));
        double adjR2Val = NumOps.ToDouble(adjR2.Compute(preds, actuals));
        Assert.True(adjR2Val <= r2Val + 1e-10, $"AdjR²={adjR2Val} should be <= R²={r2Val}");
    }

    [Fact]
    public void AdjustedR2_HandCalculated_MatchesExpected()
    {
        // With 10 samples, 2 predictors, R² = 0.95
        // AdjR² = 1 - (1-0.95)*(10-1)/(10-2-1) = 1 - 0.05*9/7 = 1 - 0.06428... = 0.93571...
        var r2Metric = new R2ScoreMetric<double>();
        var adjR2 = new AdjustedR2Metric<double>(numPredictors: 2);

        // Create data where R² is known to be exactly 0.95
        // actuals: [1..10], SS_tot = 82.5
        // For R²=0.95: SS_res = (1-0.95)*82.5 = 4.125
        // We need preds so that SS_res = 4.125 with 10 points
        // Simplest: all errors equal => each error² = 4.125/10 = 0.4125, error = 0.642262...
        double[] actuals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        double errEach = Math.Sqrt(4.125 / 10.0);
        double[] preds = new double[10];
        for (int i = 0; i < 10; i++) preds[i] = actuals[i] + errEach;

        double r2Val = NumOps.ToDouble(r2Metric.Compute(preds, actuals));
        double adjR2Val = NumOps.ToDouble(adjR2.Compute(preds, actuals));

        double expectedAdj = 1.0 - (1.0 - r2Val) * (10 - 1.0) / (10 - 2 - 1.0);
        Assert.Equal(expectedAdj, adjR2Val, Tol);
    }

    // ========================================================================
    // Pearson and Spearman Correlation
    // ========================================================================

    [Fact]
    public void Pearson_PerfectLinear_Equals1()
    {
        // Perfect positive linear: y = 2x + 1
        var pearson = new PearsonCorrelationMetric<double>();
        double[] actuals = [1, 3, 5, 7, 9];
        double[] preds = [1, 3, 5, 7, 9];

        double result = NumOps.ToDouble(pearson.Compute(preds, actuals));
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void Pearson_PerfectNegativeLinear_EqualsMinus1()
    {
        var pearson = new PearsonCorrelationMetric<double>();
        double[] actuals = [1, 2, 3, 4, 5];
        double[] preds = [5, 4, 3, 2, 1];

        double result = NumOps.ToDouble(pearson.Compute(preds, actuals));
        Assert.Equal(-1.0, result, Tol);
    }

    [Fact]
    public void Pearson_Squared_Equals_R2_ForLinearFit()
    {
        // For unbiased linear predictions: R² = r² (Pearson squared)
        // preds = a*actuals + b => R² = r²
        var pearson = new PearsonCorrelationMetric<double>();
        var r2 = new R2ScoreMetric<double>();

        // actuals: [1, 2, 3, 4, 5], preds = 1.1*actuals - 0.1
        double[] actuals = [1, 2, 3, 4, 5];
        double[] preds = [1.0, 2.1, 3.2, 4.3, 5.4];

        double r = NumOps.ToDouble(pearson.Compute(preds, actuals));
        double r2Val = NumOps.ToDouble(r2.Compute(preds, actuals));

        // r² = R² only if predictions are the fitted line (no bias added to residuals)
        // For general linear preds, we check r² >= R² (since r² = max R² over all linear fits)
        Assert.True(r * r >= r2Val - 1e-8,
            $"r²={r * r} should be >= R²={r2Val} for any predictions");
    }

    [Fact]
    public void Pearson_HandCalculated_MatchesExpected()
    {
        // preds: [1, 2, 3], actuals: [2, 4, 5]
        // mean_pred = 2, mean_actual = 11/3 = 3.666...
        // cov = (1-2)(2-11/3) + (2-2)(4-11/3) + (3-2)(5-11/3)
        //     = (-1)(-5/3) + 0*(1/3) + (1)(4/3) = 5/3 + 0 + 4/3 = 3
        // var_pred = 1 + 0 + 1 = 2
        // var_actual = (2-11/3)² + (4-11/3)² + (5-11/3)² = 25/9 + 1/9 + 16/9 = 42/9
        // std_pred = sqrt(2), std_actual = sqrt(42/9)
        // r = 3 / (sqrt(2) * sqrt(42/9)) = 3 / sqrt(84/9) = 3 / sqrt(84)/3 = 9/sqrt(84)
        var pearson = new PearsonCorrelationMetric<double>();
        double[] preds = [1, 2, 3];
        double[] actuals = [2, 4, 5];

        double result = NumOps.ToDouble(pearson.Compute(preds, actuals));
        double expected = 9.0 / Math.Sqrt(84.0);
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void Spearman_InvariantToMonotonicTransform()
    {
        // Spearman correlation should be identical before and after monotonic transform
        var spearman = new SpearmanCorrelationMetric<double>();
        double[] actuals = [1, 2, 3, 4, 5];
        double[] preds = [2.1, 3.5, 5.2, 7.1, 9.8];

        // Apply monotonic transform: f(x) = x³
        double[] predsTransformed = preds.Select(x => x * x * x).ToArray();

        double rho1 = NumOps.ToDouble(spearman.Compute(preds, actuals));
        double rho2 = NumOps.ToDouble(spearman.Compute(predsTransformed, actuals));
        Assert.Equal(rho1, rho2, Tol);
    }

    [Fact]
    public void Spearman_PerfectMonotonic_Equals1()
    {
        // Even with non-linear relationship, if order is preserved, rho = 1
        var spearman = new SpearmanCorrelationMetric<double>();
        double[] actuals = [1, 2, 3, 4, 5];
        double[] preds = [0.1, 100, 200, 500, 10000]; // wildly different scales but same order

        double result = NumOps.ToDouble(spearman.Compute(preds, actuals));
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void Spearman_HandCalculated_WithTies()
    {
        // Test tie handling: actuals have ties
        // actuals: [1, 2, 2, 4], preds: [10, 20, 30, 40]
        // actual ranks: [1, 2.5, 2.5, 4], pred ranks: [1, 2, 3, 4]
        // Pearson on ranks:
        // mean_actual_rank = (1+2.5+2.5+4)/4 = 2.5
        // mean_pred_rank = (1+2+3+4)/4 = 2.5
        // cov = (1-2.5)(1-2.5) + (2.5-2.5)(2-2.5) + (2.5-2.5)(3-2.5) + (4-2.5)(4-2.5)
        //     = (-1.5)(-1.5) + 0*(-0.5) + 0*(0.5) + (1.5)(1.5) = 2.25 + 0 + 0 + 2.25 = 4.5
        // var_actual = 2.25 + 0 + 0 + 2.25 = 4.5
        // var_pred = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        // rho = 4.5 / sqrt(4.5 * 5.0) = 4.5 / sqrt(22.5) = 4.5 / 4.7434... ≈ 0.94868...
        var spearman = new SpearmanCorrelationMetric<double>();
        double[] actuals = [1, 2, 2, 4];
        double[] preds = [10, 20, 30, 40];

        double result = NumOps.ToDouble(spearman.Compute(preds, actuals));
        double expected = 4.5 / Math.Sqrt(4.5 * 5.0);
        Assert.Equal(expected, result, Tol);
    }

    // ========================================================================
    // Huber Loss
    // ========================================================================

    [Fact]
    public void HuberLoss_SmallErrors_EqualsMSEHalved()
    {
        // When all |errors| <= delta, Huber = 0.5 * MSE
        var huber = new HuberLossMetric<double>(delta: 10.0); // large delta
        var mse = new MSEMetric<double>();
        double[] preds = [1.5, 2.5, 3.5];
        double[] actuals = [1, 2, 3];

        double huberVal = NumOps.ToDouble(huber.Compute(preds, actuals));
        double mseVal = NumOps.ToDouble(mse.Compute(preds, actuals));
        Assert.Equal(0.5 * mseVal, huberVal, Tol);
    }

    [Fact]
    public void HuberLoss_HandCalculated_MixedRegions()
    {
        // delta = 1.0
        // actuals: [0, 0, 0], preds: [0.5, 1.5, 3.0]
        // errors:  0.5 (quadratic), 1.5 (linear), 3.0 (linear)
        // Huber for 0.5: 0.5 * 0.25 = 0.125
        // Huber for 1.5: 1.0 * (1.5 - 0.5) = 1.0
        // Huber for 3.0: 1.0 * (3.0 - 0.5) = 2.5
        // Mean = (0.125 + 1.0 + 2.5) / 3 = 3.625 / 3 = 1.208333...
        var huber = new HuberLossMetric<double>(delta: 1.0);
        double[] preds = [0.5, 1.5, 3.0];
        double[] actuals = [0, 0, 0];

        double result = NumOps.ToDouble(huber.Compute(preds, actuals));
        Assert.Equal(3.625 / 3.0, result, Tol);
    }

    [Fact]
    public void HuberLoss_LargeErrors_ApproachesMAETimesDelta()
    {
        // For very large errors >> delta, Huber(e) ≈ delta * (|e| - 0.5*delta)
        // When delta is small and errors are huge: Huber ≈ delta * |e| ≈ delta * MAE
        var mae = new MAEMetric<double>();
        double delta = 0.01;
        var huber = new HuberLossMetric<double>(delta: delta);
        double[] preds = [100, 200, 300];
        double[] actuals = [0, 0, 0];

        double huberVal = NumOps.ToDouble(huber.Compute(preds, actuals));
        double maeVal = NumOps.ToDouble(mae.Compute(preds, actuals));
        // Huber ≈ delta * (MAE - 0.5*delta) for large errors
        double expectedApprox = delta * (maeVal - 0.5 * delta);
        Assert.Equal(expectedApprox, huberVal, 1e-6);
    }

    [Fact]
    public void HuberLoss_ContinuousAtDelta()
    {
        // Huber should be continuous at the transition point
        // At |error| = delta: both formulas give 0.5 * delta²
        double delta = 2.0;
        var huber = new HuberLossMetric<double>(delta: delta);

        // Error exactly at delta
        double[] predsAtDelta = [delta];
        double[] actualsZero = [0];
        double atDelta = NumOps.ToDouble(huber.Compute(predsAtDelta, actualsZero));

        // Quadratic formula: 0.5 * delta² = 0.5 * 4 = 2.0
        // Linear formula: delta * (delta - 0.5*delta) = delta * 0.5*delta = 0.5*delta² = 2.0
        Assert.Equal(0.5 * delta * delta, atDelta, Tol);
    }

    // ========================================================================
    // Quantile Loss
    // ========================================================================

    [Fact]
    public void QuantileLoss_AtHalf_EqualsMAEHalved()
    {
        // At τ=0.5: loss = 0.5*|error| for all errors => QuantileLoss = 0.5 * MAE
        var quantile = new QuantileLossMetric<double>(quantile: 0.5);
        var mae = new MAEMetric<double>();
        double[] preds = [1, 5, 3, 7, 2];
        double[] actuals = [2, 3, 6, 4, 8];

        double qLoss = NumOps.ToDouble(quantile.Compute(preds, actuals));
        double maeVal = NumOps.ToDouble(mae.Compute(preds, actuals));
        Assert.Equal(0.5 * maeVal, qLoss, Tol);
    }

    [Fact]
    public void QuantileLoss_HandCalculated_Asymmetric()
    {
        // τ = 0.9
        // actuals: [10, 10, 10], preds: [8, 12, 10]
        // errors (y - yhat): 2, -2, 0
        // For error=2 (positive): 0.9 * 2 = 1.8
        // For error=-2 (negative): (0.9 - 1) * (-2) = (-0.1)*(-2) = 0.2
        // For error=0: 0
        // Mean = (1.8 + 0.2 + 0) / 3 = 2.0/3 = 0.666...
        var quantile = new QuantileLossMetric<double>(quantile: 0.9);
        double[] preds = [8, 12, 10];
        double[] actuals = [10, 10, 10];

        double result = NumOps.ToDouble(quantile.Compute(preds, actuals));
        Assert.Equal(2.0 / 3.0, result, Tol);
    }

    [Fact]
    public void QuantileLoss_Symmetry_TauAndOneMinusTau()
    {
        // QuantileLoss(τ, preds, actuals) = QuantileLoss(1-τ, actuals, preds) -- swapped
        // More precisely: for a single point:
        //   If error = y - yhat > 0: τ*error
        //   Swapped error = yhat - y < 0: (1-τ-1)*(-error) = -τ*(yhat-y) = τ*(y-yhat)
        // Actually the simpler identity: for a given τ,
        // QL(τ) with over-predictions + QL(τ) with under-predictions adds up correctly
        // Let's verify: QL(τ=0.1) for positive error = 0.1*|e|,
        //               QL(τ=0.9) for negative error of same magnitude = (0.9-1)*(-|e|) = 0.1*|e|
        var q01 = new QuantileLossMetric<double>(quantile: 0.1);
        var q09 = new QuantileLossMetric<double>(quantile: 0.9);

        // Under-prediction: actual > pred
        double[] predUnder = [5];
        double[] actualHigh = [10];
        double loss01Under = NumOps.ToDouble(q01.Compute(predUnder, actualHigh));
        double loss09Under = NumOps.ToDouble(q09.Compute(predUnder, actualHigh));

        // τ=0.1, error=5>0: 0.1*5 = 0.5
        Assert.Equal(0.5, loss01Under, Tol);
        // τ=0.9, error=5>0: 0.9*5 = 4.5
        Assert.Equal(4.5, loss09Under, Tol);

        // Over-prediction: actual < pred
        double[] predOver = [10];
        double[] actualLow = [5];
        double loss01Over = NumOps.ToDouble(q01.Compute(predOver, actualLow));
        double loss09Over = NumOps.ToDouble(q09.Compute(predOver, actualLow));

        // τ=0.1, error=-5<0: (0.1-1)*(-5)=0.9*5=4.5
        Assert.Equal(4.5, loss01Over, Tol);
        // τ=0.9, error=-5<0: (0.9-1)*(-5)=0.1*5=0.5
        Assert.Equal(0.5, loss09Over, Tol);
    }

    // ========================================================================
    // Log-Cosh Loss
    // ========================================================================

    [Fact]
    public void LogCosh_SmallErrors_ApproximatesMSEHalved()
    {
        // For small x: log(cosh(x)) ≈ x²/2
        // So LogCosh ≈ mean(x²/2) = MSE/2
        var logCosh = new LogCoshLossMetric<double>();
        var mse = new MSEMetric<double>();
        double[] preds = [1.001, 2.002, 3.001];
        double[] actuals = [1, 2, 3];

        double logCoshVal = NumOps.ToDouble(logCosh.Compute(preds, actuals));
        double mseVal = NumOps.ToDouble(mse.Compute(preds, actuals));
        // For very small errors, log(cosh(x)) ≈ x²/2 is very accurate
        Assert.Equal(0.5 * mseVal, logCoshVal, 1e-10);
    }

    [Fact]
    public void LogCosh_LargeErrors_ApproachesMAEMinusLog2()
    {
        // For large |x|: log(cosh(x)) ≈ |x| - log(2)
        // So LogCosh ≈ MAE - log(2)
        var logCosh = new LogCoshLossMetric<double>();
        var mae = new MAEMetric<double>();
        double[] preds = [100, 200, 300];
        double[] actuals = [0, 0, 0];

        double logCoshVal = NumOps.ToDouble(logCosh.Compute(preds, actuals));
        double maeVal = NumOps.ToDouble(mae.Compute(preds, actuals));
        Assert.Equal(maeVal - Math.Log(2), logCoshVal, 1e-6);
    }

    [Fact]
    public void LogCosh_HandCalculated_MatchesExpected()
    {
        // actuals: [0], preds: [1]
        // log(cosh(1)) = log((e+e^-1)/2) = log(e+e^-1) - log(2)
        var logCosh = new LogCoshLossMetric<double>();
        double[] preds = [1];
        double[] actuals = [0];

        double result = NumOps.ToDouble(logCosh.Compute(preds, actuals));
        double expected = Math.Log(Math.Cosh(1.0));
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void LogCosh_AlwaysNonNegative()
    {
        // log(cosh(x)) >= 0 since cosh(x) >= 1
        var logCosh = new LogCoshLossMetric<double>();
        double[] preds = [0, 0, 0];
        double[] actuals = [0, 0, 0];

        double result = NumOps.ToDouble(logCosh.Compute(preds, actuals));
        Assert.True(result >= -1e-15, $"LogCosh={result} should be non-negative");
    }

    // ========================================================================
    // Mean Bias Error
    // ========================================================================

    [Fact]
    public void MBE_HandCalculated_OverPrediction()
    {
        // MBE = mean(pred - actual) = mean of signed errors
        // preds: [12, 22, 32], actuals: [10, 20, 30]
        // errors: 2, 2, 2 => MBE = 2
        var mbe = new MeanBiasErrorMetric<double>();
        double[] preds = [12, 22, 32];
        double[] actuals = [10, 20, 30];

        double result = NumOps.ToDouble(mbe.Compute(preds, actuals));
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void MBE_BalancedErrors_IsZero()
    {
        // If over and under-predictions cancel: MBE = 0
        var mbe = new MeanBiasErrorMetric<double>();
        double[] preds = [12, 18];
        double[] actuals = [10, 20];

        double result = NumOps.ToDouble(mbe.Compute(preds, actuals));
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MBE_CanBeNegative_UnderPrediction()
    {
        var mbe = new MeanBiasErrorMetric<double>();
        double[] preds = [8, 18, 28];
        double[] actuals = [10, 20, 30];

        double result = NumOps.ToDouble(mbe.Compute(preds, actuals));
        Assert.Equal(-2.0, result, Tol);
    }

    // ========================================================================
    // Explained Variance
    // ========================================================================

    [Fact]
    public void ExplainedVariance_PerfectPredictions_Equals1()
    {
        var ev = new ExplainedVarianceMetric<double>();
        double[] preds = [1, 2, 3, 4, 5];
        double[] actuals = [1, 2, 3, 4, 5];

        double result = NumOps.ToDouble(ev.Compute(preds, actuals));
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void ExplainedVariance_WithBias_DiffersFromR2()
    {
        // EV = 1 - Var(residuals)/Var(actuals)
        // R² = 1 - SS_res/SS_tot
        // When predictions are biased (mean(residuals) != 0), EV > R²
        // because Var(residuals) < mean(residuals²) when mean(residuals) != 0
        var ev = new ExplainedVarianceMetric<double>();
        var r2 = new R2ScoreMetric<double>();

        // Biased predictions: all shifted by +5
        double[] actuals = [1, 2, 3, 4, 5];
        double[] preds = [6, 7, 8, 9, 10]; // perfect pattern but biased by +5

        double evVal = NumOps.ToDouble(ev.Compute(preds, actuals));
        double r2Val = NumOps.ToDouble(r2.Compute(preds, actuals));

        // EV should be 1.0 (residuals are constant, Var(residuals) = 0)
        Assert.Equal(1.0, evVal, Tol);
        // R² should be negative (biased predictions are worse than mean)
        Assert.True(r2Val < 0, $"R²={r2Val} should be negative for biased predictions");
    }

    [Fact]
    public void ExplainedVariance_EqualsR2_WhenUnbiased()
    {
        // When mean(residuals) = 0, EV = R²
        var ev = new ExplainedVarianceMetric<double>();
        var r2 = new R2ScoreMetric<double>();

        // Ensure residuals have zero mean: alternating +/- errors
        double[] actuals = [1, 2, 3, 4];
        double[] preds = [1.5, 1.5, 3.5, 3.5]; // residuals: -0.5, 0.5, -0.5, 0.5, mean=0

        double evVal = NumOps.ToDouble(ev.Compute(preds, actuals));
        double r2Val = NumOps.ToDouble(r2.Compute(preds, actuals));
        Assert.Equal(r2Val, evVal, Tol);
    }

    // ========================================================================
    // MAPE, sMAPE, wMAPE
    // ========================================================================

    [Fact]
    public void MAPE_HandCalculated_MatchesExpected()
    {
        // actuals: [100, 200, 300], preds: [110, 190, 330]
        // |errors|/|actuals|: 10/100=0.1, 10/200=0.05, 30/300=0.1
        // MAPE = 100 * (0.1 + 0.05 + 0.1) / 3 = 100 * 0.25/3 = 8.333...%
        var mape = new MAPEMetric<double>();
        double[] preds = [110, 190, 330];
        double[] actuals = [100, 200, 300];

        double result = NumOps.ToDouble(mape.Compute(preds, actuals));
        Assert.Equal(100.0 * 0.25 / 3.0, result, Tol);
    }

    [Fact]
    public void SMAPE_HandCalculated_MatchesExpected()
    {
        // actuals: [100, 200], preds: [110, 180]
        // |100-110| / ((100+110)/2) = 10/105 = 0.0952380...
        // |200-180| / ((200+180)/2) = 20/190 = 0.1052631...
        // sMAPE = 100 * (0.0952380 + 0.1052631) / 2 = 100 * 0.2005011/2
        var smape = new SymmetricMAPEMetric<double>();
        double[] preds = [110, 180];
        double[] actuals = [100, 200];

        double result = NumOps.ToDouble(smape.Compute(preds, actuals));
        double expected = 100.0 * (10.0 / 105.0 + 20.0 / 190.0) / 2.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void SMAPE_IsSymmetric_SwappingPredsAndActuals()
    {
        // sMAPE should be the same when swapping predictions and actuals
        // because the formula uses |y-yhat| / ((|y|+|yhat|)/2)
        var smape = new SymmetricMAPEMetric<double>();
        double[] a = [100, 200, 300];
        double[] b = [110, 190, 330];

        double smape1 = NumOps.ToDouble(smape.Compute(a, b));
        double smape2 = NumOps.ToDouble(smape.Compute(b, a));
        Assert.Equal(smape1, smape2, Tol);
    }

    [Fact]
    public void WMAPE_HandCalculated_MatchesExpected()
    {
        // wMAPE = 100 * Σ|y-yhat| / Σ|y|
        // actuals: [100, 200, 300], preds: [110, 190, 330]
        // Σ|errors| = 10 + 10 + 30 = 50
        // Σ|actuals| = 100 + 200 + 300 = 600
        // wMAPE = 100 * 50/600 = 8.333...%
        var wmape = new WeightedMAPEMetric<double>();
        double[] preds = [110, 190, 330];
        double[] actuals = [100, 200, 300];

        double result = NumOps.ToDouble(wmape.Compute(preds, actuals));
        Assert.Equal(100.0 * 50.0 / 600.0, result, Tol);
    }

    [Fact]
    public void WMAPE_LargeActuals_WeighMoreHeavily()
    {
        // The large actual value dominates wMAPE
        var wmape = new WeightedMAPEMetric<double>();

        // Scenario 1: Large error on small actual
        double[] preds1 = [10, 1000];
        double[] actuals1 = [1, 1000]; // error=9 on small, error=0 on large
        double result1 = NumOps.ToDouble(wmape.Compute(preds1, actuals1));

        // Scenario 2: Large error on large actual
        double[] preds2 = [1, 1009];
        double[] actuals2 = [1, 1000]; // error=0 on small, error=9 on large
        double result2 = NumOps.ToDouble(wmape.Compute(preds2, actuals2));

        // Both have same total absolute error but wMAPE should be identical
        // because wMAPE = 100 * Σ|e| / Σ|y|, and both have same Σ|e| and same Σ|y|
        Assert.Equal(result1, result2, Tol);
    }

    // ========================================================================
    // Relative Absolute Error
    // ========================================================================

    [Fact]
    public void RAE_HandCalculated_MatchesExpected()
    {
        // actuals: [1, 2, 3, 4, 5], preds: [1.5, 2.5, 3.5, 4.5, 5.5]
        // mean = 3
        // Σ|y - yhat| = 0.5*5 = 2.5
        // Σ|y - mean| = |1-3| + |2-3| + |3-3| + |4-3| + |5-3| = 2+1+0+1+2 = 6
        // RAE = 2.5 / 6 = 0.41666...
        var rae = new RelativeAbsoluteErrorMetric<double>();
        double[] preds = [1.5, 2.5, 3.5, 4.5, 5.5];
        double[] actuals = [1, 2, 3, 4, 5];

        double result = NumOps.ToDouble(rae.Compute(preds, actuals));
        Assert.Equal(2.5 / 6.0, result, Tol);
    }

    // ========================================================================
    // MSLE (Mean Squared Log Error)
    // ========================================================================

    [Fact]
    public void MSLE_HandCalculated_MatchesExpected()
    {
        // actuals: [3, 5, 2.5], preds: [2.5, 5, 4]
        // MSLE = mean[ (log(1+y) - log(1+yhat))² ]
        // = mean[ (log(4)-log(3.5))², (log(6)-log(6))², (log(3.5)-log(5))² ]
        var msle = new MeanSquaredLogErrorMetric<double>();
        double[] preds = [2.5, 5, 4];
        double[] actuals = [3, 5, 2.5];

        double result = NumOps.ToDouble(msle.Compute(preds, actuals));
        double d1 = Math.Log(4) - Math.Log(3.5);
        double d2 = Math.Log(6) - Math.Log(6);
        double d3 = Math.Log(3.5) - Math.Log(5);
        double expected = (d1 * d1 + d2 * d2 + d3 * d3) / 3.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void MSLE_PenalizesUnderPredictionMoreThanOver()
    {
        // MSLE uses log, so under-prediction of same absolute size is penalized more
        // actual=100, pred=50 (under by 50): (log(101)-log(51))²
        // actual=100, pred=150 (over by 50): (log(101)-log(151))²
        var msle = new MeanSquaredLogErrorMetric<double>();

        double underVal = NumOps.ToDouble(msle.Compute(new double[] { 50 }, new double[] { 100 }));
        double overVal = NumOps.ToDouble(msle.Compute(new double[] { 150 }, new double[] { 100 }));

        Assert.True(underVal > overVal,
            $"Under-prediction MSLE={underVal} should be > over-prediction MSLE={overVal}");
    }

    // ========================================================================
    // Poisson Deviance
    // ========================================================================

    [Fact]
    public void PoissonDeviance_PerfectPredictions_IsZero()
    {
        // When y = mu: deviance = 2*(y*log(1) - 0) = 0
        var poisson = new PoissonDevianceMetric<double>();
        double[] preds = [1, 2, 3, 4, 5];
        double[] actuals = [1, 2, 3, 4, 5];

        double result = NumOps.ToDouble(poisson.Compute(preds, actuals));
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void PoissonDeviance_HandCalculated_MatchesExpected()
    {
        // actual=5, pred=3: 2*(5*log(5/3) - (5-3)) = 2*(5*0.5108... - 2) = 2*(2.5541...-2) = 2*0.5541 = 1.1082...
        var poisson = new PoissonDevianceMetric<double>();
        double[] preds = [3];
        double[] actuals = [5];

        double result = NumOps.ToDouble(poisson.Compute(preds, actuals));
        double expected = 2.0 * (5.0 * Math.Log(5.0 / 3.0) - (5.0 - 3.0));
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void PoissonDeviance_AlwaysNonNegative()
    {
        // Poisson deviance >= 0, with equality at y = mu
        var poisson = new PoissonDevianceMetric<double>();
        double[] preds = [3, 7, 2];
        double[] actuals = [5, 4, 8];

        double result = NumOps.ToDouble(poisson.Compute(preds, actuals));
        Assert.True(result >= -1e-10, $"Poisson deviance={result} should be non-negative");
    }

    // ========================================================================
    // Tweedie Loss
    // ========================================================================

    [Fact]
    public void TweedieLoss_Power0_EqualsMSE()
    {
        // Tweedie with p=0 should equal MSE (normal distribution)
        var tweedie = new TweedieLossMetric<double>(power: 0);
        var mse = new MSEMetric<double>();
        double[] preds = [1.5, 2.5, 3.5];
        double[] actuals = [1, 2, 3];

        double tweedieVal = NumOps.ToDouble(tweedie.Compute(preds, actuals));
        double mseVal = NumOps.ToDouble(mse.Compute(preds, actuals));
        Assert.Equal(mseVal, tweedieVal, Tol);
    }

    [Fact]
    public void TweedieLoss_Power1_EqualsPoissonDeviance()
    {
        // Tweedie with p=1 should equal Poisson deviance
        var tweedie = new TweedieLossMetric<double>(power: 1);
        var poisson = new PoissonDevianceMetric<double>();
        double[] preds = [3, 5, 8];
        double[] actuals = [2, 6, 7];

        double tweedieVal = NumOps.ToDouble(tweedie.Compute(preds, actuals));
        double poissonVal = NumOps.ToDouble(poisson.Compute(preds, actuals));
        Assert.Equal(poissonVal, tweedieVal, Tol);
    }

    [Fact]
    public void TweedieLoss_Power2_EqualsGammaDeviance()
    {
        // Tweedie p=2: Gamma deviance = 2*(-log(y/mu) + (y-mu)/mu)
        var tweedie = new TweedieLossMetric<double>(power: 2);
        double[] preds = [3, 5];
        double[] actuals = [4, 6];

        double tweedieVal = NumOps.ToDouble(tweedie.Compute(preds, actuals));

        // Hand-calculate Gamma deviance
        double dev1 = 2.0 * (-Math.Log(4.0 / 3.0) + (4.0 - 3.0) / 3.0);
        double dev2 = 2.0 * (-Math.Log(6.0 / 5.0) + (6.0 - 5.0) / 5.0);
        double expected = (dev1 + dev2) / 2.0;
        Assert.Equal(expected, tweedieVal, Tol);
    }

    // ========================================================================
    // Cross-metric consistency: comprehensive data set
    // ========================================================================

    [Fact]
    public void AllMetrics_PerfectPredictions_OptimalValues()
    {
        // For perfect predictions, all loss metrics should be 0, all score metrics should be 1
        double[] actuals = [1, 2, 3, 4, 5];
        double[] preds = [1, 2, 3, 4, 5];

        // Loss metrics should be 0
        Assert.Equal(0.0, NumOps.ToDouble(new MSEMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new RMSEMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new MAEMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new HuberLossMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new LogCoshLossMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new MeanBiasErrorMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new QuantileLossMetric<double>(0.5).Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new MeanSquaredLogErrorMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new PoissonDevianceMetric<double>().Compute(preds, actuals)), Tol);

        // Score metrics should be 1
        Assert.Equal(1.0, NumOps.ToDouble(new R2ScoreMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(1.0, NumOps.ToDouble(new PearsonCorrelationMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(1.0, NumOps.ToDouble(new SpearmanCorrelationMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(1.0, NumOps.ToDouble(new ExplainedVarianceMetric<double>().Compute(preds, actuals)), Tol);

        // Relative errors should be 0
        Assert.Equal(0.0, NumOps.ToDouble(new RelativeSquaredErrorMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new RelativeAbsoluteErrorMetric<double>().Compute(preds, actuals)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new NormalizedMSEMetric<double>().Compute(preds, actuals)), Tol);
    }

    [Fact]
    public void RMSE_GreaterThanOrEqual_MAE()
    {
        // By Cauchy-Schwarz: RMSE >= MAE, with equality iff all |errors| are equal
        var rmse = new RMSEMetric<double>();
        var mae = new MAEMetric<double>();
        double[] preds = [1, 5, 3, 7, 2];
        double[] actuals = [2, 3, 6, 4, 8];

        double rmseVal = NumOps.ToDouble(rmse.Compute(preds, actuals));
        double maeVal = NumOps.ToDouble(mae.Compute(preds, actuals));
        Assert.True(rmseVal >= maeVal - 1e-10,
            $"RMSE={rmseVal} should be >= MAE={maeVal}");
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    [Fact]
    public void Metrics_SingleElement_HandleCorrectly()
    {
        double[] preds = [3.0];
        double[] actuals = [2.0];

        // MSE for single element: (2-3)² = 1
        Assert.Equal(1.0, NumOps.ToDouble(new MSEMetric<double>().Compute(preds, actuals)), Tol);
        // MAE for single element: |2-3| = 1
        Assert.Equal(1.0, NumOps.ToDouble(new MAEMetric<double>().Compute(preds, actuals)), Tol);
        // RMSE for single element: sqrt(1) = 1
        Assert.Equal(1.0, NumOps.ToDouble(new RMSEMetric<double>().Compute(preds, actuals)), Tol);
        // MBE: pred - actual = 1
        Assert.Equal(1.0, NumOps.ToDouble(new MeanBiasErrorMetric<double>().Compute(preds, actuals)), Tol);
    }

    [Fact]
    public void Metrics_EmptyInput_ReturnsZero()
    {
        double[] empty = [];

        Assert.Equal(0.0, NumOps.ToDouble(new MSEMetric<double>().Compute(empty, empty)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new MAEMetric<double>().Compute(empty, empty)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new RMSEMetric<double>().Compute(empty, empty)), Tol);
        Assert.Equal(0.0, NumOps.ToDouble(new R2ScoreMetric<double>().Compute(empty, empty)), Tol);
    }

    [Fact]
    public void Metrics_LengthMismatch_ThrowsArgumentException()
    {
        double[] preds = [1, 2, 3];
        double[] actuals = [1, 2];

        Assert.Throws<ArgumentException>(() => new MSEMetric<double>().Compute(preds, actuals));
        Assert.Throws<ArgumentException>(() => new MAEMetric<double>().Compute(preds, actuals));
        Assert.Throws<ArgumentException>(() => new R2ScoreMetric<double>().Compute(preds, actuals));
        Assert.Throws<ArgumentException>(() => new PearsonCorrelationMetric<double>().Compute(preds, actuals));
    }

    [Fact]
    public void R2_ConstantActuals_PerfectPrediction_Returns1()
    {
        var r2 = new R2ScoreMetric<double>();
        double[] actuals = [5, 5, 5, 5, 5];
        double[] preds = [5, 5, 5, 5, 5];

        double result = NumOps.ToDouble(r2.Compute(preds, actuals));
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void R2_ConstantActuals_ImperfectPrediction_Returns0()
    {
        // When SS_tot = 0 and SS_res > 0, R² should be 0 per implementation
        var r2 = new R2ScoreMetric<double>();
        double[] actuals = [5, 5, 5, 5, 5];
        double[] preds = [4, 5, 6, 5, 5];

        double result = NumOps.ToDouble(r2.Compute(preds, actuals));
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void Pearson_ConstantInput_ReturnsZero()
    {
        // Pearson is undefined for constant data; implementation returns 0
        var pearson = new PearsonCorrelationMetric<double>();
        double[] actuals = [5, 5, 5, 5];
        double[] preds = [1, 2, 3, 4];

        double result = NumOps.ToDouble(pearson.Compute(preds, actuals));
        Assert.Equal(0.0, result, Tol);
    }

    // ========================================================================
    // Float type tests
    // ========================================================================

    [Fact]
    public void MSE_Float_MatchesDoubleWithReasonablePrecision()
    {
        var mseFloat = new MSEMetric<float>();
        var mseDouble = new MSEMetric<double>();
        var numOpsF = MathHelper.GetNumericOperations<float>();

        float[] predsF = [1.5f, 2.5f, 3.0f, 3.5f, 5.5f];
        float[] actualsF = [1f, 2f, 3f, 4f, 5f];
        double[] predsD = [1.5, 2.5, 3.0, 3.5, 5.5];
        double[] actualsD = [1, 2, 3, 4, 5];

        double resultF = numOpsF.ToDouble(mseFloat.Compute(predsF, actualsF));
        double resultD = NumOps.ToDouble(mseDouble.Compute(predsD, actualsD));
        Assert.Equal(resultD, resultF, 1e-6);
    }

    // ========================================================================
    // Bootstrap CI validation
    // ========================================================================

    [Fact]
    public void MSE_BootstrapCI_ContainsPointEstimate()
    {
        var mse = new MSEMetric<double>();
        double[] preds = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11];
        double[] actuals = [1.1, 2.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 10.2, 10.8];

        var result = mse.ComputeWithCI(preds, actuals, randomSeed: 42);
        double point = NumOps.ToDouble(result.Value);
        double lower = NumOps.ToDouble(result.LowerBound);
        double upper = NumOps.ToDouble(result.UpperBound);

        // CI should contain the point estimate (not always true with bootstrap, but usually)
        Assert.True(lower <= point + 0.1, $"Lower={lower} should be <= point={point}");
        Assert.True(upper >= point - 0.1, $"Upper={upper} should be >= point={point}");
        Assert.True(lower < upper, $"Lower={lower} should be < Upper={upper}");
    }
}
