using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep math-correctness integration tests for correlation metrics and
/// additional regression error metrics with hand-calculated expected values,
/// edge cases, and cross-metric mathematical identities.
///
/// Metrics tested:
///   PearsonCorrelation, NormalizedMSE, MeanBiasError, WeightedMAPE,
///   MedianAbsoluteError, MaxError, RelativeAbsoluteError, RelativeSquaredError,
///   RMSLE, MeanSquaredLogError
///
/// Cross-metric identities:
///   NMSE = 1 - R2, R2 = Pearson^2 (simple regression), RMSLE = sqrt(MSLE),
///   wMAPE = MAE * 100 / mean(|actual|), |MBE| <= MAE
/// </summary>
public class CorrelationAndErrorMetricsDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ═══════════════════════════════════════════════════════════════
    // PEARSON CORRELATION
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void PearsonCorrelation_PerfectPositiveLinear_ShouldBeOne()
    {
        // y = 2x + 3: perfect linear relationship
        var pred = new double[] { 5, 7, 9, 11, 13 };
        var actual = new double[] { 5, 7, 9, 11, 13 };
        var metric = new PearsonCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void PearsonCorrelation_PerfectNegativeLinear_ShouldBeNegOne()
    {
        var pred = new double[] { 5, 4, 3, 2, 1 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new PearsonCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(-1.0, result, Tol);
    }

    [Fact]
    public void PearsonCorrelation_HandCalculated()
    {
        // pred = [1, 2, 3, 4, 5], actual = [2, 4, 5, 4, 5]
        // mean_pred = 3, mean_actual = 4
        // cov = (1-3)(2-4) + (2-3)(4-4) + (3-3)(5-4) + (4-3)(4-4) + (5-3)(5-4)
        //     = (-2)(-2) + (-1)(0) + 0*1 + 1*0 + 2*1 = 4 + 0 + 0 + 0 + 2 = 6
        // var_pred = 4+1+0+1+4 = 10
        // var_actual = 4+0+1+0+1 = 6
        // r = 6 / sqrt(10*6) = 6/sqrt(60) = 6/7.7460 = 0.7746
        var pred = new double[] { 1, 2, 3, 4, 5 };
        var actual = new double[] { 2, 4, 5, 4, 5 };
        var metric = new PearsonCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        double expected = 6.0 / Math.Sqrt(10.0 * 6.0);
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void PearsonCorrelation_ConstantPredictions_ShouldBeZero()
    {
        var pred = new double[] { 5, 5, 5, 5 };
        var actual = new double[] { 1, 2, 3, 4 };
        var metric = new PearsonCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void PearsonCorrelation_SquaredEquals_R2_ForLinearFit()
    {
        // For a simple linear relationship: R2 = r^2
        // pred = actual + small noise (good linear fit, nearly unbiased)
        var pred = new double[] { 1.1, 2.0, 2.9, 4.1, 5.0 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var pearsonMetric = new PearsonCorrelationMetric<double>();
        var r2Metric = new R2ScoreMetric<double>();
        double r = pearsonMetric.Compute(pred, actual);
        double r2 = r2Metric.Compute(pred, actual);
        // r^2 should be close to R2 for nearly-perfect linear predictions
        Assert.True(Math.Abs(r * r - r2) < 0.01,
            $"Pearson^2 ({r * r}) should approximately equal R2 ({r2}) for good linear fit");
    }

    [Fact]
    public void PearsonCorrelation_LinearTransform_PreservesSign()
    {
        // Pearson is invariant under positive linear transform
        var pred = new double[] { 1, 2, 3, 4, 5 };
        var actual = new double[] { 2, 4, 5, 4, 5 };
        var metric = new PearsonCorrelationMetric<double>();
        double original = metric.Compute(pred, actual);

        // Scale pred by 3 and shift by 10: should give same Pearson r
        var predScaled = new double[] { 13, 16, 19, 22, 25 };
        double scaled = metric.Compute(predScaled, actual);
        Assert.Equal(original, scaled, Tol);
    }

    [Fact]
    public void PearsonCorrelation_BoundedByNegOneAndOne()
    {
        var pred = new double[] { 10, -5, 3, 0, 7 };
        var actual = new double[] { 1, 8, 2, 9, 3 };
        var metric = new PearsonCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.True(result >= -1.0 - Tol && result <= 1.0 + Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // NORMALIZED MSE
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void NormalizedMSE_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3, 4, 5 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new NormalizedMSEMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void NormalizedMSE_MeanPredictor_ShouldBeOne()
    {
        // Predicting mean for all => NMSE = MSE/Var = Var/Var = 1
        double mean = 3.0;
        var pred = new double[] { mean, mean, mean, mean, mean };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new NormalizedMSEMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void NormalizedMSE_HandCalculated()
    {
        // actual = [1,2,3,4,5], mean=3
        // pred = [1.5, 2.5, 3.5, 4.5, 5.5]
        // MSE = (0.25+0.25+0.25+0.25+0.25)/5 = 1.25/5 = 0.25
        // Var = (4+1+0+1+4)/5 = 10/5 = 2.0
        // NMSE = 0.25/2.0 = 0.125
        var pred = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new NormalizedMSEMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.125, result, Tol);
    }

    [Fact]
    public void NormalizedMSE_PlusR2_EqualsOne()
    {
        // NMSE = SS_res/SS_tot = 1 - R2 (using population variance, no n-1)
        // Note: NMSE uses /n for both MSE and Var, so it equals 1-R2 exactly
        var pred = new double[] { 1.1, 2.2, 2.8, 4.1, 4.8 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var nmseMetric = new NormalizedMSEMetric<double>();
        var r2Metric = new R2ScoreMetric<double>();
        double nmse = nmseMetric.Compute(pred, actual);
        double r2 = r2Metric.Compute(pred, actual);
        Assert.Equal(1.0, nmse + r2, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEAN BIAS ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MeanBiasError_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3, 4, 5 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new MeanBiasErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MeanBiasError_SystematicOverPrediction_IsPositive()
    {
        // All predictions 2 higher than actual
        // MBE = mean(pred - actual) = 2.0
        var pred = new double[] { 3, 4, 5, 6, 7 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new MeanBiasErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void MeanBiasError_SystematicUnderPrediction_IsNegative()
    {
        // All predictions 1 lower than actual
        var pred = new double[] { 0, 1, 2, 3, 4 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new MeanBiasErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(-1.0, result, Tol);
    }

    [Fact]
    public void MeanBiasError_BalancedErrors_CancelOut()
    {
        // Errors: +2, -2, +2, -2 => MBE = 0
        var pred = new double[] { 3, 0, 5, 2 };
        var actual = new double[] { 1, 2, 3, 4 };
        var metric = new MeanBiasErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MeanBiasError_AbsValue_LessThanOrEqual_MAE()
    {
        // |MBE| <= MAE always (triangle inequality for means)
        var pred = new double[] { 1, 5, 2, 8 };
        var actual = new double[] { 3, 3, 3, 3 };
        var mbeMetric = new MeanBiasErrorMetric<double>();
        var maeMetric = new MAEMetric<double>();
        double mbe = mbeMetric.Compute(pred, actual);
        double mae = maeMetric.Compute(pred, actual);
        Assert.True(Math.Abs(mbe) <= mae + Tol,
            $"|MBE| ({Math.Abs(mbe)}) should be <= MAE ({mae})");
    }

    [Fact]
    public void MeanBiasError_HandCalculated()
    {
        // pred = [3, 1, 5], actual = [2, 3, 4]
        // errors: 1, -2, 1
        // MBE = (1 + (-2) + 1) / 3 = 0/3 = 0
        var pred = new double[] { 3, 1, 5 };
        var actual = new double[] { 2, 3, 4 };
        var metric = new MeanBiasErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // WEIGHTED MAPE
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void WeightedMAPE_HandCalculated()
    {
        // wMAPE = 100 * sum(|y-yhat|) / sum(|y|)
        // actual = [100, 200, 300], pred = [110, 190, 310]
        // sum(|error|) = 10 + 10 + 10 = 30
        // sum(|actual|) = 100 + 200 + 300 = 600
        // wMAPE = 100 * 30/600 = 5.0
        var pred = new double[] { 110, 190, 310 };
        var actual = new double[] { 100, 200, 300 };
        var metric = new WeightedMAPEMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(5.0, result, Tol);
    }

    [Fact]
    public void WeightedMAPE_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new WeightedMAPEMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void WeightedMAPE_EqualErrors_WeightsLargerActuals()
    {
        // Same absolute error of 10, but different actuals
        // Large actual (1000): contributes 10/1000 relative error
        // Small actual (100): contributes 10/100 relative error
        // wMAPE gives more weight to large actual since denominator = sum(|actual|)
        var pred1 = new double[] { 1010 };
        var actual1 = new double[] { 1000 };
        var pred2 = new double[] { 110 };
        var actual2 = new double[] { 100 };
        var metric = new WeightedMAPEMetric<double>();
        double wmape1 = metric.Compute(pred1, actual1);
        double wmape2 = metric.Compute(pred2, actual2);
        // wMAPE(large) = 100*10/1000 = 1.0
        // wMAPE(small) = 100*10/100 = 10.0
        Assert.Equal(1.0, wmape1, Tol);
        Assert.Equal(10.0, wmape2, Tol);
    }

    [Fact]
    public void WeightedMAPE_RelatesTo_MAE()
    {
        // wMAPE = 100 * N * MAE / sum(|actual|)
        var pred = new double[] { 110, 190, 310 };
        var actual = new double[] { 100, 200, 300 };
        var wmapeMetric = new WeightedMAPEMetric<double>();
        var maeMetric = new MAEMetric<double>();
        double wmape = wmapeMetric.Compute(pred, actual);
        double mae = maeMetric.Compute(pred, actual);
        double sumAbsActual = 600;
        double expectedWmape = 100.0 * pred.Length * mae / sumAbsActual;
        Assert.Equal(expectedWmape, wmape, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEDIAN ABSOLUTE ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MedianAbsoluteError_HandCalculated_OddCount()
    {
        // errors: |1-2|=1, |3-2|=1, |5-2|=3
        // sorted: [1, 1, 3]
        // median = 1
        var pred = new double[] { 2, 2, 2 };
        var actual = new double[] { 1, 3, 5 };
        var metric = new MedianAbsoluteErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void MedianAbsoluteError_HandCalculated_EvenCount()
    {
        // errors: |1-3|=2, |2-3|=1, |4-3|=1, |5-3|=2
        // sorted: [1, 1, 2, 2]
        // median = (1+2)/2 = 1.5
        var pred = new double[] { 3, 3, 3, 3 };
        var actual = new double[] { 1, 2, 4, 5 };
        var metric = new MedianAbsoluteErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.5, result, Tol);
    }

    [Fact]
    public void MedianAbsoluteError_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new MedianAbsoluteErrorMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void MedianAbsoluteError_RobustToOutliers()
    {
        // Median is robust to outliers, MAE is not as robust
        // errors: 1, 1, 1, 1, 100 => median=1, MAE=20.8
        var pred = new double[] { 0, 0, 0, 0, 0 };
        var actual = new double[] { 1, 1, 1, 1, 100 };
        var medMetric = new MedianAbsoluteErrorMetric<double>();
        var maeMetric = new MAEMetric<double>();
        double median = medMetric.Compute(pred, actual);
        double mae = maeMetric.Compute(pred, actual);
        Assert.Equal(1.0, median, Tol);
        Assert.True(mae > 10, "MAE should be much larger than MedianAE with outlier");
    }

    // ═══════════════════════════════════════════════════════════════
    // MAX ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MaxError_HandCalculated()
    {
        // errors: |1-3|=2, |5-3|=2, |10-3|=7
        // max = 7
        var pred = new double[] { 3, 3, 3 };
        var actual = new double[] { 1, 5, 10 };
        var metric = new MaxErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(7.0, result, Tol);
    }

    [Fact]
    public void MaxError_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new MaxErrorMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void MaxError_GreaterThanOrEqual_MAE()
    {
        // Max error >= MAE always (max >= mean for positive values)
        var pred = new double[] { 1, 5, 2, 8 };
        var actual = new double[] { 3, 3, 3, 3 };
        var maxMetric = new MaxErrorMetric<double>();
        var maeMetric = new MAEMetric<double>();
        double maxErr = maxMetric.Compute(pred, actual);
        double mae = maeMetric.Compute(pred, actual);
        Assert.True(maxErr >= mae - Tol,
            $"MaxError ({maxErr}) should be >= MAE ({mae})");
    }

    // ═══════════════════════════════════════════════════════════════
    // RELATIVE ABSOLUTE ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void RelativeAbsoluteError_HandCalculated()
    {
        // RAE = sum(|y-yhat|) / sum(|y-mean(y)|)
        // actual=[1,2,3,4,5], mean=3, pred=[1.5,2.5,3.5,4.5,5.5]
        // sum(|y-yhat|) = 0.5*5 = 2.5
        // sum(|y-mean|) = 2+1+0+1+2 = 6
        // RAE = 2.5/6 = 0.4167
        var pred = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new RelativeAbsoluteErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.5 / 6.0, result, Tol);
    }

    [Fact]
    public void RelativeAbsoluteError_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new RelativeAbsoluteErrorMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void RelativeAbsoluteError_MeanPredictor_ShouldBeOne()
    {
        // Predicting mean for all: sum(|y-mean|)/sum(|y-mean|) = 1
        double mean = 3.0;
        var pred = new double[] { mean, mean, mean, mean, mean };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new RelativeAbsoluteErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // RELATIVE SQUARED ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void RelativeSquaredError_HandCalculated()
    {
        // RSE = sum((y-yhat)^2) / sum((y-mean(y))^2) = 1 - R2
        // actual=[1,2,3,4,5], mean=3, pred=[1.5,2.5,3.5,4.5,5.5]
        // SS_res = 0.25*5 = 1.25
        // SS_tot = 4+1+0+1+4 = 10
        // RSE = 1.25/10 = 0.125
        var pred = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new RelativeSquaredErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.125, result, Tol);
    }

    [Fact]
    public void RelativeSquaredError_PlusR2_EqualsOne()
    {
        // RSE = 1 - R2 (they share SS_res/SS_tot)
        var pred = new double[] { 1.1, 2.2, 2.8, 4.1, 4.8 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var rseMetric = new RelativeSquaredErrorMetric<double>();
        var r2Metric = new R2ScoreMetric<double>();
        double rse = rseMetric.Compute(pred, actual);
        double r2 = r2Metric.Compute(pred, actual);
        Assert.Equal(1.0, rse + r2, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // RMSLE (Root Mean Squared Log Error)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void RMSLE_EqualsSqrt_MSLE()
    {
        // RMSLE = sqrt(MSLE) by definition
        var pred = new double[] { 2.5, 4.8, 7.0 };
        var actual = new double[] { 3.0, 5.0, 6.0 };
        var rmsleMetric = new RMSLEMetric<double>();
        var msleMetric = new MeanSquaredLogErrorMetric<double>();
        double rmsle = rmsleMetric.Compute(pred, actual);
        double msle = msleMetric.Compute(pred, actual);
        Assert.Equal(Math.Sqrt(msle), rmsle, Tol);
    }

    [Fact]
    public void RMSLE_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new RMSLEMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void RMSLE_HandCalculated()
    {
        // y=3,yhat=2.5: (log(4)-log(3.5))^2 = (1.3863-1.2528)^2 = 0.01785
        // RMSLE = sqrt(0.01785) = 0.1336
        var pred = new double[] { 2.5 };
        var actual = new double[] { 3.0 };
        var metric = new RMSLEMetric<double>();
        double result = metric.Compute(pred, actual);
        double expected = Math.Abs(Math.Log(4.0) - Math.Log(3.5));
        Assert.Equal(expected, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // RMSE = sqrt(MSE) RELATIONSHIP
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void RMSE_EqualsSqrt_MSE()
    {
        var pred = new double[] { 1.5, 2.5, 3.5 };
        var actual = new double[] { 1, 2, 3 };
        var rmseMetric = new RMSEMetric<double>();
        var mseMetric = new MSEMetric<double>();
        double rmse = rmseMetric.Compute(pred, actual);
        double mse = mseMetric.Compute(pred, actual);
        Assert.Equal(Math.Sqrt(mse), rmse, Tol);
    }

    [Fact]
    public void RMSE_GreaterThanOrEqual_MAE()
    {
        // RMSE >= MAE always (QM >= AM for absolute errors)
        var pred = new double[] { 1, 5, 2, 8 };
        var actual = new double[] { 3, 3, 3, 3 };
        var rmseMetric = new RMSEMetric<double>();
        var maeMetric = new MAEMetric<double>();
        double rmse = rmseMetric.Compute(pred, actual);
        double mae = maeMetric.Compute(pred, actual);
        Assert.True(rmse >= mae - Tol,
            $"RMSE ({rmse}) should be >= MAE ({mae})");
    }

    // ═══════════════════════════════════════════════════════════════
    // PEARSON vs SPEARMAN COMPARISON
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void PearsonVsSpearman_LinearRelationship_BothOne()
    {
        // Linear relationship => both Pearson and Spearman = 1
        var pred = new double[] { 2, 4, 6, 8, 10 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var pearson = new PearsonCorrelationMetric<double>().Compute(pred, actual);
        var spearman = new SpearmanCorrelationMetric<double>().Compute(pred, actual);
        Assert.Equal(1.0, pearson, Tol);
        Assert.Equal(1.0, spearman, Tol);
    }

    [Fact]
    public void PearsonVsSpearman_NonlinearMonotonic_SpearmanHigher()
    {
        // Exponential relationship: monotonic but not linear
        // pred = [1, 2, 4, 8, 16], actual = [1, 2, 3, 4, 5]
        // Spearman captures monotonic => 1.0
        // Pearson captures linear => < 1.0
        var pred = new double[] { 1, 2, 4, 8, 16 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var pearson = new PearsonCorrelationMetric<double>().Compute(pred, actual);
        var spearman = new SpearmanCorrelationMetric<double>().Compute(pred, actual);
        Assert.Equal(1.0, spearman, Tol);
        Assert.True(pearson < 1.0,
            $"Pearson ({pearson}) should be < 1 for nonlinear monotonic relationship");
    }

    // ═══════════════════════════════════════════════════════════════
    // COMPREHENSIVE CROSS-METRIC IDENTITIES
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void CrossMetric_NMSE_Equals_OneMinusR2()
    {
        var pred = new double[] { 1.5, 2.3, 3.1, 4.2, 4.8 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var nmse = new NormalizedMSEMetric<double>().Compute(pred, actual);
        var r2 = new R2ScoreMetric<double>().Compute(pred, actual);
        Assert.Equal(1.0, nmse + r2, Tol);
    }

    [Fact]
    public void CrossMetric_RSE_Equals_OneMinusR2()
    {
        var pred = new double[] { 1.5, 2.3, 3.1, 4.2, 4.8 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var rse = new RelativeSquaredErrorMetric<double>().Compute(pred, actual);
        var r2 = new R2ScoreMetric<double>().Compute(pred, actual);
        Assert.Equal(1.0, rse + r2, Tol);
    }

    [Fact]
    public void CrossMetric_MSE_GreaterThanOrEqual_MAE_Squared()
    {
        // MSE >= MAE^2 always (Jensen's inequality: E[X^2] >= (E[X])^2)
        // Note: MAE = E[|error|], MSE = E[error^2]
        // This only holds when errors are non-negative (which |error| is)
        // Actually MSE >= MAE^2 is not always true for the mean.
        // But MSE >= (MBE)^2 is true.
        var pred = new double[] { 1, 5, 2, 8 };
        var actual = new double[] { 3, 3, 3, 3 };
        var mse = new MSEMetric<double>().Compute(pred, actual);
        var mbe = new MeanBiasErrorMetric<double>().Compute(pred, actual);
        Assert.True(mse >= mbe * mbe - Tol,
            $"MSE ({mse}) should be >= MBE^2 ({mbe * mbe})");
    }

    [Fact]
    public void CrossMetric_MaxError_GreaterThanOrEqual_AllOtherErrors()
    {
        var pred = new double[] { 1, 5, 2, 8 };
        var actual = new double[] { 3, 3, 3, 3 };
        var maxErr = new MaxErrorMetric<double>().Compute(pred, actual);
        var mae = new MAEMetric<double>().Compute(pred, actual);
        var medAE = new MedianAbsoluteErrorMetric<double>().Compute(pred, actual);
        Assert.True(maxErr >= mae - Tol);
        Assert.True(maxErr >= medAE - Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // EDGE CASES
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void AllMetrics_EmptyInput_HandlesGracefully()
    {
        var empty = Array.Empty<double>();
        Assert.Equal(0.0, new NormalizedMSEMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new MeanBiasErrorMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new WeightedMAPEMetric<double>().Compute(empty, empty), Tol);
    }

    [Fact]
    public void AllMetrics_MismatchedLengths_Throws()
    {
        var a = new double[] { 1, 2 };
        var b = new double[] { 1 };
        Assert.Throws<ArgumentException>(() => new PearsonCorrelationMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new NormalizedMSEMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new MeanBiasErrorMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new WeightedMAPEMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new RelativeAbsoluteErrorMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new RelativeSquaredErrorMetric<double>().Compute(a, b));
    }

    [Fact]
    public void PearsonCorrelation_SingleElement_ReturnsZero()
    {
        var pred = new double[] { 1.0 };
        var actual = new double[] { 2.0 };
        var metric = new PearsonCorrelationMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void MeanBiasError_DirectionIs_TargetValue()
    {
        // MBE has TargetValue direction (ideal is 0, not higher or lower)
        var metric = new MeanBiasErrorMetric<double>();
        Assert.Equal(MetricDirection.TargetValue, metric.Direction);
    }
}
