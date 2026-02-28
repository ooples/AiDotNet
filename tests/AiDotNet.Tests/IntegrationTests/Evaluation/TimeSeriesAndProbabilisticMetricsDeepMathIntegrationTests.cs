using AiDotNet.Evaluation.Metrics.Probabilistic;
using AiDotNet.Evaluation.Metrics.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep math-correctness integration tests for time series metrics (MASE, SMAPE, TheilU, WAPE,
/// MASESeasonal), probabilistic metrics (CRPS, PinballLoss).
/// Verifies hand-calculated values, mathematical identities, edge cases, and known properties.
/// </summary>
public class TimeSeriesAndProbabilisticMetricsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    #region SMAPE - Symmetric Mean Absolute Percentage Error

    [Fact]
    public void SMAPE_HandCalculated_SimpleCase()
    {
        // SMAPE = (100/N) * sum( |y - yhat| / ((|y| + |yhat|) / 2) )
        // y = [100, 200, 300], yhat = [110, 190, 330]
        // Element 0: |100-110| / ((100+110)/2) = 10/105 = 0.09524
        // Element 1: |200-190| / ((200+190)/2) = 10/195 = 0.05128
        // Element 2: |300-330| / ((300+330)/2) = 30/315 = 0.09524
        // Sum = 0.24176, SMAPE = 100 * 0.24176 / 3 = 8.0587
        var metric = new SMAPEMetric<double>();
        double[] pred = [110, 190, 330];
        double[] actual = [100, 200, 300];

        var result = metric.Compute(pred, actual);
        double expected = 100.0 * (10.0 / 105 + 10.0 / 195 + 30.0 / 315) / 3;
        Assert.Equal(expected, result, LooseTolerance);
    }

    [Fact]
    public void SMAPE_PerfectPredictions_ReturnsZero()
    {
        var metric = new SMAPEMetric<double>();
        double[] values = [100, 200, 300, 400];
        var result = metric.Compute(values, values);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SMAPE_Bounded_Between0And200()
    {
        // SMAPE should always be in [0, 200]
        var metric = new SMAPEMetric<double>();
        double[] pred = [1, 1, 1];
        double[] actual = [1000, 1000, 1000];
        var result = metric.Compute(pred, actual);
        Assert.True(result >= 0, $"SMAPE {result} should be >= 0");
        Assert.True(result <= 200, $"SMAPE {result} should be <= 200");
    }

    [Fact]
    public void SMAPE_SymmetricInOverUnderPrediction()
    {
        // SMAPE should give the same value for y=100,yhat=200 and y=200,yhat=100
        var metric = new SMAPEMetric<double>();
        double[] pred1 = [200];
        double[] actual1 = [100];
        double[] pred2 = [100];
        double[] actual2 = [200];

        var result1 = metric.Compute(pred1, actual1);
        var result2 = metric.Compute(pred2, actual2);
        Assert.Equal(result1, result2, Tolerance);
    }

    [Fact]
    public void SMAPE_UniformScaling_SameResult()
    {
        // SMAPE is scale-invariant: scaling both pred and actual by same factor gives same result
        var metric = new SMAPEMetric<double>();
        double[] pred = [110, 190, 330];
        double[] actual = [100, 200, 300];
        double[] predScaled = [1100, 1900, 3300];
        double[] actualScaled = [1000, 2000, 3000];

        var result1 = metric.Compute(pred, actual);
        var result2 = metric.Compute(predScaled, actualScaled);
        Assert.Equal(result1, result2, LooseTolerance);
    }

    [Fact]
    public void SMAPE_MaximumValue_OppositeSignPredictions()
    {
        // |y - yhat| / ((|y| + |yhat|)/2) = |100 - (-100)| / ((100+100)/2) = 200/100 = 2
        // SMAPE = 100 * 2 = 200
        var metric = new SMAPEMetric<double>();
        double[] pred = [-100, -200];
        double[] actual = [100, 200];
        var result = metric.Compute(pred, actual);
        Assert.Equal(200.0, result, LooseTolerance);
    }

    #endregion

    #region MASE - Mean Absolute Scaled Error

    [Fact]
    public void MASE_HandCalculated()
    {
        // Actuals: [10, 12, 14, 16, 18] with period=1
        // Naive forecast errors: |12-10|+|14-12|+|16-14|+|18-16| = 2+2+2+2=8
        // MAE_naive = 8/4 = 2
        // Predictions: [11, 13, 13, 17, 17], errors: 1+1+1+1+1=5
        // MAE = 5/5 = 1
        // MASE = 1/2 = 0.5
        var metric = new MASEMetric<double>();
        double[] pred = [11, 13, 13, 17, 17];
        double[] actual = [10, 12, 14, 16, 18];

        var result = metric.Compute(pred, actual, 1);
        Assert.Equal(0.5, result, LooseTolerance);
    }

    [Fact]
    public void MASE_PerfectPredictions_ReturnsZero()
    {
        var metric = new MASEMetric<double>();
        double[] values = [10, 20, 30, 40, 50];
        var result = metric.Compute(values, values, 1);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MASE_NaiveForecaster_ReturnsOne()
    {
        // If MAE = MAE_naive, MASE = 1
        var metric = new MASEMetric<double>();
        double[] actual = [10, 15, 20, 25, 30];
        // MAE_naive for period=1: (5+5+5+5)/4 = 5
        // Need MAE of predictions = 5
        double[] pred = [15, 10, 25, 20, 35]; // errors: 5,5,5,5,5 -> MAE=5
        var result = metric.Compute(pred, actual, 1);
        Assert.Equal(1.0, result, LooseTolerance);
    }

    [Fact]
    public void MASE_WithSeasonalPeriod()
    {
        // Actuals: [1, 2, 3, 4, 5, 6] with period=3
        // Seasonal naive errors (i>=3): |4-1|+|5-2|+|6-3| = 3+3+3=9
        // MAE_naive = 9/3 = 3
        // Predictions: [1, 2, 3, 5, 5, 5], errors: 0+0+0+1+0+1=2
        // MAE = 2/6 = 1/3
        // MASE = (1/3) / 3 = 1/9
        var metric = new MASEMetric<double>();
        double[] pred = [1, 2, 3, 5, 5, 5];
        double[] actual = [1, 2, 3, 4, 5, 6];

        var result = metric.Compute(pred, actual, 3);
        double expectedMASE = (2.0 / 6) / 3.0;
        Assert.Equal(expectedMASE, result, LooseTolerance);
    }

    [Fact]
    public void MASE_BetterThanNaive_LessThanOne()
    {
        var metric = new MASEMetric<double>();
        // Linear trend with small prediction errors
        double[] actual = [10, 20, 30, 40, 50];
        double[] pred = [11, 19, 31, 39, 51]; // very close
        var result = metric.Compute(pred, actual, 1);
        Assert.True(result < 1.0, $"MASE {result} should be < 1 for predictions better than naive");
    }

    #endregion

    #region MASESeasonal - Seasonal MASE

    [Fact]
    public void MASESeasonal_MatchesMASE_WithSamePeriod()
    {
        var mase = new MASEMetric<double>();
        var maseSeasonal = new MASESeasonalMetric<double>(1);

        double[] pred = [11, 13, 13, 17, 17];
        double[] actual = [10, 12, 14, 16, 18];

        var resultMASE = mase.Compute(pred, actual, 1);
        var resultSeasonal = maseSeasonal.Compute(pred, actual, 1);
        Assert.Equal(resultMASE, resultSeasonal, LooseTolerance);
    }

    [Fact]
    public void MASESeasonal_PerfectPredictions_ReturnsZero()
    {
        var metric = new MASESeasonalMetric<double>(2);
        double[] values = [1, 2, 3, 4, 5, 6, 7, 8];
        var result = metric.Compute(values, values, 2);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MASESeasonal_InsufficientData_ReturnsZero()
    {
        // When data length <= seasonal period, should return 0
        var metric = new MASESeasonalMetric<double>(10);
        double[] pred = [1, 2, 3];
        double[] actual = [1, 2, 3];
        var result = metric.Compute(pred, actual, 10);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Theil's U Statistic

    [Fact]
    public void TheilU_HandCalculated()
    {
        // Actuals: [10, 12, 15, 13, 16]
        // Predictions: [11, 13, 14, 14, 15]
        // Squared errors: 1+1+1+1+1 = 5
        // Naive changes^2: 4+9+4+9 = 26
        // TheilU = sqrt(5/26)
        var metric = new TheilUMetric<double>();
        double[] pred = [11, 13, 14, 14, 15];
        double[] actual = [10, 12, 15, 13, 16];

        var result = metric.Compute(pred, actual);
        double expected = Math.Sqrt(5.0 / 26.0);
        Assert.Equal(expected, result, LooseTolerance);
    }

    [Fact]
    public void TheilU_PerfectPredictions_ReturnsZero()
    {
        var metric = new TheilUMetric<double>();
        double[] values = [10, 20, 30, 40];
        var result = metric.Compute(values, values);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TheilU_NaiveForecast_ReturnsOne()
    {
        // pred = [actual[0], actual[0], actual[1]] (naive no-change from first value)
        // actual = [10, 20, 30]
        // pred = [10, 10, 20]
        // Squared errors: 0 + 100 + 100 = 200
        // Naive changes^2: 100 + 100 = 200
        // TheilU = sqrt(200/200) = 1.0
        var metric = new TheilUMetric<double>();
        double[] pred = [10, 10, 20];
        double[] actual = [10, 20, 30];

        var result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, LooseTolerance);
    }

    [Fact]
    public void TheilU_ConstantActuals_PerfectPred_ReturnsZero()
    {
        var metric = new TheilUMetric<double>();
        double[] actual = [5, 5, 5, 5];
        double[] pred = [5, 5, 5, 5];
        var result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TheilU_BetterThanNaive_LessThanOne()
    {
        var metric = new TheilUMetric<double>();
        double[] actual = [10, 20, 30, 40, 50];
        double[] predGood = [11, 19, 31, 39, 51];
        var result = metric.Compute(predGood, actual);
        Assert.True(result < 1.0, $"TheilU {result} should be < 1 for good predictions");
    }

    #endregion

    #region WAPE - Weighted Absolute Percentage Error

    [Fact]
    public void WAPE_HandCalculated()
    {
        // WAPE = sum(|y - yhat|) / sum(|y|)
        // sum(|error|) = 10+10+30 = 50, sum(|y|) = 100+200+300 = 600
        // WAPE = 50/600 = 0.08333
        var metric = new WAPEMetric<double>();
        double[] pred = [110, 190, 330];
        double[] actual = [100, 200, 300];

        var result = metric.Compute(pred, actual);
        Assert.Equal(50.0 / 600.0, result, LooseTolerance);
    }

    [Fact]
    public void WAPE_PerfectPredictions_ReturnsZero()
    {
        var metric = new WAPEMetric<double>();
        double[] values = [100, 200, 300];
        var result = metric.Compute(values, values);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void WAPE_ScaleInvariant()
    {
        var metric = new WAPEMetric<double>();
        double[] pred = [110, 190, 330];
        double[] actual = [100, 200, 300];
        double c = 5.0;
        double[] predScaled = pred.Select(x => c * x).ToArray();
        double[] actualScaled = actual.Select(x => c * x).ToArray();

        var result1 = metric.Compute(pred, actual);
        var result2 = metric.Compute(predScaled, actualScaled);
        Assert.Equal(result1, result2, Tolerance);
    }

    [Fact]
    public void WAPE_EqualsMAETimesN_DividedBySumActuals()
    {
        var metric = new WAPEMetric<double>();
        double[] pred = [12, 18, 25];
        double[] actual = [10, 20, 30];

        var wape = metric.Compute(pred, actual);
        double mae = (Math.Abs(10 - 12) + Math.Abs(20 - 18) + Math.Abs(30 - 25)) / 3.0;
        double sumActuals = 10 + 20 + 30;
        double wapeFromMAE = mae * 3 / sumActuals;
        Assert.Equal(wapeFromMAE, wape, Tolerance);
    }

    [Fact]
    public void WAPE_WeightsLargerActualsMore()
    {
        var metric = new WAPEMetric<double>();
        var wape1 = metric.Compute(new double[] { 110 }, new double[] { 100 }); // 10/100 = 0.1
        var wape2 = metric.Compute(new double[] { 1010 }, new double[] { 1000 }); // 10/1000 = 0.01
        Assert.Equal(0.1, wape1, LooseTolerance);
        Assert.Equal(0.01, wape2, LooseTolerance);
    }

    #endregion

    #region CRPS - Continuous Ranked Probability Score (Point Predictions)

    [Fact]
    public void CRPS_PointPredictions_EqualsMAE()
    {
        var metric = new CRPSMetric<double>();
        double[] pred = [11, 18, 32];
        double[] actual = [10, 20, 30];

        var crps = metric.Compute(pred, actual);
        double mae = (1.0 + 2.0 + 2.0) / 3.0;
        Assert.Equal(mae, crps, Tolerance);
    }

    [Fact]
    public void CRPS_PerfectPredictions_ReturnsZero()
    {
        var metric = new CRPSMetric<double>();
        double[] values = [1, 2, 3, 4, 5];
        var result = metric.Compute(values, values);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CRPS_NonNegative()
    {
        var metric = new CRPSMetric<double>();
        double[] pred = [-5, 100, 0];
        double[] actual = [10, -20, 50];
        var result = metric.Compute(pred, actual);
        Assert.True(result >= 0, $"CRPS {result} should be non-negative");
    }

    [Fact]
    public void CRPS_TriangleInequality()
    {
        var metric = new CRPSMetric<double>();
        double[] pred = [10, 20, 30];
        double[] actual = [15, 25, 35];
        double[] mid = [12.5, 22.5, 32.5];

        var directCRPS = metric.Compute(pred, actual);
        var crps1 = metric.Compute(pred, mid);
        var crps2 = metric.Compute(mid, actual);

        Assert.True(directCRPS <= crps1 + crps2 + Tolerance,
            $"CRPS triangle inequality: {directCRPS} <= {crps1} + {crps2}");
    }

    #endregion

    #region Pinball Loss (Quantile Loss)

    [Fact]
    public void PinballLoss_Median_EqualsHalfMAE()
    {
        var metric = new PinballLossMetric<double>(0.5);
        double[] pred = [11, 18, 32];
        double[] actual = [10, 20, 30];

        var loss = metric.Compute(pred, actual);
        double mae = (1.0 + 2.0 + 2.0) / 3.0;
        Assert.Equal(mae * 0.5, loss, Tolerance);
    }

    [Fact]
    public void PinballLoss_HandCalculated_Tau09()
    {
        // tau=0.9: overpred penalty = 0.1*|diff|, underpred penalty = 0.9*|diff|
        // pred=12, actual=10: overpred, loss = 0.1*2 = 0.2
        // pred=8, actual=10: underpred, loss = 0.9*2 = 1.8
        var metric = new PinballLossMetric<double>(0.9);
        double[] pred = [12, 8];
        double[] actual = [10, 10];

        var loss = metric.Compute(pred, actual);
        Assert.Equal((0.2 + 1.8) / 2.0, loss, Tolerance);
    }

    [Fact]
    public void PinballLoss_HandCalculated_Tau01()
    {
        // tau=0.1: overpred penalty = 0.9*|diff|, underpred penalty = 0.1*|diff|
        var metric = new PinballLossMetric<double>(0.1);
        double[] pred = [12, 8];
        double[] actual = [10, 10];

        var loss = metric.Compute(pred, actual);
        Assert.Equal((1.8 + 0.2) / 2.0, loss, Tolerance);
    }

    [Fact]
    public void PinballLoss_SymmetricOverUnder_SameMagnitude()
    {
        // For same magnitude errors: Pinball(tau, overpredict) vs Pinball(1-tau, underpredict)
        var metricHigh = new PinballLossMetric<double>(0.9);
        var metricLow = new PinballLossMetric<double>(0.1);

        double[] predOver = [12, 22, 32]; // overprediction by 2
        double[] actual = [10, 20, 30];
        double[] predUnder = [8, 18, 28]; // underprediction by 2

        var lossHighOver = metricHigh.Compute(predOver, actual);  // 0.1 * 2 = 0.2
        var lossLowUnder = metricLow.Compute(predUnder, actual);  // 0.1 * 2 = 0.2
        Assert.Equal(lossHighOver, lossLowUnder, Tolerance);
    }

    [Fact]
    public void PinballLoss_PerfectPredictions_ReturnsZero()
    {
        var metric = new PinballLossMetric<double>(0.75);
        double[] values = [10, 20, 30];
        var result = metric.Compute(values, values);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void PinballLoss_NonNegative()
    {
        var metric = new PinballLossMetric<double>(0.5);
        double[] pred = [100, -50, 0];
        double[] actual = [-10, 200, 30];
        var result = metric.Compute(pred, actual);
        Assert.True(result >= 0, $"Pinball loss {result} should be non-negative");
    }

    [Fact]
    public void PinballLoss_HigherTau_PenalizesUnderpredictionMore()
    {
        var metricLow = new PinballLossMetric<double>(0.1);
        var metricHigh = new PinballLossMetric<double>(0.9);

        double[] pred = [8]; // underprediction
        double[] actual = [10];

        var lossLow = metricLow.Compute(pred, actual);  // 0.1 * 2 = 0.2
        var lossHigh = metricHigh.Compute(pred, actual); // 0.9 * 2 = 1.8
        Assert.True(lossHigh > lossLow,
            $"Higher tau should penalize underprediction more: {lossHigh} > {lossLow}");
    }

    [Fact]
    public void PinballLoss_InvalidTau_ThrowsException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PinballLossMetric<double>(0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PinballLossMetric<double>(1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PinballLossMetric<double>(-0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PinballLossMetric<double>(1.5));
    }

    [Fact]
    public void PinballLoss_AllQuantiles_Consistent()
    {
        // For any quantile, pinball loss of perfect predictions is 0
        double[] values = [5, 10, 15];
        foreach (double tau in new[] { 0.1, 0.25, 0.5, 0.75, 0.9 })
        {
            var metric = new PinballLossMetric<double>(tau);
            var result = metric.Compute(values, values);
            Assert.Equal(0.0, result, Tolerance);
        }
    }

    #endregion

    #region Cross-Metric Consistency

    [Fact]
    public void CRPS_WAPE_Relationship()
    {
        // CRPS = MAE = sum(|errors|)/N
        // WAPE = sum(|errors|)/sum(|actuals|)
        // Therefore: CRPS = WAPE * mean(|actuals|)
        var crpsMetric = new CRPSMetric<double>();
        var wapeMetric = new WAPEMetric<double>();

        double[] pred = [12, 18, 35, 42];
        double[] actual = [10, 20, 30, 40];

        var crps = crpsMetric.Compute(pred, actual);
        var wape = wapeMetric.Compute(pred, actual);
        double meanAbsActual = actual.Select(Math.Abs).Average();
        Assert.Equal(crps, wape * meanAbsActual, LooseTolerance);
    }

    [Fact]
    public void AllMetrics_PerfectPredictions_AllReturnZero()
    {
        double[] values = [10, 20, 30, 40, 50];

        Assert.Equal(0.0, new SMAPEMetric<double>().Compute(values, values), Tolerance);
        Assert.Equal(0.0, new MASEMetric<double>().Compute(values, values, 1), Tolerance);
        Assert.Equal(0.0, new TheilUMetric<double>().Compute(values, values), Tolerance);
        Assert.Equal(0.0, new WAPEMetric<double>().Compute(values, values), Tolerance);
        Assert.Equal(0.0, new CRPSMetric<double>().Compute(values, values), Tolerance);
    }

    [Fact]
    public void AllMetrics_WorsePredictions_GiveHigherValues()
    {
        double[] actual = [10, 20, 30, 40, 50];
        double[] predGood = [11, 19, 31, 39, 51];
        double[] predBad = [20, 10, 40, 30, 60];

        var smape = new SMAPEMetric<double>();
        var wape = new WAPEMetric<double>();
        var crps = new CRPSMetric<double>();

        Assert.True(smape.Compute(predBad, actual) > smape.Compute(predGood, actual));
        Assert.True(wape.Compute(predBad, actual) > wape.Compute(predGood, actual));
        Assert.True(crps.Compute(predBad, actual) > crps.Compute(predGood, actual));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void SMAPE_EmptyInput_ReturnsZero()
    {
        var metric = new SMAPEMetric<double>();
        var result = metric.Compute(ReadOnlySpan<double>.Empty, ReadOnlySpan<double>.Empty);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MASE_EmptyInput_ReturnsZero()
    {
        var metric = new MASEMetric<double>();
        var result = metric.Compute(ReadOnlySpan<double>.Empty, ReadOnlySpan<double>.Empty);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TheilU_SingleElement_ReturnsZero()
    {
        var metric = new TheilUMetric<double>();
        double[] pred = [5];
        double[] actual = [5];
        var result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void WAPE_EmptyInput_ReturnsZero()
    {
        var metric = new WAPEMetric<double>();
        var result = metric.Compute(ReadOnlySpan<double>.Empty, ReadOnlySpan<double>.Empty);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CRPS_EmptyInput_ReturnsZero()
    {
        var metric = new CRPSMetric<double>();
        var result = metric.Compute(ReadOnlySpan<double>.Empty, ReadOnlySpan<double>.Empty);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SMAPE_MismatchedLengths_ThrowsException()
    {
        var metric = new SMAPEMetric<double>();
        Assert.Throws<ArgumentException>(() => metric.Compute(new double[] { 1, 2, 3 }, new double[] { 1, 2 }));
    }

    [Fact]
    public void TheilU_MismatchedLengths_ThrowsException()
    {
        var metric = new TheilUMetric<double>();
        Assert.Throws<ArgumentException>(() => metric.Compute(new double[] { 1, 2, 3 }, new double[] { 1, 2 }));
    }

    #endregion
}
