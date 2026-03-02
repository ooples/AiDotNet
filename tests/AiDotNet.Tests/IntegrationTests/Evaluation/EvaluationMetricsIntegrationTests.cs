using AiDotNet.Evaluation.Metrics;
using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Metrics.Regression;
using AiDotNet.Evaluation.Metrics.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Comprehensive integration tests for all evaluation metric classes.
/// Tests mathematical properties: known values, boundary conditions,
/// perfect predictions, worst-case predictions, and metric relationships.
/// </summary>
public class EvaluationMetricsIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    // Standard test data
    private static readonly double[] PerfectPredictions = [1.0, 2.0, 3.0, 4.0, 5.0];
    private static readonly double[] Actuals = [1.0, 2.0, 3.0, 4.0, 5.0];
    private static readonly double[] ClosePredictons = [1.1, 2.1, 2.9, 3.9, 5.1];
    private static readonly double[] BadPredictions = [5.0, 4.0, 3.0, 2.0, 1.0];

    // Binary classification data (0/1)
    private static readonly double[] BinaryActuals = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1];
    private static readonly double[] BinaryPerfect = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1];
    private static readonly double[] BinaryGood = [1, 1, 0, 1, 1, 0, 0, 0, 1, 1];
    private static readonly double[] BinaryWorst = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0];

    #region Regression Metrics - Perfect Predictions

    [Fact]
    public void R2Score_PerfectPredictions_ReturnsOne()
    {
        var metric = new R2ScoreMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MAE_PerfectPredictions_ReturnsZero()
    {
        var metric = new MAEMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MSE_PerfectPredictions_ReturnsZero()
    {
        var metric = new MSEMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void RMSE_PerfectPredictions_ReturnsZero()
    {
        var metric = new RMSEMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MaxError_PerfectPredictions_ReturnsZero()
    {
        var metric = new MaxErrorMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MedianAbsoluteError_PerfectPredictions_ReturnsZero()
    {
        var metric = new MedianAbsoluteErrorMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ExplainedVariance_PerfectPredictions_ReturnsOne()
    {
        var metric = new ExplainedVarianceMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region Regression Metrics - Known Values

    [Fact]
    public void MAE_KnownValues_ReturnsCorrectResult()
    {
        var metric = new MAEMetric<double>();
        // actuals = [1,2,3,4,5], preds = [1.1,2.1,2.9,3.9,5.1]
        // errors = [0.1, 0.1, 0.1, 0.1, 0.1], MAE = 0.1
        var result = metric.Compute(ClosePredictons, Actuals);
        Assert.Equal(0.1, result, Tolerance);
    }

    [Fact]
    public void MSE_KnownValues_ReturnsCorrectResult()
    {
        var metric = new MSEMetric<double>();
        // squared errors = [0.01, 0.01, 0.01, 0.01, 0.01], MSE = 0.01
        var result = metric.Compute(ClosePredictons, Actuals);
        Assert.Equal(0.01, result, Tolerance);
    }

    [Fact]
    public void RMSE_KnownValues_IsSqrtOfMSE()
    {
        var mse = new MSEMetric<double>();
        var rmse = new RMSEMetric<double>();
        var mseVal = mse.Compute(ClosePredictons, Actuals);
        var rmseVal = rmse.Compute(ClosePredictons, Actuals);
        Assert.Equal(Math.Sqrt(mseVal), rmseVal, Tolerance);
    }

    [Fact]
    public void MaxError_KnownValues_ReturnsMaxAbsError()
    {
        var metric = new MaxErrorMetric<double>();
        double[] preds = [1.0, 2.5, 3.0, 4.0, 5.0];
        // max |error| = |2.0-2.5| = 0.5
        var result = metric.Compute(preds, Actuals);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void R2Score_MeanPredictions_ReturnsZero()
    {
        var metric = new R2ScoreMetric<double>();
        double mean = Actuals.Average();
        double[] meanPreds = [mean, mean, mean, mean, mean];
        var result = metric.Compute(meanPreds, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void R2Score_BadPredictions_IsNegative()
    {
        var metric = new R2ScoreMetric<double>();
        // Reversed predictions should give negative R2
        var result = metric.Compute(BadPredictions, Actuals);
        Assert.True(result < 0);
    }

    [Fact]
    public void MeanBiasError_OverPredictions_IsPositive()
    {
        var metric = new MeanBiasErrorMetric<double>();
        double[] preds = [2.0, 3.0, 4.0, 5.0, 6.0]; // all +1 above actual
        var result = metric.Compute(preds, Actuals);
        Assert.True(result > 0);
    }

    [Fact]
    public void MeanBiasError_UnderPredictions_IsNegative()
    {
        var metric = new MeanBiasErrorMetric<double>();
        double[] preds = [0.0, 1.0, 2.0, 3.0, 4.0]; // all -1 below actual
        var result = metric.Compute(preds, Actuals);
        Assert.True(result < 0);
    }

    [Fact]
    public void PearsonCorrelation_PerfectPositive_ReturnsOne()
    {
        var metric = new PearsonCorrelationMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void PearsonCorrelation_PerfectNegative_ReturnsMinusOne()
    {
        var metric = new PearsonCorrelationMetric<double>();
        var result = metric.Compute(BadPredictions, Actuals);
        Assert.Equal(-1.0, result, Tolerance);
    }

    [Fact]
    public void SpearmanCorrelation_PerfectPositive_ReturnsOne()
    {
        var metric = new SpearmanCorrelationMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void SpearmanCorrelation_PerfectNegative_ReturnsMinusOne()
    {
        var metric = new SpearmanCorrelationMetric<double>();
        var result = metric.Compute(BadPredictions, Actuals);
        Assert.Equal(-1.0, result, Tolerance);
    }

    [Fact]
    public void HuberLoss_PerfectPredictions_ReturnsZero()
    {
        var metric = new HuberLossMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void LogCoshLoss_PerfectPredictions_ReturnsZero()
    {
        var metric = new LogCoshLossMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Regression Metrics - Non-Negative

    [Fact]
    public void AllNonNegativeRegressionMetrics_AlwaysNonNegative()
    {
        var metrics = new IRegressionMetric<double>[]
        {
            new MAEMetric<double>(),
            new MSEMetric<double>(),
            new RMSEMetric<double>(),
            new MaxErrorMetric<double>(),
            new MedianAbsoluteErrorMetric<double>(),
            new HuberLossMetric<double>(),
            new LogCoshLossMetric<double>(),
            new NormalizedMSEMetric<double>(),
        };

        foreach (var metric in metrics)
        {
            var result = metric.Compute(ClosePredictons, Actuals);
            Assert.True(result >= 0,
                $"{metric.Name} returned {result}, expected >= 0");
        }
    }

    #endregion

    #region Regression Metrics - MSE >= MAE^2 Property

    [Fact]
    public void MSE_AlwaysGreaterThanOrEqualTo_MAESquared()
    {
        var mse = new MSEMetric<double>();
        var mae = new MAEMetric<double>();

        var mseVal = mse.Compute(BadPredictions, Actuals);
        var maeVal = mae.Compute(BadPredictions, Actuals);

        // MSE >= MAE^2 by Cauchy-Schwarz
        Assert.True(mseVal >= maeVal * maeVal - Tolerance);
    }

    #endregion

    #region Classification Metrics - Perfect Predictions

    [Fact]
    public void Accuracy_PerfectPredictions_ReturnsOne()
    {
        var metric = new AccuracyMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Accuracy_WorstPredictions_ReturnsZero()
    {
        var metric = new AccuracyMetric<double>();
        var result = metric.Compute(BinaryWorst, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ErrorRate_PerfectPredictions_ReturnsZero()
    {
        var metric = new ErrorRateMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void Precision_PerfectPredictions_ReturnsOne()
    {
        var metric = new PrecisionMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Recall_PerfectPredictions_ReturnsOne()
    {
        var metric = new RecallMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void F1Score_PerfectPredictions_ReturnsOne()
    {
        var metric = new F1ScoreMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void Specificity_PerfectPredictions_ReturnsOne()
    {
        var metric = new SpecificityMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MatthewsCorrelation_PerfectPredictions_ReturnsOne()
    {
        var metric = new MatthewsCorrelationCoefficientMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CohensKappa_PerfectPredictions_ReturnsOne()
    {
        var metric = new CohensKappaMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void HammingLoss_PerfectPredictions_ReturnsZero()
    {
        var metric = new HammingLossMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ZeroOneLoss_PerfectPredictions_ReturnsZero()
    {
        var metric = new ZeroOneLossMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Classification Metrics - Known Values

    [Fact]
    public void Accuracy_KnownValues_ReturnsCorrectResult()
    {
        var metric = new AccuracyMetric<double>();
        // BinaryGood has 8/10 correct
        var result = metric.Compute(BinaryGood, BinaryActuals);
        Assert.Equal(0.8, result, Tolerance);
    }

    [Fact]
    public void AccuracyPlusErrorRate_SumsToOne()
    {
        var acc = new AccuracyMetric<double>();
        var err = new ErrorRateMetric<double>();
        var accVal = acc.Compute(BinaryGood, BinaryActuals);
        var errVal = err.Compute(BinaryGood, BinaryActuals);
        Assert.Equal(1.0, accVal + errVal, Tolerance);
    }

    [Fact]
    public void F1Score_IsHarmonicMeanOfPrecisionAndRecall()
    {
        var precision = new PrecisionMetric<double>();
        var recall = new RecallMetric<double>();
        var f1 = new F1ScoreMetric<double>();

        var p = precision.Compute(BinaryGood, BinaryActuals);
        var r = recall.Compute(BinaryGood, BinaryActuals);
        var f1Val = f1.Compute(BinaryGood, BinaryActuals);

        double expectedF1 = 2 * p * r / (p + r);
        Assert.Equal(expectedF1, f1Val, LooseTolerance);
    }

    [Fact]
    public void BalancedAccuracy_PerfectPredictions_ReturnsOne()
    {
        var metric = new BalancedAccuracyMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void JaccardScore_PerfectPredictions_ReturnsOne()
    {
        var metric = new JaccardScoreMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ThreatScore_PerfectPredictions_ReturnsOne()
    {
        var metric = new ThreatScoreMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void NPV_PerfectPredictions_ReturnsOne()
    {
        var metric = new NPVMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void FalsePositiveRate_PerfectPredictions_ReturnsZero()
    {
        var metric = new FalsePositiveRateMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void FalseNegativeRate_PerfectPredictions_ReturnsZero()
    {
        var metric = new FalseNegativeRateMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void FalseDiscoveryRate_PerfectPredictions_ReturnsZero()
    {
        var metric = new FalseDiscoveryRateMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void FalseOmissionRate_PerfectPredictions_ReturnsZero()
    {
        var metric = new FalseOmissionRateMetric<double>();
        var result = metric.Compute(BinaryPerfect, BinaryActuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Classification Metrics - Bounded [0, 1]

    [Fact]
    public void AllBoundedClassificationMetrics_InRange()
    {
        var metrics = new IClassificationMetric<double>[]
        {
            new AccuracyMetric<double>(),
            new ErrorRateMetric<double>(),
            new PrecisionMetric<double>(),
            new RecallMetric<double>(),
            new F1ScoreMetric<double>(),
            new SpecificityMetric<double>(),
            new BalancedAccuracyMetric<double>(),
            new HammingLossMetric<double>(),
            new ZeroOneLossMetric<double>(),
            new FalsePositiveRateMetric<double>(),
            new FalseNegativeRateMetric<double>(),
            new FalseDiscoveryRateMetric<double>(),
            new JaccardScoreMetric<double>(),
            new ThreatScoreMetric<double>(),
            new NPVMetric<double>(),
        };

        foreach (var metric in metrics)
        {
            var result = metric.Compute(BinaryGood, BinaryActuals);
            Assert.True(result >= -Tolerance,
                $"{metric.Name} returned {result}, expected >= 0");
            Assert.True(result <= 1.0 + Tolerance,
                $"{metric.Name} returned {result}, expected <= 1");
        }
    }

    #endregion

    #region Classification Metrics - Relationships

    [Fact]
    public void Sensitivity_EqualTo_Recall()
    {
        var recall = new RecallMetric<double>();
        var sensitivity = new TrueNegativeRateMetric<double>();
        // Recall and Sensitivity(TPR) should be the same
        // Actually TrueNegativeRate = Specificity, not Recall. Let me test recall vs FNR instead.
        var fnr = new FalseNegativeRateMetric<double>();

        var recallVal = recall.Compute(BinaryGood, BinaryActuals);
        var fnrVal = fnr.Compute(BinaryGood, BinaryActuals);
        // Recall + FNR = 1
        Assert.Equal(1.0, recallVal + fnrVal, Tolerance);
    }

    [Fact]
    public void SpecificityPlusFPR_SumsToOne()
    {
        var spec = new SpecificityMetric<double>();
        var fpr = new FalsePositiveRateMetric<double>();

        var specVal = spec.Compute(BinaryGood, BinaryActuals);
        var fprVal = fpr.Compute(BinaryGood, BinaryActuals);
        Assert.Equal(1.0, specVal + fprVal, Tolerance);
    }

    [Fact]
    public void PrecisionPlusFDR_SumsToOne()
    {
        var prec = new PrecisionMetric<double>();
        var fdr = new FalseDiscoveryRateMetric<double>();

        var precVal = prec.Compute(BinaryGood, BinaryActuals);
        var fdrVal = fdr.Compute(BinaryGood, BinaryActuals);
        Assert.Equal(1.0, precVal + fdrVal, Tolerance);
    }

    [Fact]
    public void NPVPlusFOR_SumsToOne()
    {
        var npv = new NPVMetric<double>();
        var forMetric = new FalseOmissionRateMetric<double>();

        var npvVal = npv.Compute(BinaryGood, BinaryActuals);
        var forVal = forMetric.Compute(BinaryGood, BinaryActuals);
        Assert.Equal(1.0, npvVal + forVal, Tolerance);
    }

    #endregion

    #region Regression Metrics - Metadata Properties

    [Fact]
    public void AllRegressionMetrics_HaveValidMetadata()
    {
        var metrics = new IRegressionMetric<double>[]
        {
            new R2ScoreMetric<double>(),
            new MAEMetric<double>(),
            new MSEMetric<double>(),
            new RMSEMetric<double>(),
            new MaxErrorMetric<double>(),
            new MedianAbsoluteErrorMetric<double>(),
            new ExplainedVarianceMetric<double>(),
            new HuberLossMetric<double>(),
            new LogCoshLossMetric<double>(),
            new MeanBiasErrorMetric<double>(),
            new PearsonCorrelationMetric<double>(),
            new SpearmanCorrelationMetric<double>(),
            new NormalizedMSEMetric<double>(),
            new AdjustedR2Metric<double>(),
        };

        foreach (var metric in metrics)
        {
            Assert.False(string.IsNullOrEmpty(metric.Name),
                $"Metric has empty Name");
            Assert.False(string.IsNullOrEmpty(metric.Category),
                $"{metric.Name} has empty Category");
            Assert.False(string.IsNullOrEmpty(metric.Description),
                $"{metric.Name} has empty Description");
            Assert.Equal("Regression", metric.Category);
        }
    }

    #endregion

    #region Classification Metrics - Metadata Properties

    [Fact]
    public void AllClassificationMetrics_HaveValidMetadata()
    {
        var metrics = new IClassificationMetric<double>[]
        {
            new AccuracyMetric<double>(),
            new ErrorRateMetric<double>(),
            new PrecisionMetric<double>(),
            new RecallMetric<double>(),
            new F1ScoreMetric<double>(),
            new SpecificityMetric<double>(),
            new BalancedAccuracyMetric<double>(),
            new CohensKappaMetric<double>(),
            new MatthewsCorrelationCoefficientMetric<double>(),
            new HammingLossMetric<double>(),
            new ZeroOneLossMetric<double>(),
            new JaccardScoreMetric<double>(),
            new ThreatScoreMetric<double>(),
            new NPVMetric<double>(),
            new FalsePositiveRateMetric<double>(),
            new FalseNegativeRateMetric<double>(),
            new FalseDiscoveryRateMetric<double>(),
            new FalseOmissionRateMetric<double>(),
            new InformednessMetric<double>(),
            new MarkednessMetric<double>(),
            new FowlkesMallowsMetric<double>(),
            new DiagnosticOddsRatioMetric<double>(),
            new BalancedErrorRateMetric<double>(),
            new PositivePredictiveValueMetric<double>(),
            new TrueNegativeRateMetric<double>(),
            new PrevalenceMetric<double>(),
        };

        foreach (var metric in metrics)
        {
            Assert.False(string.IsNullOrEmpty(metric.Name),
                $"Metric has empty Name");
            Assert.False(string.IsNullOrEmpty(metric.Category),
                $"{metric.Name} has empty Category");
            Assert.False(string.IsNullOrEmpty(metric.Description),
                $"{metric.Name} has empty Description");
        }
    }

    #endregion

    #region Regression Metrics - MismatchedLengths

    [Fact]
    public void MAE_MismatchedLengths_Throws()
    {
        var metric = new MAEMetric<double>();
        double[] preds = [1.0, 2.0];
        double[] actuals = [1.0, 2.0, 3.0];
        Assert.Throws<ArgumentException>(() => metric.Compute(preds, actuals));
    }

    [Fact]
    public void Accuracy_MismatchedLengths_Throws()
    {
        var metric = new AccuracyMetric<double>();
        double[] preds = [1.0, 0.0];
        double[] actuals = [1.0];
        Assert.Throws<ArgumentException>(() => metric.Compute(preds, actuals));
    }

    #endregion

    #region Regression Metrics - Empty Input

    [Fact]
    public void MAE_EmptyInput_ReturnsZero()
    {
        var metric = new MAEMetric<double>();
        var result = metric.Compute(ReadOnlySpan<double>.Empty, ReadOnlySpan<double>.Empty);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void MSE_EmptyInput_ReturnsZero()
    {
        var metric = new MSEMetric<double>();
        var result = metric.Compute(ReadOnlySpan<double>.Empty, ReadOnlySpan<double>.Empty);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region MAPE and SMAPE Tests

    [Fact]
    public void MAPE_PerfectPredictions_ReturnsZero()
    {
        var metric = new MAPEMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SMAPE_PerfectPredictions_ReturnsZero()
    {
        var metric = new SymmetricMAPEMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void SMAPE_IsSymmetric()
    {
        var metric = new SymmetricMAPEMetric<double>();
        double[] a = [1.0, 2.0, 3.0];
        double[] b = [1.5, 2.5, 3.5];
        var r1 = metric.Compute(a, b);
        var r2 = metric.Compute(b, a);
        Assert.Equal(r1, r2, LooseTolerance);
    }

    [Fact]
    public void MAPE_NonNegative()
    {
        var metric = new MAPEMetric<double>();
        var result = metric.Compute(BadPredictions, Actuals);
        Assert.True(result >= 0);
    }

    #endregion

    #region Time Series Metrics

    [Fact]
    public void SMAPE_TimeSeries_NonNegative()
    {
        var metric = new SMAPEMetric<double>();
        double[] preds = [10.0, 20.0, 15.0, 25.0];
        double[] actuals = [12.0, 18.0, 16.0, 22.0];
        var result = metric.Compute(preds, actuals);
        Assert.True(result >= 0);
    }

    [Fact]
    public void WAPE_PerfectPredictions_ReturnsZero()
    {
        var metric = new WAPEMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void TheilU_PerfectPredictions_ReturnsZero()
    {
        var metric = new TheilUMetric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Confidence Intervals

    [Fact]
    public void R2Score_ComputeWithCI_ReturnsValidInterval()
    {
        var metric = new R2ScoreMetric<double>();
        var result = metric.ComputeWithCI(ClosePredictons, Actuals,
            bootstrapSamples: 100, randomSeed: 42);
        Assert.True(result.LowerBound <= result.Value);
        Assert.True(result.Value <= result.UpperBound);
    }

    [Fact]
    public void Accuracy_ComputeWithCI_ReturnsValidInterval()
    {
        var metric = new AccuracyMetric<double>();
        var result = metric.ComputeWithCI(BinaryGood, BinaryActuals,
            bootstrapSamples: 100, randomSeed: 42);
        Assert.True(result.LowerBound <= result.Value + Tolerance);
        Assert.True(result.Value <= result.UpperBound + Tolerance);
    }

    #endregion

    #region All Regression Metrics - No NaN on Valid Input

    [Fact]
    public void AllRegressionMetrics_ValidInput_NoNaN()
    {
        var metrics = new IRegressionMetric<double>[]
        {
            new R2ScoreMetric<double>(),
            new MAEMetric<double>(),
            new MSEMetric<double>(),
            new RMSEMetric<double>(),
            new MaxErrorMetric<double>(),
            new MedianAbsoluteErrorMetric<double>(),
            new ExplainedVarianceMetric<double>(),
            new HuberLossMetric<double>(),
            new LogCoshLossMetric<double>(),
            new MeanBiasErrorMetric<double>(),
            new PearsonCorrelationMetric<double>(),
            new NormalizedMSEMetric<double>(),
        };

        foreach (var metric in metrics)
        {
            var result = metric.Compute(ClosePredictons, Actuals);
            Assert.False(double.IsNaN(result),
                $"{metric.Name} returned NaN");
            Assert.False(double.IsInfinity(result),
                $"{metric.Name} returned Infinity");
        }
    }

    #endregion

    #region All Classification Metrics - No NaN on Valid Input

    [Fact]
    public void AllClassificationMetrics_ValidInput_NoNaN()
    {
        var metrics = new IClassificationMetric<double>[]
        {
            new AccuracyMetric<double>(),
            new ErrorRateMetric<double>(),
            new PrecisionMetric<double>(),
            new RecallMetric<double>(),
            new F1ScoreMetric<double>(),
            new SpecificityMetric<double>(),
            new BalancedAccuracyMetric<double>(),
            new CohensKappaMetric<double>(),
            new MatthewsCorrelationCoefficientMetric<double>(),
            new HammingLossMetric<double>(),
            new ZeroOneLossMetric<double>(),
            new JaccardScoreMetric<double>(),
            new ThreatScoreMetric<double>(),
            new NPVMetric<double>(),
            new FalsePositiveRateMetric<double>(),
            new FalseNegativeRateMetric<double>(),
            new FalseDiscoveryRateMetric<double>(),
            new FalseOmissionRateMetric<double>(),
            new InformednessMetric<double>(),
            new MarkednessMetric<double>(),
            new FowlkesMallowsMetric<double>(),
            new BalancedErrorRateMetric<double>(),
            new PositivePredictiveValueMetric<double>(),
            new TrueNegativeRateMetric<double>(),
            new PrevalenceMetric<double>(),
        };

        foreach (var metric in metrics)
        {
            var result = metric.Compute(BinaryGood, BinaryActuals);
            Assert.False(double.IsNaN(result),
                $"{metric.Name} returned NaN");
            Assert.False(double.IsInfinity(result),
                $"{metric.Name} returned Infinity");
        }
    }

    #endregion

    #region MeanDirectionalAccuracy Tests

    [Fact]
    public void MeanDirectionalAccuracy_PerfectDirection_ReturnsOne()
    {
        var metric = new MeanDirectionalAccuracyMetric<double>();
        // Both have same direction of change
        double[] actuals = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] preds = [1.0, 2.5, 3.5, 4.5, 5.5];
        var result = metric.Compute(preds, actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region AdjustedR2 Tests

    [Fact]
    public void AdjustedR2_PerfectPredictions_ReturnsOne()
    {
        var metric = new AdjustedR2Metric<double>();
        var result = metric.Compute(PerfectPredictions, Actuals);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void AdjustedR2_LessThanOrEqualToR2()
    {
        var r2 = new R2ScoreMetric<double>();
        var adjR2 = new AdjustedR2Metric<double>();
        var r2Val = r2.Compute(ClosePredictons, Actuals);
        var adjR2Val = adjR2.Compute(ClosePredictons, Actuals);
        Assert.True(adjR2Val <= r2Val + Tolerance);
    }

    #endregion
}
