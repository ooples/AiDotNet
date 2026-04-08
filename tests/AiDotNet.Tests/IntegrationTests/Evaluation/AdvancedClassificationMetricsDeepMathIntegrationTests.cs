using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics.Classification;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep math-correctness integration tests for advanced classification metrics:
/// FBetaScore, NPV, DiagnosticOddsRatio, FowlkesMallows, Informedness, Markedness,
/// PositiveLikelihoodRatio, NegativeLikelihoodRatio, PrevalenceThreshold,
/// JaccardScore, ThreatScore, and cross-metric mathematical identities.
/// All expected values are hand-calculated from first principles.
/// </summary>
public class AdvancedClassificationMetricsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    // Shared test dataset 1: TP=3, TN=3, FP=2, FN=2 (symmetric confusion matrix)
    // Predictions: [1,1,0,0,1,0,1,1,0,0]
    // Actuals:     [1,0,0,1,1,0,0,1,1,0]
    private static readonly double[] SymPred = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0];
    private static readonly double[] SymActual = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0];
    // TP=3, TN=3, FP=2, FN=2
    // Sensitivity=3/5=0.6, Specificity=3/5=0.6
    // Precision=3/5=0.6, NPV=3/5=0.6, FPR=0.4, FNR=0.4

    // Shared test dataset 2: Asymmetric confusion matrix
    // TP=2, TN=3, FP=2, FN=1
    private static readonly double[] AsymPred = [1, 1, 1, 1, 0, 0, 0, 0];
    private static readonly double[] AsymActual = [1, 1, 0, 0, 0, 0, 0, 1];
    // Sensitivity=2/3, Specificity=3/5=0.6
    // Precision=2/4=0.5, NPV=3/4=0.75, FPR=2/5=0.4, FNR=1/3

    #region FBetaScore - Binary

    [Fact]
    public void FBeta_F1_HandCalculated_Binary()
    {
        // TP=2, FP=2, FN=1 → Precision=0.5, Recall=2/3
        // F1 = 2 * 0.5 * (2/3) / (0.5 + 2/3) = (2/3) / (7/6) = 4/7
        var metric = new FBetaScoreMetric<double>(1.0);
        var result = metric.Compute(AsymPred, AsymActual);
        Assert.Equal(4.0 / 7.0, result, Tolerance);
    }

    [Fact]
    public void FBeta_F2_WeightsRecallMore()
    {
        // TP=2, FP=2, FN=1 → Precision=0.5, Recall=2/3
        // F2 = (1+4) * 0.5 * (2/3) / (4*0.5 + 2/3) = 5*(1/3) / (2+2/3) = (5/3)/(8/3) = 5/8
        var metric = new FBetaScoreMetric<double>(2.0);
        var result = metric.Compute(AsymPred, AsymActual);
        Assert.Equal(5.0 / 8.0, result, Tolerance);
    }

    [Fact]
    public void FBeta_F05_WeightsPrecisionMore()
    {
        // TP=2, FP=2, FN=1 → Precision=0.5, Recall=2/3
        // F0.5 = (1+0.25) * 0.5 * (2/3) / (0.25*0.5 + 2/3) = 1.25*(1/3) / (1/8+2/3)
        // = (5/12) / (3/24+16/24) = (5/12) / (19/24) = (5/12)*(24/19) = 10/19
        var metric = new FBetaScoreMetric<double>(0.5);
        var result = metric.Compute(AsymPred, AsymActual);
        Assert.Equal(10.0 / 19.0, result, Tolerance);
    }

    [Fact]
    public void FBeta_F1_GreaterThanF05_WhenRecallGreaterThanPrecision()
    {
        // When recall > precision, F1 weights recall more than F0.5 does
        // So F1 > F0.5 when recall > precision
        // Here precision=0.5, recall=2/3, so recall > precision
        var f1 = new FBetaScoreMetric<double>(1.0);
        var f05 = new FBetaScoreMetric<double>(0.5);
        var f2 = new FBetaScoreMetric<double>(2.0);

        double f1Val = f1.Compute(AsymPred, AsymActual);
        double f05Val = f05.Compute(AsymPred, AsymActual);
        double f2Val = f2.Compute(AsymPred, AsymActual);

        // F2 > F1 > F0.5 when recall > precision
        Assert.True(f2Val > f1Val, $"F2={f2Val} should be > F1={f1Val}");
        Assert.True(f1Val > f05Val, $"F1={f1Val} should be > F0.5={f05Val}");
    }

    [Fact]
    public void FBeta_PerfectPrediction_IsOne()
    {
        double[] preds = [1, 0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0, 1];
        var metric = new FBetaScoreMetric<double>(1.0);
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void FBeta_NoPredictedPositives_IsZero()
    {
        double[] preds = [0, 0, 0, 0];
        double[] actuals = [1, 1, 0, 0];
        // TP=0, FP=0, FN=2 → Precision undefined (0/0→0), Recall=0
        var metric = new FBetaScoreMetric<double>(1.0);
        Assert.Equal(0.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void FBeta_InvalidBeta_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(double.PositiveInfinity));
    }

    #endregion

    #region FBetaScore - Multi-class

    [Fact]
    public void FBeta_MacroAverage_HandCalculated()
    {
        // 3-class problem:
        // Actuals:     [0, 0, 0, 0, 1, 1, 2, 2, 2]
        // Predictions: [0, 0, 0, 1, 1, 0, 2, 2, 1]
        // Class 0: TP=3, FP=1, FN=1 → P=3/4, R=3/4, F1=3/4
        // Class 1: TP=1, FP=2, FN=1 → P=1/3, R=1/2, F1=2/5
        // Class 2: TP=2, FP=0, FN=1 → P=1, R=2/3, F1=4/5
        // Macro F1 = (3/4 + 2/5 + 4/5) / 3 = (0.75 + 0.4 + 0.8) / 3 = 1.95/3 = 0.65
        double[] actuals = [0, 0, 0, 0, 1, 1, 2, 2, 2];
        double[] preds = [0, 0, 0, 1, 1, 0, 2, 2, 1];

        var metric = new FBetaScoreMetric<double>(1.0, AveragingMethod.Macro);
        var result = metric.Compute(preds, actuals);
        Assert.Equal(0.65, result, Tolerance);
    }

    [Fact]
    public void FBeta_WeightedAverage_HandCalculated()
    {
        // Same data as macro test
        // Weighted F1 = (4*(3/4) + 2*(2/5) + 3*(4/5)) / (4+2+3)
        //             = (3 + 0.8 + 2.4) / 9 = 6.2/9
        double[] actuals = [0, 0, 0, 0, 1, 1, 2, 2, 2];
        double[] preds = [0, 0, 0, 1, 1, 0, 2, 2, 1];

        var metric = new FBetaScoreMetric<double>(1.0, AveragingMethod.Weighted);
        var result = metric.Compute(preds, actuals);
        Assert.Equal(6.2 / 9.0, result, Tolerance);
    }

    [Fact]
    public void FBeta_MicroAverage_HandCalculated()
    {
        // Same data: Total TP=6, Total FP=3, Total FN=3
        // Micro-Precision = 6/9 = 2/3, Micro-Recall = 6/9 = 2/3
        // Micro-F1 = 2*(2/3)*(2/3) / ((2/3)+(2/3)) = 2/3
        double[] actuals = [0, 0, 0, 0, 1, 1, 2, 2, 2];
        double[] preds = [0, 0, 0, 1, 1, 0, 2, 2, 1];

        var metric = new FBetaScoreMetric<double>(1.0, AveragingMethod.Micro);
        var result = metric.Compute(preds, actuals);
        Assert.Equal(2.0 / 3.0, result, Tolerance);
    }

    [Fact]
    public void FBeta_MacroVsWeighted_DifferWhenClassesImbalanced()
    {
        // Macro treats all classes equally, weighted weights by support
        double[] actuals = [0, 0, 0, 0, 1, 1, 2, 2, 2];
        double[] preds = [0, 0, 0, 1, 1, 0, 2, 2, 1];

        var macro = new FBetaScoreMetric<double>(1.0, AveragingMethod.Macro);
        var weighted = new FBetaScoreMetric<double>(1.0, AveragingMethod.Weighted);
        var macroVal = macro.Compute(preds, actuals);
        var weightedVal = weighted.Compute(preds, actuals);

        // They should be different when class supports differ
        Assert.NotEqual(macroVal, weightedVal);
    }

    #endregion

    #region NPV

    [Fact]
    public void NPV_HandCalculated_Symmetric()
    {
        // TP=3, TN=3, FP=2, FN=2
        // Negative predictions: TN + FN = 5
        // NPV = TN / (TN + FN) = 3/5 = 0.6
        var metric = new NPVMetric<double>();
        var result = metric.Compute(SymPred, SymActual);
        Assert.Equal(0.6, result, Tolerance);
    }

    [Fact]
    public void NPV_HandCalculated_Asymmetric()
    {
        // TP=2, TN=3, FP=2, FN=1
        // NPV = TN / (TN + FN) = 3/4 = 0.75
        var metric = new NPVMetric<double>();
        var result = metric.Compute(AsymPred, AsymActual);
        Assert.Equal(0.75, result, Tolerance);
    }

    [Fact]
    public void NPV_AllPositivePredictions_ReturnsZero()
    {
        // No negative predictions → NPV undefined → returns 0
        double[] preds = [1, 1, 1, 1];
        double[] actuals = [1, 0, 1, 0];
        var metric = new NPVMetric<double>();
        Assert.Equal(0.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void NPV_PerfectPrediction_IsOne()
    {
        double[] preds = [1, 0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0, 1];
        var metric = new NPVMetric<double>();
        // All negative predictions are correct (TN=2, FN=0), NPV = 2/2 = 1
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region DiagnosticOddsRatio

    [Fact]
    public void DOR_HandCalculated_WithHaldaneCorrection()
    {
        // TP=2, TN=3, FP=2, FN=1
        // Haldane-Anscombe correction: add 0.5 to each cell
        // DOR = (2.5 * 3.5) / (2.5 * 1.5) = 8.75 / 3.75 = 2.333...
        var metric = new DiagnosticOddsRatioMetric<double>();
        var result = metric.Compute(AsymPred, AsymActual);
        Assert.Equal(8.75 / 3.75, result, Tolerance);
    }

    [Fact]
    public void DOR_HandCalculated_Symmetric()
    {
        // TP=3, TN=3, FP=2, FN=2
        // DOR = (3.5 * 3.5) / (2.5 * 2.5) = 12.25 / 6.25 = 1.96
        var metric = new DiagnosticOddsRatioMetric<double>();
        var result = metric.Compute(SymPred, SymActual);
        Assert.Equal(12.25 / 6.25, result, Tolerance);
    }

    [Fact]
    public void DOR_PerfectPrediction_VeryHigh()
    {
        double[] preds = [1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 0, 1, 0];
        // TP=3, TN=3, FP=0, FN=0
        // DOR = (3.5 * 3.5) / (0.5 * 0.5) = 12.25 / 0.25 = 49
        var metric = new DiagnosticOddsRatioMetric<double>();
        Assert.Equal(49.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void DOR_HaldaneCorrection_PreventsInfinity()
    {
        // Without Haldane correction, FP=0 and FN=0 would give division by zero
        // With correction, the result is finite
        double[] preds = [1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 0];
        var metric = new DiagnosticOddsRatioMetric<double>();
        double result = metric.Compute(preds, actuals);
        Assert.True((!double.IsNaN(result) && !double.IsInfinity(result)), "DOR should be finite due to Haldane correction");
    }

    [Fact]
    public void DOR_AllWrong_VeryLow()
    {
        double[] preds = [0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0];
        // TP=0, TN=0, FP=2, FN=2
        // DOR = (0.5 * 0.5) / (2.5 * 2.5) = 0.25 / 6.25 = 0.04
        var metric = new DiagnosticOddsRatioMetric<double>();
        Assert.Equal(0.04, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region FowlkesMallows

    [Fact]
    public void FowlkesMallows_HandCalculated_Symmetric()
    {
        // TP=3, FP=2, FN=2
        // Precision = 3/5 = 0.6, Recall = 3/5 = 0.6
        // FM = sqrt(0.6 * 0.6) = sqrt(0.36) = 0.6
        var metric = new FowlkesMallowsMetric<double>();
        Assert.Equal(0.6, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void FowlkesMallows_HandCalculated_Asymmetric()
    {
        // TP=2, FP=2, FN=1
        // Precision = 2/4 = 0.5, Recall = 2/3
        // FM = sqrt(0.5 * 2/3) = sqrt(1/3) = 1/sqrt(3)
        var metric = new FowlkesMallowsMetric<double>();
        Assert.Equal(1.0 / Math.Sqrt(3.0), metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void FowlkesMallows_IsGeometricMean_LessOrEqualArithmeticMean()
    {
        // AM-GM inequality: geometric mean <= arithmetic mean
        // FM = sqrt(P*R) <= (P+R)/2
        var metric = new FowlkesMallowsMetric<double>();
        double fm = metric.Compute(AsymPred, AsymActual);

        double precision = 0.5; // TP=2, FP=2
        double recall = 2.0 / 3.0; // TP=2, FN=1
        double arithmeticMean = (precision + recall) / 2.0;

        Assert.True(fm <= arithmeticMean + Tolerance,
            $"FM={fm} should be <= AM={arithmeticMean} by AM-GM inequality");
    }

    [Fact]
    public void FowlkesMallows_PerfectPrediction_IsOne()
    {
        double[] preds = [1, 0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0, 1];
        var metric = new FowlkesMallowsMetric<double>();
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void FowlkesMallows_NoTruePositives_IsZero()
    {
        double[] preds = [0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0];
        // TP=0, FP=2, FN=2 → P=0, R=0 → FM=0
        var metric = new FowlkesMallowsMetric<double>();
        Assert.Equal(0.0, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region Informedness (Youden's J)

    [Fact]
    public void Informedness_HandCalculated_Symmetric()
    {
        // TP=3, TN=3, FP=2, FN=2
        // TPR = 3/5 = 0.6, TNR = 3/5 = 0.6
        // Informedness = 0.6 + 0.6 - 1 = 0.2
        var metric = new InformednessMetric<double>();
        Assert.Equal(0.2, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void Informedness_HandCalculated_Asymmetric()
    {
        // TP=2, TN=3, FP=2, FN=1
        // TPR = 2/3, TNR = 3/5
        // Informedness = 2/3 + 3/5 - 1 = 10/15 + 9/15 - 15/15 = 4/15
        var metric = new InformednessMetric<double>();
        Assert.Equal(4.0 / 15.0, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void Informedness_PerfectPrediction_IsOne()
    {
        double[] preds = [1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 0, 1, 0];
        var metric = new InformednessMetric<double>();
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void Informedness_RandomClassifier_IsZero()
    {
        // A classifier that predicts all same class has TPR=1, TNR=0 or TPR=0, TNR=1
        // Either way Informedness = 0
        double[] preds = [1, 1, 1, 1, 1];
        double[] actuals = [1, 0, 1, 0, 1];
        // TP=3, FP=2, FN=0, TN=0
        // TPR = 3/3 = 1, TNR = 0/2 = 0
        // Informedness = 1 + 0 - 1 = 0
        var metric = new InformednessMetric<double>();
        Assert.Equal(0.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void Informedness_InvertedPredictions_IsMinusOne()
    {
        double[] preds = [0, 1, 0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0, 1, 0];
        // TP=0, TN=0, FP=3, FN=3
        // TPR = 0, TNR = 0
        // Informedness = 0 + 0 - 1 = -1
        var metric = new InformednessMetric<double>();
        Assert.Equal(-1.0, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region Markedness

    [Fact]
    public void Markedness_HandCalculated_Symmetric()
    {
        // TP=3, TN=3, FP=2, FN=2
        // PPV = 3/5 = 0.6, NPV = 3/5 = 0.6
        // Markedness = 0.6 + 0.6 - 1 = 0.2
        var metric = new MarkednessMetric<double>();
        Assert.Equal(0.2, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void Markedness_HandCalculated_Asymmetric()
    {
        // TP=2, TN=3, FP=2, FN=1
        // PPV = 2/4 = 0.5, NPV = 3/4 = 0.75
        // Markedness = 0.5 + 0.75 - 1 = 0.25
        var metric = new MarkednessMetric<double>();
        Assert.Equal(0.25, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void Markedness_PerfectPrediction_IsOne()
    {
        double[] preds = [1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 0, 1, 0];
        var metric = new MarkednessMetric<double>();
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void Markedness_AllSamePrediction_Negative()
    {
        double[] preds = [1, 1, 1, 1, 1];
        double[] actuals = [1, 0, 1, 0, 1];
        // TP=3, FP=2, FN=0, TN=0
        // PPV = 3/5 = 0.6, NPV = 0/(0+0) → 0 (no negative predictions)
        // Markedness = 0.6 + 0 - 1 = -0.4
        var metric = new MarkednessMetric<double>();
        Assert.Equal(-0.4, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region Positive Likelihood Ratio (LR+)

    [Fact]
    public void LRPlus_HandCalculated_Symmetric()
    {
        // TP=3, TN=3, FP=2, FN=2
        // Sensitivity = 3/5 = 0.6, FPR = 2/5 = 0.4
        // LR+ = 0.6 / 0.4 = 1.5
        var metric = new PositiveLikelihoodRatioMetric<double>();
        Assert.Equal(1.5, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void LRPlus_HandCalculated_Asymmetric()
    {
        // TP=2, TN=3, FP=2, FN=1
        // Sensitivity = 2/3, FPR = 2/5 = 0.4
        // LR+ = (2/3) / 0.4 = (2/3) / (2/5) = (2/3)*(5/2) = 5/3
        var metric = new PositiveLikelihoodRatioMetric<double>();
        Assert.Equal(5.0 / 3.0, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void LRPlus_PerfectSpecificity_ReturnsMaxValue()
    {
        // No false positives → FPR = 0 → LR+ = infinity
        double[] preds = [1, 0, 0, 0];
        double[] actuals = [1, 0, 0, 0];
        // TP=1, TN=3, FP=0, FN=0 → FPR=0
        var metric = new PositiveLikelihoodRatioMetric<double>();
        Assert.Equal(double.MaxValue, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void LRPlus_RandomClassifier_IsOne()
    {
        // For a random classifier, LR+ ≈ 1 (sensitivity ≈ FPR)
        // Construct: sensitivity=0.5, specificity=0.5
        double[] preds = [1, 0, 1, 0];
        double[] actuals = [1, 1, 0, 0];
        // TP=1, FN=1, FP=1, TN=1
        // Sensitivity=0.5, FPR=0.5, LR+=1
        var metric = new PositiveLikelihoodRatioMetric<double>();
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region Negative Likelihood Ratio (LR-)

    [Fact]
    public void LRMinus_HandCalculated_Symmetric()
    {
        // TP=3, TN=3, FP=2, FN=2
        // FNR = 2/5 = 0.4, Specificity = 3/5 = 0.6
        // LR- = 0.4 / 0.6 = 2/3
        var metric = new NegativeLikelihoodRatioMetric<double>();
        Assert.Equal(2.0 / 3.0, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void LRMinus_HandCalculated_Asymmetric()
    {
        // TP=2, TN=3, FP=2, FN=1
        // FNR = 1/3, Specificity = 3/5
        // LR- = (1/3) / (3/5) = (1/3)*(5/3) = 5/9
        var metric = new NegativeLikelihoodRatioMetric<double>();
        Assert.Equal(5.0 / 9.0, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void LRMinus_PerfectSensitivity_IsZero()
    {
        // No false negatives → FNR = 0 → LR- = 0
        double[] preds = [1, 1, 1, 0, 0];
        double[] actuals = [1, 1, 1, 0, 0];
        // TP=3, TN=2, FP=0, FN=0 → FNR=0
        var metric = new NegativeLikelihoodRatioMetric<double>();
        Assert.Equal(0.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void LRMinus_RandomClassifier_IsOne()
    {
        // For random classifier: FNR ≈ Specificity → LR- ≈ 1
        double[] preds = [1, 0, 1, 0];
        double[] actuals = [1, 1, 0, 0];
        // FNR=0.5, Specificity=0.5, LR-=1
        var metric = new NegativeLikelihoodRatioMetric<double>();
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region PrevalenceThreshold

    [Fact]
    public void PrevalenceThreshold_HandCalculated_Symmetric()
    {
        // TP=3, TN=3, FP=2, FN=2
        // TPR = 0.6, FPR = 0.4
        // PT = sqrt(0.4) / (sqrt(0.6) + sqrt(0.4))
        double expected = Math.Sqrt(0.4) / (Math.Sqrt(0.6) + Math.Sqrt(0.4));
        var metric = new PrevalenceThresholdMetric<double>();
        Assert.Equal(expected, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void PrevalenceThreshold_HandCalculated_Asymmetric()
    {
        // TP=2, TN=3, FP=2, FN=1
        // TPR = 2/3, FPR = 2/5
        // PT = sqrt(0.4) / (sqrt(2/3) + sqrt(0.4))
        double tpr = 2.0 / 3.0;
        double fpr = 0.4;
        double expected = Math.Sqrt(fpr) / (Math.Sqrt(tpr) + Math.Sqrt(fpr));
        var metric = new PrevalenceThresholdMetric<double>();
        Assert.Equal(expected, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void PrevalenceThreshold_PerfectClassifier_IsZero()
    {
        // Perfect: TPR=1, FPR=0 → PT = sqrt(0)/(sqrt(1)+sqrt(0)) = 0/1 = 0
        double[] preds = [1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 0];
        var metric = new PrevalenceThresholdMetric<double>();
        Assert.Equal(0.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void PrevalenceThreshold_RandomClassifier_IsHalf()
    {
        // Random: TPR=0.5, FPR=0.5
        // PT = sqrt(0.5)/(sqrt(0.5)+sqrt(0.5)) = 1/2 = 0.5
        double[] preds = [1, 0, 1, 0];
        double[] actuals = [1, 1, 0, 0];
        var metric = new PrevalenceThresholdMetric<double>();
        Assert.Equal(0.5, metric.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region JaccardScore and ThreatScore

    [Fact]
    public void JaccardScore_HandCalculated_Symmetric()
    {
        // TP=3, FP=2, FN=2
        // Jaccard = 3 / (3+2+2) = 3/7
        var metric = new JaccardScoreMetric<double>();
        Assert.Equal(3.0 / 7.0, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void JaccardScore_HandCalculated_Asymmetric()
    {
        // TP=2, FP=2, FN=1
        // Jaccard = 2 / (2+2+1) = 2/5
        var metric = new JaccardScoreMetric<double>();
        Assert.Equal(0.4, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void ThreatScore_HandCalculated_Symmetric()
    {
        // TP=3, FP=2, FN=2
        // TS = 3 / (3+2+2) = 3/7
        var metric = new ThreatScoreMetric<double>();
        Assert.Equal(3.0 / 7.0, metric.Compute(SymPred, SymActual), Tolerance);
    }

    [Fact]
    public void ThreatScore_HandCalculated_Asymmetric()
    {
        // TP=2, FP=2, FN=1
        // TS = 2 / (2+2+1) = 2/5
        var metric = new ThreatScoreMetric<double>();
        Assert.Equal(0.4, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    [Fact]
    public void JaccardScore_Equals_ThreatScore_Always()
    {
        // Jaccard = TP/(TP+FP+FN) = ThreatScore = TP/(TP+FN+FP) - same formula
        var jaccard = new JaccardScoreMetric<double>();
        var threat = new ThreatScoreMetric<double>();

        // Test on multiple datasets
        double[][] predSets = [SymPred, AsymPred, [1, 0, 1, 0], [1, 1, 1, 0, 0]];
        double[][] actualSets = [SymActual, AsymActual, [1, 1, 0, 0], [1, 0, 1, 1, 0]];

        for (int i = 0; i < predSets.Length; i++)
        {
            double jVal = jaccard.Compute(predSets[i], actualSets[i]);
            double tVal = threat.Compute(predSets[i], actualSets[i]);
            Assert.Equal(jVal, tVal, Tolerance);
        }
    }

    [Fact]
    public void JaccardScore_AllNegatives_IsOne()
    {
        // TP=0, FP=0, FN=0 → denominator=0 → returns 1 (all correct negatives)
        double[] preds = [0, 0, 0, 0];
        double[] actuals = [0, 0, 0, 0];
        var metric = new JaccardScoreMetric<double>();
        Assert.Equal(1.0, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void JaccardScore_RelationToF1()
    {
        // Jaccard = F1 / (2 - F1) (mathematical identity)
        // Equivalently: F1 = 2*J / (1+J)
        var jaccard = new JaccardScoreMetric<double>();
        var f1 = new FBetaScoreMetric<double>(1.0);

        double jVal = jaccard.Compute(AsymPred, AsymActual);
        double f1Val = f1.Compute(AsymPred, AsymActual);

        double jFromF1 = f1Val / (2.0 - f1Val);
        Assert.Equal(jVal, jFromF1, Tolerance);
    }

    #endregion

    #region Cross-Metric Mathematical Identities

    [Fact]
    public void MCC_Squared_Equals_Informedness_Times_Markedness()
    {
        // Mathematical identity: MCC^2 = Informedness * Markedness
        // Both have numerator (TP*TN - FP*FN) and product of all marginals in denominator
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        var informedness = new InformednessMetric<double>();
        var markedness = new MarkednessMetric<double>();

        double mccVal = mcc.Compute(AsymPred, AsymActual);
        double infVal = informedness.Compute(AsymPred, AsymActual);
        double markVal = markedness.Compute(AsymPred, AsymActual);

        // MCC^2 should equal Informedness * Markedness
        Assert.Equal(mccVal * mccVal, infVal * markVal, Tolerance);
    }

    [Fact]
    public void MCC_Squared_Equals_Informedness_Times_Markedness_SymmetricCase()
    {
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        var informedness = new InformednessMetric<double>();
        var markedness = new MarkednessMetric<double>();

        double mccVal = mcc.Compute(SymPred, SymActual);
        double infVal = informedness.Compute(SymPred, SymActual);
        double markVal = markedness.Compute(SymPred, SymActual);

        Assert.Equal(mccVal * mccVal, infVal * markVal, Tolerance);
    }

    [Fact]
    public void DOR_Equals_LRPlus_Over_LRMinus_WithoutCorrection()
    {
        // DOR = LR+ / LR- (this identity holds without Haldane correction)
        // Since DOR uses Haldane correction but LR+ and LR- don't,
        // the identity DOR = LR+/LR- only holds exactly without correction
        // We verify the raw DOR (TP*TN)/(FP*FN) = LR+/LR-
        var lrPlus = new PositiveLikelihoodRatioMetric<double>();
        var lrMinus = new NegativeLikelihoodRatioMetric<double>();

        double lrPlusVal = lrPlus.Compute(AsymPred, AsymActual);
        double lrMinusVal = lrMinus.Compute(AsymPred, AsymActual);

        // Raw DOR without correction: TP=2, TN=3, FP=2, FN=1
        // DOR_raw = (2*3)/(2*1) = 3
        double rawDOR = 3.0;
        Assert.Equal(rawDOR, lrPlusVal / lrMinusVal, Tolerance);
    }

    [Fact]
    public void Informedness_Is_Dual_Of_Markedness()
    {
        // Both have same numerator (TP*TN - FP*FN)
        // Informedness = (TP*TN - FP*FN) / ((TP+FN)*(TN+FP))
        // Markedness = (TP*TN - FP*FN) / ((TP+FP)*(TN+FN))
        // For symmetric confusion matrix (TP=TN, FP=FN), they should be equal
        var informedness = new InformednessMetric<double>();
        var markedness = new MarkednessMetric<double>();

        double infVal = informedness.Compute(SymPred, SymActual);
        double markVal = markedness.Compute(SymPred, SymActual);

        // TP=3, TN=3, FP=2, FN=2 → symmetric marginals → equal
        Assert.Equal(infVal, markVal, Tolerance);
    }

    [Fact]
    public void Informedness_Plus_One_Equals_TPR_Plus_TNR()
    {
        // Informedness = TPR + TNR - 1, so Informedness + 1 = TPR + TNR
        var informedness = new InformednessMetric<double>();
        double infVal = informedness.Compute(AsymPred, AsymActual);

        // TPR = 2/3, TNR = 3/5
        double tprPlusTnr = 2.0 / 3.0 + 3.0 / 5.0;
        Assert.Equal(tprPlusTnr, infVal + 1.0, Tolerance);
    }

    [Fact]
    public void FowlkesMallows_GreaterThan_F1_WhenPrecisionNearRecall()
    {
        // Geometric mean >= Harmonic mean (by AM-GM-HM inequality)
        // So FM = sqrt(P*R) >= 2PR/(P+R) = F1
        // This holds with equality iff P = R
        var fm = new FowlkesMallowsMetric<double>();
        var f1 = new FBetaScoreMetric<double>(1.0);

        double fmVal = fm.Compute(AsymPred, AsymActual);
        double f1Val = f1.Compute(AsymPred, AsymActual);

        // GM >= HM always
        Assert.True(fmVal >= f1Val - Tolerance,
            $"FM={fmVal} should be >= F1={f1Val} by GM-HM inequality");
    }

    [Fact]
    public void LRPlus_Times_LRMinus_Identity()
    {
        // LR+ * LR- = (TPR/FPR) * (FNR/TNR) = (TPR * FNR) / (FPR * TNR)
        // For TP=2, TN=3, FP=2, FN=1:
        // = (2/3 * 1/3) / (2/5 * 3/5) = (2/9) / (6/25) = (2/9)*(25/6) = 50/54 = 25/27
        var lrPlus = new PositiveLikelihoodRatioMetric<double>();
        var lrMinus = new NegativeLikelihoodRatioMetric<double>();

        double product = lrPlus.Compute(AsymPred, AsymActual) * lrMinus.Compute(AsymPred, AsymActual);
        Assert.Equal(25.0 / 27.0, product, Tolerance);
    }

    #endregion

    #region Balanced Accuracy vs Informedness Relationship

    [Fact]
    public void BalancedAccuracy_Equals_HalfInformedness_Plus_Half()
    {
        // BalancedAccuracy = (TPR + TNR) / 2 = (Informedness + 1) / 2
        var balAcc = new BalancedAccuracyMetric<double>();
        var informedness = new InformednessMetric<double>();

        double balAccVal = balAcc.Compute(AsymPred, AsymActual);
        double infVal = informedness.Compute(AsymPred, AsymActual);

        Assert.Equal(balAccVal, (infVal + 1.0) / 2.0, Tolerance);
    }

    [Fact]
    public void BalancedAccuracy_Equals_HalfInformedness_Plus_Half_SymmetricCase()
    {
        var balAcc = new BalancedAccuracyMetric<double>();
        var informedness = new InformednessMetric<double>();

        double balAccVal = balAcc.Compute(SymPred, SymActual);
        double infVal = informedness.Compute(SymPred, SymActual);

        Assert.Equal(balAccVal, (infVal + 1.0) / 2.0, Tolerance);
    }

    #endregion

    #region Edge Cases and Validation

    [Fact]
    public void AllMetrics_MismatchedLengths_Throws()
    {
        double[] short1 = [1, 0, 1];
        double[] long1 = [1, 0, 1, 0];

        Assert.Throws<ArgumentException>(() => new FBetaScoreMetric<double>(1.0).Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new NPVMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new DiagnosticOddsRatioMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new FowlkesMallowsMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new InformednessMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new MarkednessMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new PositiveLikelihoodRatioMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new NegativeLikelihoodRatioMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new PrevalenceThresholdMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new JaccardScoreMetric<double>().Compute(short1, long1));
        Assert.Throws<ArgumentException>(() => new ThreatScoreMetric<double>().Compute(short1, long1));
    }

    [Fact]
    public void AllMetrics_EmptyInput_ReturnsDefined()
    {
        double[] empty = [];

        // Each metric has its own empty-input behavior
        Assert.Equal(0.0, new FBetaScoreMetric<double>(1.0).Compute(empty, empty), Tolerance);
        Assert.Equal(0.0, new NPVMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(1.0, new DiagnosticOddsRatioMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(0.0, new FowlkesMallowsMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(0.0, new InformednessMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(0.0, new MarkednessMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(1.0, new PositiveLikelihoodRatioMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(1.0, new NegativeLikelihoodRatioMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(1.0, new PrevalenceThresholdMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(0.0, new JaccardScoreMetric<double>().Compute(empty, empty), Tolerance);
        Assert.Equal(0.0, new ThreatScoreMetric<double>().Compute(empty, empty), Tolerance);
    }

    [Fact]
    public void FBeta_ComputeWithCI_InvalidBootstrapSamples_Throws()
    {
        var metric = new FBetaScoreMetric<double>(1.0);
        double[] preds = [1, 0, 1];
        double[] actuals = [1, 0, 0];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(preds, actuals, bootstrapSamples: 1));
    }

    [Fact]
    public void FBeta_ComputeWithCI_InvalidConfidenceLevel_Throws()
    {
        var metric = new FBetaScoreMetric<double>(1.0);
        double[] preds = [1, 0, 1];
        double[] actuals = [1, 0, 0];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(preds, actuals, confidenceLevel: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(preds, actuals, confidenceLevel: 1.0));
    }

    [Fact]
    public void FBeta_ComputeWithCI_Reproducible()
    {
        var metric = new FBetaScoreMetric<double>(1.0);
        double[] preds = [1, 0, 1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 0, 0, 1, 1, 1, 0];

        var ci1 = metric.ComputeWithCI(preds, actuals, randomSeed: 42);
        var ci2 = metric.ComputeWithCI(preds, actuals, randomSeed: 42);

        Assert.Equal(ci1.LowerBound, ci2.LowerBound);
        Assert.Equal(ci1.UpperBound, ci2.UpperBound);
    }

    [Fact]
    public void CohensKappa_MultiClass_HandCalculated()
    {
        // 3-class: actuals=[0,0,0,1,1,1,2,2,2], preds=[0,0,1,1,1,2,2,2,0]
        // Confusion matrix (rows=actual, cols=predicted):
        //       0  1  2
        // 0  [  2  1  0 ]
        // 1  [  0  2  1 ]
        // 2  [  1  0  2 ]
        //
        // po = (2+2+2)/9 = 6/9 = 2/3
        // pe = sum( (rowSum_c/n) * (colSum_c/n) ) for each class
        // Row sums: [3, 3, 3], Col sums: [3, 3, 3]
        // pe = (3/9)*(3/9) + (3/9)*(3/9) + (3/9)*(3/9) = 3*(1/3*1/3) = 3*(1/9) = 1/3
        // kappa = (2/3 - 1/3) / (1 - 1/3) = (1/3) / (2/3) = 1/2 = 0.5
        double[] actuals = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        double[] preds = [0, 0, 1, 1, 1, 2, 2, 2, 0];

        var metric = new CohensKappaMetric<double>();
        Assert.Equal(0.5, metric.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void CohensKappa_BinaryAsymmetric_HandCalculated()
    {
        // TP=2, TN=3, FP=2, FN=1 using AsymPred/AsymActual
        // Confusion matrix:
        //       0  1
        // 0  [  3  2 ]  (actual=0)
        // 1  [  1  2 ]  (actual=1)
        //
        // po = (3+2)/8 = 5/8
        // Row sums: [5, 3], Col sums: [4, 4]
        // pe = (5/8)*(4/8) + (3/8)*(4/8) = 20/64 + 12/64 = 32/64 = 0.5
        // kappa = (5/8 - 0.5) / (1 - 0.5) = (1/8) / (1/2) = 1/4 = 0.25
        var metric = new CohensKappaMetric<double>();
        Assert.Equal(0.25, metric.Compute(AsymPred, AsymActual), Tolerance);
    }

    #endregion
}
