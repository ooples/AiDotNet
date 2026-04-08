using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics.Classification;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep math integration tests for advanced classification metrics:
/// MCC, Cohen's Kappa, LogLoss, BrierScore, Jaccard, DOR, Fowlkes-Mallows,
/// Informedness, Markedness, ThreatScore, LR+, LR-, FBeta multi-class averaging.
/// Tests use hand-calculated expected values and cross-metric mathematical identities.
/// </summary>
public class AdvancedClassificationAndProbabilisticMetricsDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ═══════════════════════════════════════════════════════════════
    // Shared test data: TP=3, TN=3, FP=2, FN=2
    // Precision=Recall=Specificity=NPV=0.6, FPR=FNR=0.4, Accuracy=0.6
    // ═══════════════════════════════════════════════════════════════
    private static readonly double[] Pred10 = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0];
    private static readonly double[] Actual10 = [1, 1, 0, 0, 1, 0, 0, 0, 1, 1];

    // ═══════════════════════════════════════════════════════════════
    // MATTHEWS CORRELATION COEFFICIENT
    // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MCC_HandCalculated_TP3TN3FP2FN2()
    {
        // MCC = (3*3 - 2*2) / sqrt(5*5*5*5) = 5/25 = 0.2
        var metric = new MatthewsCorrelationCoefficientMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(0.2, result, Tol);
    }

    [Fact]
    public void MCC_PerfectPrediction_ReturnsOne()
    {
        var metric = new MatthewsCorrelationCoefficientMetric<double>();
        double[] pred = [1, 1, 0, 0, 1];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void MCC_InvertedPrediction_ReturnsNegativeOne()
    {
        // Every positive predicted as negative and vice versa
        var metric = new MatthewsCorrelationCoefficientMetric<double>();
        double[] pred = [0, 0, 1, 1, 0];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(-1.0, result, Tol);
    }

    [Fact]
    public void MCC_AllSameClass_ReturnsZero()
    {
        // Classifier predicts all positive - TP=3, FP=2, TN=0, FN=0
        // denominator has (TN+FP)=2 and (TN+FN)=0, so denominator = 0 => MCC = 0
        var metric = new MatthewsCorrelationCoefficientMetric<double>();
        double[] pred = [1, 1, 1, 1, 1];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MCC_MultiClass_GorodkinFormula()
    {
        // 3-class confusion matrix:
        //          pred=0 pred=1 pred=2
        // actual=0:  2      0      1
        // actual=1:  0      3      0
        // actual=2:  1      0      2
        // n=9, c=7, p=[3,3,3], t=[3,3,3]
        // sumPkTk=27, sumPk2=27, sumTk2=27
        // MCC = (c*s - sumPkTk) / sqrt((s^2-sumPk2)(s^2-sumTk2))
        //     = (7*9-27) / sqrt((81-27)(81-27)) = 36/54 = 2/3
        var metric = new MatthewsCorrelationCoefficientMetric<double>();
        double[] pred = [0, 0, 1, 1, 2, 2, 0, 1, 2];
        double[] actual = [0, 0, 1, 1, 2, 0, 2, 1, 2];
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.0 / 3.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // COHEN'S KAPPA
    // Kappa = (po - pe) / (1 - pe)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void CohensKappa_HandCalculated_TP3TN3FP2FN2()
    {
        // po = 6/10 = 0.6
        // pe = (5/10)*(5/10) + (5/10)*(5/10) = 0.5
        // Kappa = (0.6 - 0.5) / (1 - 0.5) = 0.2
        var metric = new CohensKappaMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(0.2, result, Tol);
    }

    [Fact]
    public void CohensKappa_PerfectAgreement_ReturnsOne()
    {
        var metric = new CohensKappaMetric<double>();
        double[] pred = [1, 0, 1, 0, 1];
        double[] actual = [1, 0, 1, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void CohensKappa_RandomAgreement_ReturnsZero()
    {
        // 50% accuracy on balanced classes = no better than chance
        // pred  = [1,1,0,0]
        // actual= [1,0,1,0]
        // TP=1, TN=1, FP=1, FN=1
        // po = 2/4 = 0.5
        // pe = (2/4)*(2/4) + (2/4)*(2/4) = 0.25 + 0.25 = 0.5
        // Kappa = (0.5 - 0.5) / (1 - 0.5) = 0
        var metric = new CohensKappaMetric<double>();
        double[] pred = [1, 1, 0, 0];
        double[] actual = [1, 0, 1, 0];
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void CohensKappa_MultiClass_ThreeClasses()
    {
        // Same confusion matrix as MCC multi-class test
        var metric = new CohensKappaMetric<double>();
        double[] pred = [0, 0, 1, 1, 2, 2, 0, 1, 2];
        double[] actual = [0, 0, 1, 1, 2, 0, 2, 1, 2];
        // po = 7/9
        // pe = (3/9)*(3/9) + (3/9)*(3/9) + (3/9)*(3/9) = 3 * (1/9) = 1/3
        // Kappa = (7/9 - 1/3) / (1 - 1/3) = (7/9 - 3/9) / (6/9) = (4/9) / (6/9) = 4/6 = 2/3
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.0 / 3.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // LOG LOSS (CROSS-ENTROPY)
    // LogLoss = -1/N * Σ[y*log(p) + (1-y)*log(1-p)]
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void LogLoss_HandCalculated_BinaryClassification()
    {
        // prob  = [0.9, 0.8, 0.3, 0.1]
        // actual= [1,   1,   0,   0  ]
        // LogLoss = -(1/4)*[log(0.9) + log(0.8) + log(0.7) + log(0.9)]
        // = -(1/4)*[-0.10536 + -0.22314 + -0.35667 + -0.10536]
        // = -(1/4)*(-0.79054) = 0.197635
        var metric = new LogLossMetric<double>();
        double[] probs = [0.9, 0.8, 0.3, 0.1];
        double[] actual = [1, 1, 0, 0];
        double result = metric.Compute(probs, actual);
        double expected = -(Math.Log(0.9) + Math.Log(0.8) + Math.Log(0.7) + Math.Log(0.9)) / 4.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void LogLoss_PerfectProbabilities_NearZero()
    {
        // Near-perfect probabilities: 0.999 for correct class
        var metric = new LogLossMetric<double>();
        double[] probs = [0.999, 0.999, 0.001, 0.001];
        double[] actual = [1, 1, 0, 0];
        double result = metric.Compute(probs, actual);
        Assert.True(result < 0.01, $"LogLoss ({result}) should be near zero for perfect probabilities");
    }

    [Fact]
    public void LogLoss_ConfidentlyWrong_VeryHigh()
    {
        // High probability assigned to wrong class
        var metric = new LogLossMetric<double>();
        double[] probs = [0.01, 0.99]; // predicts 0.01 for actual=1, 0.99 for actual=0
        double[] actual = [1, 0];
        double result = metric.Compute(probs, actual);
        // -(log(0.01) + log(0.01))/2 = -(-4.605 + -4.605)/2 = 4.605
        Assert.True(result > 3.0, $"LogLoss ({result}) should be very high for confidently wrong predictions");
    }

    [Fact]
    public void LogLoss_UniformProbabilities_IsLog2()
    {
        // Random guessing with p=0.5 gives LogLoss = -log(0.5) = log(2) ≈ 0.6931
        var metric = new LogLossMetric<double>();
        double[] probs = [0.5, 0.5, 0.5, 0.5];
        double[] actual = [1, 0, 1, 0];
        double result = metric.Compute(probs, actual);
        Assert.Equal(Math.Log(2), result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // BRIER SCORE
    // Brier = (1/N) * Σ(p - y)²
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void BrierScore_HandCalculated()
    {
        // prob  = [0.9, 0.8, 0.3, 0.1]
        // actual= [1,   1,   0,   0  ]
        // Brier = (1/4)*[(0.9-1)^2 + (0.8-1)^2 + (0.3-0)^2 + (0.1-0)^2]
        // = (1/4)*[0.01 + 0.04 + 0.09 + 0.01] = 0.15/4 = 0.0375
        var metric = new BrierScoreMetric<double>();
        double[] probs = [0.9, 0.8, 0.3, 0.1];
        double[] actual = [1, 1, 0, 0];
        double result = metric.Compute(probs, actual);
        Assert.Equal(0.0375, result, Tol);
    }

    [Fact]
    public void BrierScore_RandomGuessing_ReturnsQuarter()
    {
        // With p=0.5 for all: Brier = (0.5-y)^2 = 0.25 for all
        var metric = new BrierScoreMetric<double>();
        double[] probs = [0.5, 0.5, 0.5, 0.5];
        double[] actual = [1, 0, 1, 0];
        double result = metric.Compute(probs, actual);
        Assert.Equal(0.25, result, Tol);
    }

    [Fact]
    public void BrierScore_PerfectProbabilities_ReturnsZero()
    {
        var metric = new BrierScoreMetric<double>();
        double[] probs = [1.0, 0.0, 1.0, 0.0];
        double[] actual = [1, 0, 1, 0];
        double result = metric.Compute(probs, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void BrierScore_WorstCase_ReturnsOne()
    {
        // Predicting 0 for all actual=1 and 1 for all actual=0
        var metric = new BrierScoreMetric<double>();
        double[] probs = [0.0, 1.0, 0.0, 1.0];
        double[] actual = [1, 0, 1, 0];
        double result = metric.Compute(probs, actual);
        Assert.Equal(1.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // JACCARD SCORE
    // Jaccard = TP / (TP + FP + FN) = Intersection / Union
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void JaccardScore_HandCalculated_TP3FP2FN2()
    {
        // Jaccard = 3/(3+2+2) = 3/7
        var metric = new JaccardScoreMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(3.0 / 7.0, result, Tol);
    }

    [Fact]
    public void JaccardScore_PerfectPrediction_ReturnsOne()
    {
        var metric = new JaccardScoreMetric<double>();
        double[] pred = [1, 1, 0, 0, 1];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void JaccardScore_NoOverlap_ReturnsZero()
    {
        // All predictions wrong
        var metric = new JaccardScoreMetric<double>();
        double[] pred = [0, 0, 1, 1];
        double[] actual = [1, 1, 0, 0];
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // THREAT SCORE (CRITICAL SUCCESS INDEX) - identical to Jaccard
    // TS = TP / (TP + FN + FP)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ThreatScore_EqualsJaccard_SameFormula()
    {
        // Both compute TP/(TP+FP+FN)
        var jaccard = new JaccardScoreMetric<double>();
        var threat = new ThreatScoreMetric<double>();
        double jResult = jaccard.Compute(Pred10, Actual10);
        double tResult = threat.Compute(Pred10, Actual10);
        Assert.Equal(jResult, tResult, Tol);
    }

    [Fact]
    public void ThreatScore_HandCalculated()
    {
        var metric = new ThreatScoreMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(3.0 / 7.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // FOWLKES-MALLOWS INDEX
    // FM = sqrt(Precision * Recall)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void FowlkesMallows_HandCalculated()
    {
        // PPV = 3/5 = 0.6, TPR = 3/5 = 0.6
        // FM = sqrt(0.6 * 0.6) = 0.6
        var metric = new FowlkesMallowsMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(0.6, result, Tol);
    }

    [Fact]
    public void FowlkesMallows_GeometricMeanProperty()
    {
        // For asymmetric precision/recall:
        // pred  = [1,1,1,1,1,1,0,0,0,0]
        // actual= [1,1,1,0,0,0,0,0,0,1]
        // TP=3, FP=3, FN=1, TN=3
        // Precision = 3/6 = 0.5, Recall = 3/4 = 0.75
        // FM = sqrt(0.5 * 0.75) = sqrt(0.375) ≈ 0.61237
        var metric = new FowlkesMallowsMetric<double>();
        double[] pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        double[] actual = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(Math.Sqrt(0.5 * 0.75), result, Tol);
    }

    [Fact]
    public void FowlkesMallows_BetweenF1AndArithmeticMean()
    {
        // AM-GM-HM inequality: HM <= GM <= AM
        // F1 = harmonic mean(P, R), FM = geometric mean(P, R)
        // So F1 <= FM <= (P + R) / 2
        double[] pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        double[] actual = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1];
        // P=0.5, R=0.75
        double precision = 0.5;
        double recall = 0.75;
        double f1 = 2 * precision * recall / (precision + recall); // harmonic mean
        double fm = Math.Sqrt(precision * recall); // geometric mean
        double am = (precision + recall) / 2; // arithmetic mean

        var metric = new FowlkesMallowsMetric<double>();
        double fmResult = metric.Compute(pred, actual);

        Assert.True(f1 <= fmResult + Tol, $"F1 ({f1}) should be <= FM ({fmResult})");
        Assert.True(fmResult <= am + Tol, $"FM ({fmResult}) should be <= AM ({am})");
    }

    // ═══════════════════════════════════════════════════════════════
    // INFORMEDNESS (YOUDEN'S J)
    // Informedness = TPR + TNR - 1
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Informedness_HandCalculated()
    {
        // TPR = 0.6, TNR = 0.6
        // Informedness = 0.6 + 0.6 - 1 = 0.2
        var metric = new InformednessMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(0.2, result, Tol);
    }

    [Fact]
    public void Informedness_PerfectClassifier_ReturnsOne()
    {
        var metric = new InformednessMetric<double>();
        double[] pred = [1, 1, 0, 0, 1];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void Informedness_RandomClassifier_ReturnsZero()
    {
        // TP=1, TN=1, FP=1, FN=1 => TPR=0.5, TNR=0.5 => J=0
        var metric = new InformednessMetric<double>();
        double[] pred = [1, 1, 0, 0];
        double[] actual = [1, 0, 1, 0];
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MARKEDNESS
    // Markedness = PPV + NPV - 1
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Markedness_HandCalculated()
    {
        // PPV = 3/5 = 0.6, NPV = 3/5 = 0.6
        // Markedness = 0.6 + 0.6 - 1 = 0.2
        var metric = new MarkednessMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(0.2, result, Tol);
    }

    [Fact]
    public void Markedness_PerfectClassifier_ReturnsOne()
    {
        var metric = new MarkednessMetric<double>();
        double[] pred = [1, 1, 0, 0, 1];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // DIAGNOSTIC ODDS RATIO
    // DOR = (TP*TN)/(FP*FN) with Haldane-Anscombe correction (+0.5)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void DOR_HandCalculated_WithCorrection()
    {
        // TP=3, TN=3, FP=2, FN=2
        // DOR = (3.5*3.5)/(2.5*2.5) = 12.25/6.25 = 1.96
        var metric = new DiagnosticOddsRatioMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(1.96, result, Tol);
    }

    [Fact]
    public void DOR_PerfectClassifier_VeryHigh()
    {
        // TP=3, TN=2, FP=0, FN=0
        // DOR = (3.5*2.5)/(0.5*0.5) = 8.75/0.25 = 35
        var metric = new DiagnosticOddsRatioMetric<double>();
        double[] pred = [1, 1, 0, 0, 1];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(35.0, result, Tol);
    }

    [Fact]
    public void DOR_WorstClassifier_VeryLow()
    {
        // All predictions inverted: TP=0, TN=0, FP=2, FN=3
        // DOR = (0.5*0.5)/(2.5*3.5) = 0.25/8.75 ≈ 0.02857
        var metric = new DiagnosticOddsRatioMetric<double>();
        double[] pred = [0, 0, 1, 1, 0];
        double[] actual = [1, 1, 0, 0, 1];
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.25 / 8.75, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // POSITIVE LIKELIHOOD RATIO (LR+)
    // LR+ = TPR / FPR = Sensitivity / (1 - Specificity)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void LRPositive_HandCalculated()
    {
        // TPR = 0.6, FPR = 0.4
        // LR+ = 0.6/0.4 = 1.5
        var metric = new PositiveLikelihoodRatioMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(1.5, result, Tol);
    }

    [Fact]
    public void LRPositive_PerfectSpecificity_ReturnsMaxValue()
    {
        // No false positives => FPR = 0 => LR+ = infinity
        var metric = new PositiveLikelihoodRatioMetric<double>();
        double[] pred = [1, 1, 0, 0, 0];
        double[] actual = [1, 1, 0, 0, 0];
        double result = metric.Compute(pred, actual);
        Assert.Equal(double.MaxValue, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // NEGATIVE LIKELIHOOD RATIO (LR-)
    // LR- = FNR / TNR = (1 - Sensitivity) / Specificity
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void LRNegative_HandCalculated()
    {
        // FNR = 0.4, TNR = 0.6
        // LR- = 0.4/0.6 = 2/3
        var metric = new NegativeLikelihoodRatioMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.Equal(2.0 / 3.0, result, Tol);
    }

    [Fact]
    public void LRNegative_PerfectSensitivity_ReturnsZero()
    {
        // No false negatives => FNR = 0 => LR- = 0
        var metric = new NegativeLikelihoodRatioMetric<double>();
        double[] pred = [1, 1, 1, 0, 0];
        double[] actual = [1, 1, 1, 0, 0];
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // FBETA SCORE - MULTI-CLASS AVERAGING
    // F_beta = (1+β²)(P*R)/(β²P+R)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void FBeta_F1_MacroAverage_ThreeClasses()
    {
        // 3-class: pred=[0,0,1,1,2,2], actual=[0,1,1,2,2,0]
        // Class 0: TP=1, FP=1, FN=1 => P=1/2, R=1/2, F1=1/2
        // Class 1: TP=1, FP=1, FN=1 => P=1/2, R=1/2, F1=1/2
        // Class 2: TP=1, FP=1, FN=1 => P=1/2, R=1/2, F1=1/2
        // Macro F1 = (1/2 + 1/2 + 1/2) / 3 = 1/2
        var metric = new FBetaScoreMetric<double>(1.0, averaging: AveragingMethod.Macro);
        double[] pred = [0, 0, 1, 1, 2, 2];
        double[] actual = [0, 1, 1, 2, 2, 0];
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.5, result, Tol);
    }

    [Fact]
    public void FBeta_F2_WeightsRecallMore()
    {
        // With β=2, recall is weighted 4x more than precision
        // pred=[1,1,1,1,1,0,0,0], actual=[1,1,1,0,0,0,0,1]
        // TP=3, FP=2, FN=1
        // Precision = 3/5 = 0.6, Recall = 3/4 = 0.75
        // F2 = (1+4)(0.6*0.75)/(4*0.6+0.75) = 5*0.45/(2.4+0.75) = 2.25/3.15 ≈ 0.71429
        var f2 = new FBetaScoreMetric<double>(2.0);
        double[] pred = [1, 1, 1, 1, 1, 0, 0, 0];
        double[] actual = [1, 1, 1, 0, 0, 0, 0, 1];
        double result = f2.Compute(pred, actual);
        Assert.Equal(2.25 / 3.15, result, Tol);
    }

    [Fact]
    public void FBeta_F05_WeightsPrecisionMore()
    {
        // With β=0.5, precision is weighted 4x more than recall
        // Same data as above: P=0.6, R=0.75
        // F0.5 = (1+0.25)(0.6*0.75)/(0.25*0.6+0.75) = 1.25*0.45/(0.15+0.75) = 0.5625/0.9 = 0.625
        var f05 = new FBetaScoreMetric<double>(0.5);
        double[] pred = [1, 1, 1, 1, 1, 0, 0, 0];
        double[] actual = [1, 1, 1, 0, 0, 0, 0, 1];
        double result = f05.Compute(pred, actual);
        Assert.Equal(0.625, result, Tol);
    }

    [Fact]
    public void FBeta_HigherBeta_CloserToRecall()
    {
        // As β increases, FBeta approaches Recall
        double[] pred = [1, 1, 1, 1, 1, 0, 0, 0];
        double[] actual = [1, 1, 1, 0, 0, 0, 0, 1];
        // P=0.6, R=0.75
        var f1 = new FBetaScoreMetric<double>(1.0);
        var f2 = new FBetaScoreMetric<double>(2.0);
        var f5 = new FBetaScoreMetric<double>(5.0);

        double r1 = f1.Compute(pred, actual);
        double r2 = f2.Compute(pred, actual);
        double r5 = f5.Compute(pred, actual);

        // F1 < F2 < F5 < R=0.75 (when R > P)
        Assert.True(r1 < r2, $"F1 ({r1}) should be < F2 ({r2}) when R > P");
        Assert.True(r2 < r5, $"F2 ({r2}) should be < F5 ({r5}) when R > P");
        Assert.True(r5 < 0.75 + Tol, $"F5 ({r5}) should approach Recall (0.75)");
    }

    // ═══════════════════════════════════════════════════════════════
    // CROSS-METRIC MATHEMATICAL IDENTITIES
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Identity_MCC_Squared_Equals_Informedness_Times_Markedness()
    {
        // Fundamental identity: MCC² = Informedness × Markedness
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        var inform = new InformednessMetric<double>();
        var marked = new MarkednessMetric<double>();

        double mccVal = mcc.Compute(Pred10, Actual10);
        double informVal = inform.Compute(Pred10, Actual10);
        double markedVal = marked.Compute(Pred10, Actual10);

        Assert.Equal(mccVal * mccVal, informVal * markedVal, Tol);
    }

    [Fact]
    public void Identity_MCC_Squared_Equals_Informedness_Times_Markedness_Asymmetric()
    {
        // Test with asymmetric confusion matrix (P ≠ R)
        double[] pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        double[] actual = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1];
        // TP=3, FP=3, FN=1, TN=3
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        var inform = new InformednessMetric<double>();
        var marked = new MarkednessMetric<double>();

        double mccVal = mcc.Compute(pred, actual);
        double informVal = inform.Compute(pred, actual);
        double markedVal = marked.Compute(pred, actual);

        Assert.Equal(mccVal * mccVal, informVal * markedVal, 1e-4);
    }

    [Fact]
    public void Identity_Jaccard_F1_Relationship()
    {
        // Jaccard = F1 / (2 - F1)
        var jaccard = new JaccardScoreMetric<double>();
        var f1 = new FBetaScoreMetric<double>(1.0);

        double jVal = jaccard.Compute(Pred10, Actual10);
        double f1Val = f1.Compute(Pred10, Actual10);
        double expected = f1Val / (2 - f1Val);

        Assert.Equal(expected, jVal, Tol);
    }

    [Fact]
    public void Identity_Jaccard_F1_Relationship_Asymmetric()
    {
        // Test with more asymmetric data
        double[] pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        double[] actual = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1];

        var jaccard = new JaccardScoreMetric<double>();
        var f1 = new FBetaScoreMetric<double>(1.0);

        double jVal = jaccard.Compute(pred, actual);
        double f1Val = f1.Compute(pred, actual);
        double expected = f1Val / (2 - f1Val);

        Assert.Equal(expected, jVal, Tol);
    }

    [Fact]
    public void Identity_DOR_Equals_LRPositive_Over_LRNegative_WithoutCorrection()
    {
        // DOR (without Haldane-Anscombe) = LR+ / LR-
        // With correction applied, DOR ≈ LR+/LR- but not exactly
        var lrPlus = new PositiveLikelihoodRatioMetric<double>();
        var lrMinus = new NegativeLikelihoodRatioMetric<double>();

        double lrPlusVal = lrPlus.Compute(Pred10, Actual10);
        double lrMinusVal = lrMinus.Compute(Pred10, Actual10);

        // LR+/LR- = 1.5 / (2/3) = 2.25 = TP*TN/(FP*FN) = 9/4 (uncorrected DOR)
        Assert.Equal(2.25, lrPlusVal / lrMinusVal, Tol);
    }

    [Fact]
    public void Identity_Brier_Bounded_By_LogLoss()
    {
        // For binary classification with well-calibrated probabilities:
        // BrierScore <= LogLoss (since Brier uses squared error, LogLoss uses log error)
        // This holds because -log(x) >= 1-x >= (1-x)^2 for x in [0,1]
        // But this is not universally true - it depends on specific probabilities
        // Instead test: for p=0.5, Brier=0.25, LogLoss=log(2)≈0.693
        var brier = new BrierScoreMetric<double>();
        var logLoss = new LogLossMetric<double>();
        double[] probs = [0.5, 0.5, 0.5, 0.5];
        double[] actual = [1, 0, 1, 0];

        double brierVal = brier.Compute(probs, actual);
        double logLossVal = logLoss.Compute(probs, actual);

        Assert.True(brierVal < logLossVal,
            $"BrierScore ({brierVal}) should be < LogLoss ({logLossVal}) for uniform probabilities");
    }

    [Fact]
    public void Identity_Informedness_Equals_1_Minus_FNR_Minus_FPR()
    {
        // Informedness = TPR + TNR - 1 = (1 - FNR) + (1 - FPR) - 1 = 1 - FNR - FPR
        // This equals 1 - 2*BER where BER = (FNR+FPR)/2
        var inform = new InformednessMetric<double>();
        var ber = new BalancedErrorRateMetric<double>();

        double informVal = inform.Compute(Pred10, Actual10);
        double berVal = ber.Compute(Pred10, Actual10);

        Assert.Equal(1.0 - 2.0 * berVal, informVal, Tol);
    }

    [Fact]
    public void Identity_FowlkesMallows_Squared_Equals_Precision_Times_Recall()
    {
        // FM = sqrt(P * R), so FM^2 = P * R
        var fm = new FowlkesMallowsMetric<double>();
        double fmVal = fm.Compute(Pred10, Actual10);
        // P = 0.6, R = 0.6
        Assert.Equal(0.6 * 0.6, fmVal * fmVal, Tol);
    }

    [Fact]
    public void Identity_Kappa_LessThanOrEqual_MCC_ForBalancedClasses()
    {
        // For balanced marginals, Kappa <= |MCC|
        // Our data has balanced marginals (5 pred positive, 5 pred negative, 5 actual positive, 5 actual negative)
        var kappa = new CohensKappaMetric<double>();
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();

        double kappaVal = kappa.Compute(Pred10, Actual10);
        double mccVal = mcc.Compute(Pred10, Actual10);

        // For balanced binary: Kappa = MCC (they're equal when marginals are uniform)
        Assert.Equal(mccVal, kappaVal, Tol);
    }

    [Fact]
    public void Identity_Jaccard_Reciprocal_Equals_PrecisionRecall_HarmonicReciprocal()
    {
        // 1/Jaccard = 1/TP * (TP + FP + FN) = (TP+FP)/TP + FN/TP = 1/P + 1/R - 1
        var jaccard = new JaccardScoreMetric<double>();
        double jVal = jaccard.Compute(Pred10, Actual10);
        // P = 0.6, R = 0.6
        double expected = 1.0 / 0.6 + 1.0 / 0.6 - 1.0; // ≈ 2.333
        Assert.Equal(expected, 1.0 / jVal, Tol);
    }

    [Fact]
    public void Identity_F1_GeometricMean_HarmonicMean_Ordering()
    {
        // For any P, R > 0: F1 <= FM <= (P+R)/2
        // HM <= GM <= AM inequality
        double[] pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0];
        double[] actual = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1];
        // TP=2, FP=2, FN=2, TN=4
        // P = 2/4 = 0.5, R = 2/4 = 0.5

        var f1Metric = new FBetaScoreMetric<double>(1.0);
        var fmMetric = new FowlkesMallowsMetric<double>();

        double f1Val = f1Metric.Compute(pred, actual);
        double fmVal = fmMetric.Compute(pred, actual);
        double amVal = (0.5 + 0.5) / 2.0;

        // When P = R, all three means are equal
        Assert.Equal(f1Val, fmVal, Tol);
        Assert.Equal(fmVal, amVal, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // BOUNDARY/DEGENERATE CASES
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void BrierScore_MultiClass_ThreeClasses()
    {
        // Multi-class Brier Score: (1/N*K) * ΣΣ (p_ik - y_ik)²
        // 2 samples, 3 classes
        // Sample 0: actual=0, probs=[0.7, 0.2, 0.1] → errors: (0.7-1)² + (0.2-0)² + (0.1-0)² = 0.09+0.04+0.01 = 0.14
        // Sample 1: actual=2, probs=[0.1, 0.2, 0.7] → errors: (0.1-0)² + (0.2-0)² + (0.7-1)² = 0.01+0.04+0.09 = 0.14
        // Brier = (0.14 + 0.14) / (2*3) = 0.28/6 ≈ 0.04667
        var metric = new BrierScoreMetric<double>();
        double[] probs = [0.7, 0.2, 0.1, 0.1, 0.2, 0.7]; // flattened
        double[] actual = [0, 2];
        double result = metric.Compute(probs, actual, numClasses: 3);
        Assert.Equal(0.28 / 6.0, result, Tol);
    }

    [Fact]
    public void LogLoss_ClipsExtremeProbabilities()
    {
        // LogLoss should clip probabilities to avoid log(0)
        var metric = new LogLossMetric<double>();
        double[] probs = [0.0, 1.0]; // Would cause log(0) without clipping
        double[] actual = [0, 1];
        // With epsilon=1e-15: log(1-1e-15) ≈ -1e-15 for both
        double result = metric.Compute(probs, actual);
        Assert.True((!double.IsNaN(result) && !double.IsInfinity(result)), $"LogLoss ({result}) should be finite with extreme probabilities");
        Assert.True(result >= 0, $"LogLoss ({result}) should be non-negative");
    }

    [Fact]
    public void MCC_RangeIsMinusOneToOne()
    {
        var metric = new MatthewsCorrelationCoefficientMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.True(result >= -1.0 - Tol && result <= 1.0 + Tol,
            $"MCC ({result}) should be in [-1, 1]");
    }

    [Fact]
    public void CohensKappa_RangeCheck()
    {
        var metric = new CohensKappaMetric<double>();
        double result = metric.Compute(Pred10, Actual10);
        Assert.True(result >= -1.0 - Tol && result <= 1.0 + Tol,
            $"Kappa ({result}) should be in [-1, 1]");
    }

    [Fact]
    public void FBeta_InvalidBeta_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(-1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FBetaScoreMetric<double>(double.PositiveInfinity));
    }
}
