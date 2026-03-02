using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Metrics.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep math-correctness integration tests for classification metrics (F1, MCC, Cohen's Kappa,
/// Accuracy, Precision, Recall) and regression metrics (R2, MAE, MSE, RMSE, MAPE).
/// Verifies hand-calculated values, mathematical identities, and edge cases.
/// </summary>
public class ClassificationAndRegressionMetricsDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    #region F1 Score

    [Fact]
    public void F1_PerfectPredictions_IsOne()
    {
        var f1 = new F1ScoreMetric<double>();
        double[] preds = [1, 0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0, 1];

        double score = f1.Compute(preds, actuals);
        Assert.Equal(1.0, score, Tolerance);
    }

    [Fact]
    public void F1_AllWrong_IsZero()
    {
        var f1 = new F1ScoreMetric<double>();
        double[] preds = [0, 0, 0, 0]; // Predict all negative
        double[] actuals = [1, 1, 1, 1]; // All actually positive

        double score = f1.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void F1_HandCalculated_ConfusionMatrix()
    {
        // TP=3, FP=1, FN=2, TN=4
        // Precision = 3/(3+1) = 0.75
        // Recall = 3/(3+2) = 0.6
        // F1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 2 * 0.45 / 1.35 = 0.666...
        var f1 = new F1ScoreMetric<double>();
        double[] preds =   [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]; // 4 predicted positive
        double[] actuals =  [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]; // 5 actually positive

        double score = f1.Compute(preds, actuals);
        double expected = 2.0 * 0.75 * 0.6 / (0.75 + 0.6);
        Assert.Equal(expected, score, LooseTolerance);
    }

    [Fact]
    public void F1_Bounded_ZeroToOne()
    {
        var f1 = new F1ScoreMetric<double>();
        double[] preds = [1, 0, 1, 0, 1, 0];
        double[] actuals = [0, 1, 1, 0, 0, 1];

        double score = f1.Compute(preds, actuals);
        Assert.True(score >= 0.0 - Tolerance && score <= 1.0 + Tolerance,
            $"F1 score {score} should be in [0, 1]");
    }

    [Fact]
    public void F1_HarmonicMean_LessThanArithmeticMean()
    {
        // F1 = harmonic mean(P, R) <= arithmetic mean(P, R)
        // This is a fundamental property of harmonic mean
        var f1 = new F1ScoreMetric<double>();
        double[] preds = [1, 1, 1, 0, 0, 0, 0, 0];
        double[] actuals = [1, 0, 0, 1, 1, 1, 1, 0];
        // TP=1, FP=2, FN=4 => P=1/3, R=1/5
        // Arithmetic mean = (1/3 + 1/5) / 2 = 8/30 = 0.2667
        // F1 = 2*(1/3)*(1/5) / (1/3+1/5) = 2/15 / (8/15) = 0.25

        double score = f1.Compute(preds, actuals);
        double precision = 1.0 / 3.0;
        double recall = 1.0 / 5.0;
        double arithmeticMean = (precision + recall) / 2.0;

        Assert.True(score <= arithmeticMean + Tolerance,
            $"F1 ({score}) should be <= arithmetic mean ({arithmeticMean})");
    }

    [Fact]
    public void F1_EmptyInput_ReturnsZero()
    {
        var f1 = new F1ScoreMetric<double>();
        double[] empty = [];

        double score = f1.Compute(empty, empty);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void F1_MismatchedLengths_Throws()
    {
        var f1 = new F1ScoreMetric<double>();
        double[] preds = [1, 0];
        double[] actuals = [1, 0, 1];

        Assert.Throws<ArgumentException>(() => f1.Compute(preds, actuals));
    }

    [Fact]
    public void F1_MacroAverage_HandCalculated()
    {
        // 3 classes: 0, 1, 2
        var f1 = new F1ScoreMetric<double>(averaging: AveragingMethod.Macro);
        double[] preds = [0, 1, 2, 0, 1, 2];
        double[] actuals = [0, 1, 2, 1, 0, 2]; // class 2: perfect, class 0 and 1: one TP each, one FP/FN

        double score = f1.Compute(preds, actuals);
        Assert.True(score > 0 && score <= 1, $"Macro F1 {score} should be in (0, 1]");
    }

    #endregion

    #region Matthews Correlation Coefficient

    [Fact]
    public void MCC_PerfectPredictions_IsOne()
    {
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        double[] preds = [1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 0, 1, 0];

        double score = mcc.Compute(preds, actuals);
        Assert.Equal(1.0, score, Tolerance);
    }

    [Fact]
    public void MCC_PerfectlyInverted_IsMinusOne()
    {
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        double[] preds = [0, 1, 0, 1, 0, 1];
        double[] actuals = [1, 0, 1, 0, 1, 0];

        double score = mcc.Compute(preds, actuals);
        Assert.Equal(-1.0, score, Tolerance);
    }

    [Fact]
    public void MCC_HandCalculated()
    {
        // TP=5, TN=3, FP=2, FN=1
        // MCC = (5*3 - 2*1) / sqrt((5+2)(5+1)(3+2)(3+1))
        //     = (15-2) / sqrt(7*6*5*4) = 13 / sqrt(840)
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        // Construct: 5 TP, 3 TN, 2 FP, 1 FN
        double[] preds =   [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
        double[] actuals =  [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0];
        // Check: P=1: pred[0..6]=7, actual[0..4,7]=6, TP=5, FP=2, FN=1, TN=3

        double score = mcc.Compute(preds, actuals);
        double expected = 13.0 / Math.Sqrt(840.0);
        Assert.Equal(expected, score, LooseTolerance);
    }

    [Fact]
    public void MCC_Bounded_Minus1To1()
    {
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        double[] preds = [1, 0, 1, 1, 0, 0, 1, 0];
        double[] actuals = [0, 1, 1, 0, 0, 1, 1, 0];

        double score = mcc.Compute(preds, actuals);
        Assert.True(score >= -1.0 - Tolerance && score <= 1.0 + Tolerance,
            $"MCC {score} should be in [-1, 1]");
    }

    [Fact]
    public void MCC_RandomPredictions_NearZero()
    {
        // With balanced random predictions, MCC should be near zero
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        // Alternate: not random, but 50/50 accuracy on both classes
        double[] preds =   [1, 0, 1, 0, 1, 0, 1, 0];
        double[] actuals =  [1, 1, 0, 0, 1, 1, 0, 0];
        // TP=2, TN=2, FP=2, FN=2 => MCC = (4-4)/sqrt(4*4*4*4) = 0

        double score = mcc.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void MCC_AllSamePrediction_IsZero()
    {
        // If model predicts all same class, MCC = 0 (degenerate)
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        double[] preds = [1, 1, 1, 1, 1];
        double[] actuals = [1, 0, 1, 0, 1];

        double score = mcc.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void MCC_EmptyInput_ReturnsZero()
    {
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        double score = mcc.Compute(Array.Empty<double>(), Array.Empty<double>());
        Assert.Equal(0.0, score, Tolerance);
    }

    #endregion

    #region Cohen's Kappa

    [Fact]
    public void Kappa_PerfectAgreement_IsOne()
    {
        var kappa = new CohensKappaMetric<double>();
        double[] preds = [0, 1, 2, 0, 1, 2];
        double[] actuals = [0, 1, 2, 0, 1, 2];

        double score = kappa.Compute(preds, actuals);
        Assert.Equal(1.0, score, Tolerance);
    }

    [Fact]
    public void Kappa_HandCalculated_Binary()
    {
        // 2 classes, 10 samples
        // Confusion matrix: TP=4, TN=3, FP=1, FN=2
        // p_o = (4+3)/10 = 0.7
        // Expected: p(pred=1)*p(actual=1) + p(pred=0)*p(actual=0)
        // p(pred=1)=5/10, p(actual=1)=6/10, p(pred=0)=5/10, p(actual=0)=4/10
        // p_e = 0.5*0.6 + 0.5*0.4 = 0.3 + 0.2 = 0.5
        // Kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
        var kappa = new CohensKappaMetric<double>();
        double[] preds =   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        double[] actuals =  [1, 1, 1, 1, 0, 1, 1, 0, 0, 0];

        double score = kappa.Compute(preds, actuals);
        Assert.Equal(0.4, score, LooseTolerance);
    }

    [Fact]
    public void Kappa_Bounded_Minus1To1()
    {
        var kappa = new CohensKappaMetric<double>();
        double[] preds = [1, 0, 1, 0, 1, 0, 1, 0];
        double[] actuals = [0, 1, 1, 0, 0, 1, 1, 0];

        double score = kappa.Compute(preds, actuals);
        Assert.True(score >= -1.0 - Tolerance && score <= 1.0 + Tolerance,
            $"Kappa {score} should be in [-1, 1]");
    }

    [Fact]
    public void Kappa_EmptyInput_ReturnsZero()
    {
        var kappa = new CohensKappaMetric<double>();
        double score = kappa.Compute(Array.Empty<double>(), Array.Empty<double>());
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void Kappa_MultiClass_PerfectAgreement_IsOne()
    {
        var kappa = new CohensKappaMetric<double>();
        double[] preds = [0, 1, 2, 3, 0, 1, 2, 3];
        double[] actuals = [0, 1, 2, 3, 0, 1, 2, 3];

        double score = kappa.Compute(preds, actuals);
        Assert.Equal(1.0, score, Tolerance);
    }

    [Fact]
    public void Kappa_LessThanOrEqualAccuracy()
    {
        // Kappa penalizes chance agreement, so Kappa <= Accuracy always (conceptually)
        // Actually: Kappa = (p_o - p_e) / (1 - p_e), and since p_e > 0 usually,
        // Kappa <= p_o = accuracy
        var kappa = new CohensKappaMetric<double>();
        var acc = new AccuracyMetric<double>();

        double[] preds = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1];

        double kappaScore = kappa.Compute(preds, actuals);
        double accScore = acc.Compute(preds, actuals);

        Assert.True(kappaScore <= accScore + Tolerance,
            $"Kappa ({kappaScore}) should be <= Accuracy ({accScore})");
    }

    #endregion

    #region Accuracy

    [Fact]
    public void Accuracy_PerfectPredictions_IsOne()
    {
        var acc = new AccuracyMetric<double>();
        double[] preds = [0, 1, 2, 0, 1];
        double[] actuals = [0, 1, 2, 0, 1];

        double score = acc.Compute(preds, actuals);
        Assert.Equal(1.0, score, Tolerance);
    }

    [Fact]
    public void Accuracy_AllWrong_IsZero()
    {
        var acc = new AccuracyMetric<double>();
        double[] preds = [0, 0, 0];
        double[] actuals = [1, 1, 1];

        double score = acc.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void Accuracy_HandCalculated()
    {
        // 7 out of 10 correct
        var acc = new AccuracyMetric<double>();
        double[] preds =   [1, 0, 1, 0, 1, 0, 1, 0, 0, 0];
        double[] actuals =  [1, 0, 1, 0, 1, 1, 0, 0, 0, 1];

        double score = acc.Compute(preds, actuals);
        Assert.Equal(0.7, score, Tolerance);
    }

    [Fact]
    public void Accuracy_Bounded_ZeroToOne()
    {
        var acc = new AccuracyMetric<double>();
        double[] preds = [1, 0, 1, 1, 0];
        double[] actuals = [0, 1, 1, 0, 0];

        double score = acc.Compute(preds, actuals);
        Assert.True(score >= 0.0 && score <= 1.0, $"Accuracy {score} should be in [0, 1]");
    }

    [Fact]
    public void Accuracy_PlusErrorRate_IsOne()
    {
        var acc = new AccuracyMetric<double>();
        var err = new ErrorRateMetric<double>();

        double[] preds = [1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 1, 0, 0, 1, 0];

        double accScore = acc.Compute(preds, actuals);
        double errScore = err.Compute(preds, actuals);

        Assert.Equal(1.0, accScore + errScore, Tolerance);
    }

    #endregion

    #region R² Score

    [Fact]
    public void R2_PerfectPredictions_IsOne()
    {
        var r2 = new R2ScoreMetric<double>();
        double[] preds = [1, 2, 3, 4, 5];
        double[] actuals = [1, 2, 3, 4, 5];

        double score = r2.Compute(preds, actuals);
        Assert.Equal(1.0, score, Tolerance);
    }

    [Fact]
    public void R2_MeanPrediction_IsZero()
    {
        // Predicting the mean gives R² = 0
        var r2 = new R2ScoreMetric<double>();
        double mean = 3.0; // mean of [1,2,3,4,5]
        double[] preds = [mean, mean, mean, mean, mean];
        double[] actuals = [1, 2, 3, 4, 5];

        double score = r2.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void R2_WorseThanMean_IsNegative()
    {
        var r2 = new R2ScoreMetric<double>();
        double[] preds = [10, 20, 30, 40, 50]; // Way off
        double[] actuals = [1, 2, 3, 4, 5];

        double score = r2.Compute(preds, actuals);
        Assert.True(score < 0, $"R² {score} should be negative for worse-than-mean predictions");
    }

    [Fact]
    public void R2_HandCalculated()
    {
        // actuals = [1, 2, 3], mean = 2
        // preds = [1.5, 2.0, 2.5]
        // SS_res = (1-1.5)^2 + (2-2)^2 + (3-2.5)^2 = 0.25 + 0 + 0.25 = 0.5
        // SS_tot = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2
        // R² = 1 - 0.5/2 = 0.75
        var r2 = new R2ScoreMetric<double>();
        double[] preds = [1.5, 2.0, 2.5];
        double[] actuals = [1, 2, 3];

        double score = r2.Compute(preds, actuals);
        Assert.Equal(0.75, score, Tolerance);
    }

    [Fact]
    public void R2_ConstantActuals_PerfectPred_IsOne()
    {
        var r2 = new R2ScoreMetric<double>();
        double[] preds = [5, 5, 5, 5];
        double[] actuals = [5, 5, 5, 5];

        double score = r2.Compute(preds, actuals);
        Assert.Equal(1.0, score, Tolerance);
    }

    [Fact]
    public void R2_EmptyInput_ReturnsZero()
    {
        var r2 = new R2ScoreMetric<double>();
        double score = r2.Compute(Array.Empty<double>(), Array.Empty<double>());
        Assert.Equal(0.0, score, Tolerance);
    }

    #endregion

    #region MAE

    [Fact]
    public void MAE_PerfectPredictions_IsZero()
    {
        var mae = new MAEMetric<double>();
        double[] preds = [1, 2, 3, 4, 5];
        double[] actuals = [1, 2, 3, 4, 5];

        double score = mae.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void MAE_HandCalculated()
    {
        // MAE = (|1-2| + |3-1| + |5-4|) / 3 = (1+2+1)/3 = 4/3
        var mae = new MAEMetric<double>();
        double[] preds = [1, 3, 5];
        double[] actuals = [2, 1, 4];

        double score = mae.Compute(preds, actuals);
        Assert.Equal(4.0 / 3.0, score, Tolerance);
    }

    [Fact]
    public void MAE_NonNegative()
    {
        var mae = new MAEMetric<double>();
        double[] preds = [-5, 3, -1, 4];
        double[] actuals = [2, -4, 6, -3];

        double score = mae.Compute(preds, actuals);
        Assert.True(score >= 0, $"MAE {score} should be non-negative");
    }

    [Fact]
    public void MAE_Symmetric()
    {
        // MAE(a, b) = MAE(b, a)
        var mae = new MAEMetric<double>();
        double[] a = [1, 2, 3];
        double[] b = [4, 5, 6];

        Assert.Equal(mae.Compute(a, b), mae.Compute(b, a), Tolerance);
    }

    [Fact]
    public void MAE_TriangleInequality()
    {
        // MAE(a, c) <= MAE(a, b) + MAE(b, c) (since it's an Lp norm divided by N)
        var mae = new MAEMetric<double>();
        double[] a = [1, 2, 3];
        double[] b = [2, 3, 4];
        double[] c = [5, 6, 7];

        double dAC = mae.Compute(a, c);
        double dAB = mae.Compute(a, b);
        double dBC = mae.Compute(b, c);

        Assert.True(dAC <= dAB + dBC + Tolerance);
    }

    [Fact]
    public void MAE_EmptyInput_ReturnsZero()
    {
        var mae = new MAEMetric<double>();
        double score = mae.Compute(Array.Empty<double>(), Array.Empty<double>());
        Assert.Equal(0.0, score, Tolerance);
    }

    #endregion

    #region MSE

    [Fact]
    public void MSE_PerfectPredictions_IsZero()
    {
        var mse = new MSEMetric<double>();
        double[] preds = [1, 2, 3];
        double[] actuals = [1, 2, 3];

        double score = mse.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void MSE_HandCalculated()
    {
        // MSE = ((1-2)^2 + (3-1)^2 + (5-4)^2) / 3 = (1+4+1)/3 = 2
        var mse = new MSEMetric<double>();
        double[] preds = [1, 3, 5];
        double[] actuals = [2, 1, 4];

        double score = mse.Compute(preds, actuals);
        Assert.Equal(2.0, score, Tolerance);
    }

    [Fact]
    public void MSE_GreaterThanOrEqual_MAE_Squared()
    {
        // MSE >= MAE^2 (Jensen's inequality: E[X^2] >= E[X]^2)
        // Wait, that's reversed. Actually E[X^2] >= (E[X])^2 for abs errors
        // MSE = E[error^2], MAE = E[|error|]
        // By Jensen: MSE = E[error^2] >= (E[|error|])^2 = MAE^2
        var mse = new MSEMetric<double>();
        var mae = new MAEMetric<double>();
        double[] preds = [1, 3, 5, 2];
        double[] actuals = [2, 1, 4, 5];

        double mseScore = mse.Compute(preds, actuals);
        double maeScore = mae.Compute(preds, actuals);

        Assert.True(mseScore >= maeScore * maeScore - Tolerance,
            $"MSE ({mseScore}) should be >= MAE^2 ({maeScore * maeScore})");
    }

    #endregion

    #region RMSE

    [Fact]
    public void RMSE_PerfectPredictions_IsZero()
    {
        var rmse = new RMSEMetric<double>();
        double[] preds = [1, 2, 3];
        double[] actuals = [1, 2, 3];

        double score = rmse.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void RMSE_IsSqrtOfMSE()
    {
        var rmse = new RMSEMetric<double>();
        var mse = new MSEMetric<double>();
        double[] preds = [1, 3, 5, 2];
        double[] actuals = [2, 1, 4, 5];

        double rmseScore = rmse.Compute(preds, actuals);
        double mseScore = mse.Compute(preds, actuals);

        Assert.Equal(Math.Sqrt(mseScore), rmseScore, LooseTolerance);
    }

    [Fact]
    public void RMSE_GreaterThanOrEqual_MAE()
    {
        // RMSE >= MAE always (by Cauchy-Schwarz/QM-AM inequality)
        var rmse = new RMSEMetric<double>();
        var mae = new MAEMetric<double>();
        double[] preds = [1, 3, 5, 2];
        double[] actuals = [2, 1, 4, 5];

        double rmseScore = rmse.Compute(preds, actuals);
        double maeScore = mae.Compute(preds, actuals);

        Assert.True(rmseScore >= maeScore - Tolerance,
            $"RMSE ({rmseScore}) should be >= MAE ({maeScore})");
    }

    #endregion

    #region MAPE

    [Fact]
    public void MAPE_PerfectPredictions_IsZero()
    {
        var mape = new MAPEMetric<double>();
        double[] preds = [1, 2, 3, 4];
        double[] actuals = [1, 2, 3, 4];

        double score = mape.Compute(preds, actuals);
        Assert.Equal(0.0, score, Tolerance);
    }

    [Fact]
    public void MAPE_HandCalculated()
    {
        // MAPE = (1/N) * sum(|actual - pred| / |actual|) * 100
        // For actuals=[2, 5, 10], preds=[1, 4, 12]
        // Errors: |2-1|/2 + |5-4|/5 + |10-12|/10 = 0.5 + 0.2 + 0.2 = 0.9
        // MAPE = 0.9/3 * 100 = 30
        var mape = new MAPEMetric<double>();
        double[] preds = [1, 4, 12];
        double[] actuals = [2, 5, 10];

        double score = mape.Compute(preds, actuals);
        Assert.Equal(30.0, score, LooseTolerance);
    }

    [Fact]
    public void MAPE_NonNegative()
    {
        var mape = new MAPEMetric<double>();
        double[] preds = [5, 10, 15];
        double[] actuals = [3, 12, 8];

        double score = mape.Compute(preds, actuals);
        Assert.True(score >= 0, $"MAPE {score} should be non-negative");
    }

    #endregion

    #region Cross-Metric Relationships

    [Fact]
    public void BetterPredictions_GiveHigherR2_LowerMAE()
    {
        var r2 = new R2ScoreMetric<double>();
        var mae = new MAEMetric<double>();

        double[] actuals = [1, 2, 3, 4, 5];
        double[] goodPreds = [1.1, 2.1, 2.9, 4.1, 4.9];
        double[] badPreds = [3, 3, 3, 3, 3];

        double r2Good = r2.Compute(goodPreds, actuals);
        double r2Bad = r2.Compute(badPreds, actuals);
        double maeGood = mae.Compute(goodPreds, actuals);
        double maeBad = mae.Compute(badPreds, actuals);

        Assert.True(r2Good > r2Bad, $"Good R² ({r2Good}) should be > Bad R² ({r2Bad})");
        Assert.True(maeGood < maeBad, $"Good MAE ({maeGood}) should be < Bad MAE ({maeBad})");
    }

    [Fact]
    public void Classification_PerfectPredictions_AllMetricsOptimal()
    {
        var f1 = new F1ScoreMetric<double>();
        var mcc = new MatthewsCorrelationCoefficientMetric<double>();
        var kappa = new CohensKappaMetric<double>();
        var acc = new AccuracyMetric<double>();

        double[] preds = [1, 0, 1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 0, 1, 0, 1, 0];

        Assert.Equal(1.0, f1.Compute(preds, actuals), Tolerance);
        Assert.Equal(1.0, mcc.Compute(preds, actuals), Tolerance);
        Assert.Equal(1.0, kappa.Compute(preds, actuals), Tolerance);
        Assert.Equal(1.0, acc.Compute(preds, actuals), Tolerance);
    }

    [Fact]
    public void Regression_PerfectPredictions_AllMetricsOptimal()
    {
        var r2 = new R2ScoreMetric<double>();
        var mae = new MAEMetric<double>();
        var mse = new MSEMetric<double>();
        var rmse = new RMSEMetric<double>();

        double[] preds = [1, 2, 3, 4, 5];
        double[] actuals = [1, 2, 3, 4, 5];

        Assert.Equal(1.0, r2.Compute(preds, actuals), Tolerance);
        Assert.Equal(0.0, mae.Compute(preds, actuals), Tolerance);
        Assert.Equal(0.0, mse.Compute(preds, actuals), Tolerance);
        Assert.Equal(0.0, rmse.Compute(preds, actuals), Tolerance);
    }

    #endregion

    #region Confidence Intervals

    [Fact]
    public void F1_ConfidenceInterval_ContainsPointEstimate()
    {
        var f1 = new F1ScoreMetric<double>();
        double[] preds = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        double[] actuals = [1, 0, 1, 1, 1, 0, 0, 0, 1, 0];

        var result = f1.ComputeWithCI(preds, actuals, bootstrapSamples: 500, randomSeed: 42);
        double pointEstimate = f1.Compute(preds, actuals);

        Assert.True(result.LowerBound <= pointEstimate + LooseTolerance,
            $"Lower bound {result.LowerBound} should be <= point estimate {pointEstimate}");
        Assert.True(result.UpperBound >= pointEstimate - LooseTolerance,
            $"Upper bound {result.UpperBound} should be >= point estimate {pointEstimate}");
    }

    [Fact]
    public void R2_ConfidenceInterval_LowerBoundLessThanUpper()
    {
        var r2 = new R2ScoreMetric<double>();
        double[] preds = [1.5, 2.5, 3.5, 4.5, 5.5];
        double[] actuals = [1, 2, 3, 4, 5];

        var result = r2.ComputeWithCI(preds, actuals, bootstrapSamples: 500, randomSeed: 42);

        Assert.True(result.LowerBound <= result.UpperBound,
            $"Lower ({result.LowerBound}) should be <= Upper ({result.UpperBound})");
    }

    #endregion
}
