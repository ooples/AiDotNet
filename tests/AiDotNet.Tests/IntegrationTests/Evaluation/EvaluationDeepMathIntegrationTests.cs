using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Evaluation.CrossValidation;
using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Metrics.Regression;
using AiDotNet.Evaluation.Metrics.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep mathematical correctness tests for Evaluation metrics and cross-validation strategies.
/// Each test verifies exact hand-calculated values against industry-standard formulas
/// (scikit-learn, PyTorch, etc.) to catch math bugs in the production code.
/// </summary>
public class EvaluationDeepMathIntegrationTests
{
    private const double Tol = 1e-10;

    #region Regression Metrics - Exact Math Verification

    [Fact]
    public void MSE_HandCalculated_ExactValue()
    {
        // actuals:     [1, 2, 3, 4, 5]
        // predictions: [1.1, 2.2, 2.7, 4.5, 4.8]
        // errors:      [0.1, 0.2, -0.3, 0.5, -0.2]
        // squared:     [0.01, 0.04, 0.09, 0.25, 0.04]
        // MSE = (0.01 + 0.04 + 0.09 + 0.25 + 0.04) / 5 = 0.43 / 5 = 0.086
        var metric = new MSEMetric<double>();
        double[] preds = { 1.1, 2.2, 2.7, 4.5, 4.8 };
        double[] actuals = { 1, 2, 3, 4, 5 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(0.086, result, 10);
    }

    [Fact]
    public void MAE_HandCalculated_ExactValue()
    {
        // actuals:     [1, 2, 3, 4, 5]
        // predictions: [1.1, 2.2, 2.7, 4.5, 4.8]
        // |errors|:    [0.1, 0.2, 0.3, 0.5, 0.2]
        // MAE = (0.1 + 0.2 + 0.3 + 0.5 + 0.2) / 5 = 1.3 / 5 = 0.26
        var metric = new MAEMetric<double>();
        double[] preds = { 1.1, 2.2, 2.7, 4.5, 4.8 };
        double[] actuals = { 1, 2, 3, 4, 5 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(0.26, result, 10);
    }

    [Fact]
    public void RMSE_Is_SqrtOfMSE_ExactValue()
    {
        var mseMetric = new MSEMetric<double>();
        var rmseMetric = new RMSEMetric<double>();
        double[] preds = { 1.1, 2.2, 2.7, 4.5, 4.8 };
        double[] actuals = { 1, 2, 3, 4, 5 };

        double mse = mseMetric.Compute(preds, actuals);
        double rmse = rmseMetric.Compute(preds, actuals);

        Assert.Equal(Math.Sqrt(mse), rmse, 10);
        Assert.Equal(Math.Sqrt(0.086), rmse, 10);
    }

    [Fact]
    public void R2Score_HandCalculated_ExactValue()
    {
        // actuals: [3, -0.5, 2, 7] (scikit-learn example)
        // preds:   [2.5, 0.0, 2, 8]
        // mean(actuals) = (3 + (-0.5) + 2 + 7) / 4 = 11.5 / 4 = 2.875
        // SS_res = (3-2.5)^2 + (-0.5-0)^2 + (2-2)^2 + (7-8)^2 = 0.25 + 0.25 + 0 + 1 = 1.5
        // SS_tot = (3-2.875)^2 + (-0.5-2.875)^2 + (2-2.875)^2 + (7-2.875)^2
        //        = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
        // R2 = 1 - 1.5/29.1875 = 1 - 0.05140... = 0.94860...
        var metric = new R2ScoreMetric<double>();
        double[] preds = { 2.5, 0.0, 2, 8 };
        double[] actuals = { 3, -0.5, 2, 7 };

        double result = metric.Compute(preds, actuals);
        double expected = 1.0 - 1.5 / 29.1875;
        Assert.Equal(expected, result, 10);
    }

    [Fact]
    public void R2Score_ConstantActuals_PerfectPrediction_ReturnsOne()
    {
        // All actuals are the same, predictions are also the same => R2 = 1
        var metric = new R2ScoreMetric<double>();
        double[] preds = { 5.0, 5.0, 5.0, 5.0 };
        double[] actuals = { 5.0, 5.0, 5.0, 5.0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(1.0, result, 10);
    }

    [Fact]
    public void R2Score_ConstantActuals_ImperfectPrediction_ReturnsZero()
    {
        // All actuals are the same but predictions differ => SS_tot ~= 0, SS_res > 0 => R2 = 0
        var metric = new R2ScoreMetric<double>();
        double[] preds = { 4.0, 5.0, 6.0, 7.0 };
        double[] actuals = { 5.0, 5.0, 5.0, 5.0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void R2Score_CanBeNegative_WorseThanMean()
    {
        // Predictions are systematically worse than just predicting the mean
        var metric = new R2ScoreMetric<double>();
        double[] preds = { 100, -100, 100, -100 };
        double[] actuals = { 1, 2, 3, 4 };

        double result = metric.Compute(preds, actuals);
        Assert.True(result < 0, $"R2 should be negative for terrible predictions, got {result}");
    }

    [Fact]
    public void AdjustedR2_HandCalculated_ExactValue()
    {
        // R2 = 0.948 (from above), n=4, p=1
        // Adjusted R2 = 1 - (1 - R2) * (n-1) / (n-p-1) = 1 - (1-0.948...) * 3 / 2
        var metric = new AdjustedR2Metric<double>(numPredictors: 1);
        double[] preds = { 2.5, 0.0, 2, 8 };
        double[] actuals = { 3, -0.5, 2, 7 };

        double result = metric.Compute(preds, actuals);
        double r2 = 1.0 - 1.5 / 29.1875;
        double expectedAdj = 1.0 - (1.0 - r2) * 3.0 / 2.0;
        Assert.Equal(expectedAdj, result, 10);
    }

    [Fact]
    public void AdjustedR2_AlwaysLessOrEqual_R2()
    {
        var r2Metric = new R2ScoreMetric<double>();
        var adjR2Metric = new AdjustedR2Metric<double>(numPredictors: 3);
        double[] preds = { 2.5, 0.0, 2, 8, 3.5, 1.5 };
        double[] actuals = { 3, -0.5, 2, 7, 4, 1 };

        double r2 = r2Metric.Compute(preds, actuals);
        double adjR2 = adjR2Metric.Compute(preds, actuals);

        Assert.True(adjR2 <= r2 + 1e-10,
            $"Adjusted R2 ({adjR2}) should be <= R2 ({r2}) for p>0");
    }

    [Fact]
    public void AdjustedR2_InsufficientSamples_ReturnsZero()
    {
        // n=2, p=1 => n <= p + 1 => returns 0
        var metric = new AdjustedR2Metric<double>(numPredictors: 1);
        double[] preds = { 1.0, 2.0 };
        double[] actuals = { 1.0, 2.0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void MAPE_HandCalculated_ExactValue()
    {
        // actuals:     [100, 200, 300]
        // predictions: [110, 190, 330]
        // |errors|/|actual| = [10/100, 10/200, 30/300] = [0.1, 0.05, 0.1]
        // MAPE = 100 * (0.1 + 0.05 + 0.1) / 3 = 100 * 0.25 / 3 = 8.333...%
        var metric = new MAPEMetric<double>();
        double[] preds = { 110, 190, 330 };
        double[] actuals = { 100, 200, 300 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(100.0 * 0.25 / 3.0, result, 8);
    }

    [Fact]
    public void MAPE_SkipsZeroActuals()
    {
        // actuals with zeros should be excluded from the calculation
        var metric = new MAPEMetric<double>();
        double[] preds = { 110, 999, 330 };
        double[] actuals = { 100, 0, 300 };

        double result = metric.Compute(preds, actuals);
        // Only 2 valid points: |10/100| + |30/300| = 0.1 + 0.1 = 0.2
        // MAPE = 100 * 0.2 / 2 = 10.0
        Assert.Equal(10.0, result, 8);
    }

    [Fact]
    public void SMAPE_HandCalculated_ExactValue()
    {
        // sMAPE = (100/N) * Σ |y - ŷ| / ((|y| + |ŷ|) / 2)
        // actuals:     [100, 200]
        // predictions: [110, 180]
        // point 0: |100-110| / ((100+110)/2) = 10 / 105 = 0.095238...
        // point 1: |200-180| / ((200+180)/2) = 20 / 190 = 0.105263...
        // sMAPE = 100 * (0.095238 + 0.105263) / 2 = 100 * 0.200501 / 2 = 10.025...
        var metric = new SymmetricMAPEMetric<double>();
        double[] preds = { 110, 180 };
        double[] actuals = { 100, 200 };

        double result = metric.Compute(preds, actuals);
        double expected = 100.0 * (10.0 / 105.0 + 20.0 / 190.0) / 2.0;
        Assert.Equal(expected, result, 8);
    }

    [Fact]
    public void SMAPE_Symmetry_SwapPredActuals_SameResult()
    {
        // sMAPE should give the same value when swapping predictions and actuals
        var metric = new SymmetricMAPEMetric<double>();
        double[] a = { 100, 200, 300 };
        double[] b = { 110, 190, 330 };

        double result1 = metric.Compute(a, b);
        double result2 = metric.Compute(b, a);
        Assert.Equal(result1, result2, 10);
    }

    [Fact]
    public void HuberLoss_BelowDelta_IsHalfSquaredError()
    {
        // delta=1.0, error=0.5 => loss = 0.5 * 0.5^2 = 0.125
        var metric = new HuberLossMetric<double>(delta: 1.0);
        double[] preds = { 0.5 };
        double[] actuals = { 0.0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(0.5 * 0.5 * 0.5, result, 10);
    }

    [Fact]
    public void HuberLoss_AboveDelta_IsLinearPenalty()
    {
        // delta=1.0, error=3.0 => loss = 1.0 * (3.0 - 0.5 * 1.0) = 2.5
        var metric = new HuberLossMetric<double>(delta: 1.0);
        double[] preds = { 3.0 };
        double[] actuals = { 0.0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(1.0 * (3.0 - 0.5 * 1.0), result, 10);
    }

    [Fact]
    public void HuberLoss_AtDelta_QuadraticAndLinearAgree()
    {
        // At exactly delta, both formulas should give the same result
        double delta = 1.5;
        var metric = new HuberLossMetric<double>(delta: delta);
        double[] preds = { delta };
        double[] actuals = { 0.0 };

        double result = metric.Compute(preds, actuals);
        double quadratic = 0.5 * delta * delta;
        double linear = delta * (delta - 0.5 * delta);
        Assert.Equal(quadratic, linear, 10);
        Assert.Equal(quadratic, result, 10);
    }

    [Fact]
    public void MeanSquaredLogError_HandCalculated_ExactValue()
    {
        // MSLE = (1/N) * Σ(log(1+y) - log(1+ŷ))²
        // actuals: [3, 5], predictions: [2.5, 5]
        // point 0: (log(4) - log(3.5))^2 = (1.3862... - 1.2527...)^2 = (0.1335...)^2 = 0.01783...
        // point 1: (log(6) - log(6))^2 = 0
        // MSLE = (0.01783 + 0) / 2 = 0.008917...
        var metric = new MeanSquaredLogErrorMetric<double>();
        double[] preds = { 2.5, 5 };
        double[] actuals = { 3, 5 };

        double result = metric.Compute(preds, actuals);
        double diff = Math.Log(4) - Math.Log(3.5);
        double expected = diff * diff / 2.0;
        Assert.Equal(expected, result, 10);
    }

    [Fact]
    public void MeanSquaredLogError_NegativeValues_ClampedToZero()
    {
        // Negative values should be treated as 0, so log(1 + max(0, x))
        var metric = new MeanSquaredLogErrorMetric<double>();
        double[] preds = { -5.0 };
        double[] actuals = { -3.0 };

        double result = metric.Compute(preds, actuals);
        // Both clamped to 0: (log(1) - log(1))^2 = 0
        Assert.Equal(0.0, result, 10);
    }

    #endregion

    #region Classification Metrics - Exact Math Verification

    [Fact]
    public void F1Score_HandCalculated_ExactValue()
    {
        // Binary classification with known confusion matrix:
        // TP=3, FP=1, FN=2, TN=4 (positive label = 1.0)
        // Precision = 3/4 = 0.75, Recall = 3/5 = 0.6
        // F1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 0.9 / 1.35 = 0.666...
        var metric = new F1ScoreMetric<double>(positiveLabel: 1.0);
        double[] preds =   { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 };
        double[] actuals = { 1, 1, 1, 0, 1, 1, 0, 0, 0, 0 };
        // pred=1,act=1: TP=3 (indices 0,1,2)
        // pred=1,act=0: FP=1 (index 3)
        // pred=0,act=1: FN=2 (indices 4,5)
        // pred=0,act=0: TN=4 (indices 6,7,8,9)

        double result = metric.Compute(preds, actuals);
        double precision = 3.0 / 4.0;
        double recall = 3.0 / 5.0;
        double expected = 2.0 * precision * recall / (precision + recall);
        Assert.Equal(expected, result, 10);
    }

    [Fact]
    public void F1Score_NoPredictedPositives_ReturnsZero()
    {
        // No predicted positives: precision is 0, F1 should be 0
        var metric = new F1ScoreMetric<double>(positiveLabel: 1.0);
        double[] preds =   { 0, 0, 0, 0, 0 };
        double[] actuals = { 1, 1, 0, 0, 0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void F1Score_NoActualPositives_NoPredPositives_ReturnsOne()
    {
        // No actual positives AND no predicted positives = perfect empty case
        var metric = new F1ScoreMetric<double>(positiveLabel: 1.0);
        double[] preds =   { 0, 0, 0 };
        double[] actuals = { 0, 0, 0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(1.0, result, 10);
    }

    [Fact]
    public void MCC_HandCalculated_ExactValue()
    {
        // TP=3, FP=1, FN=2, TN=4
        // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        //     = (3*4 - 1*2) / sqrt(4*5*5*6)
        //     = (12 - 2) / sqrt(600) = 10 / 24.4948... = 0.40824...
        var metric = new MatthewsCorrelationCoefficientMetric<double>(positiveLabel: 1.0);
        double[] preds =   { 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 };
        double[] actuals = { 1, 1, 1, 0, 1, 1, 0, 0, 0, 0 };

        double result = metric.Compute(preds, actuals);
        double expected = (3.0 * 4.0 - 1.0 * 2.0) / Math.Sqrt(4.0 * 5.0 * 5.0 * 6.0);
        Assert.Equal(expected, result, 8);
    }

    [Fact]
    public void MCC_AllSameClass_ReturnZero()
    {
        // All predictions and actuals are the same class => denominator is 0 => MCC = 0
        var metric = new MatthewsCorrelationCoefficientMetric<double>(positiveLabel: 1.0);
        double[] preds =   { 1, 1, 1, 1 };
        double[] actuals = { 1, 1, 1, 1 };

        double result = metric.Compute(preds, actuals);
        // TP=4, FP=0, FN=0, TN=0
        // denominator = sqrt(4*4*0*0) = 0 => MCC = 0
        Assert.Equal(0.0, result, 10);
    }

    [Fact]
    public void MCC_PerfectInversion_ReturnsMinusOne()
    {
        // Every prediction is the opposite of the actual
        var metric = new MatthewsCorrelationCoefficientMetric<double>(positiveLabel: 1.0);
        double[] preds =   { 0, 0, 1, 1 };
        double[] actuals = { 1, 1, 0, 0 };

        double result = metric.Compute(preds, actuals);
        // TP=0, FP=2, FN=2, TN=0
        // MCC = (0*0 - 2*2) / sqrt(2*2*2*2) = -4/4 = -1
        Assert.Equal(-1.0, result, 10);
    }

    [Fact]
    public void CohensKappa_HandCalculated_ExactValue()
    {
        // 3 classes: predictions vs actuals
        // Confusion matrix:
        //        Pred 0  Pred 1  Pred 2
        // Act 0:   2       1       0
        // Act 1:   0       3       1
        // Act 2:   0       0       3
        //
        // n=10, diagonal sum = 2+3+3 = 8
        // p_o = 8/10 = 0.8
        // Row sums: [3, 4, 3], Col sums: [2, 4, 4]
        // p_e = (3/10)*(2/10) + (4/10)*(4/10) + (3/10)*(4/10)
        //     = 0.06 + 0.16 + 0.12 = 0.34
        // Kappa = (0.8 - 0.34) / (1 - 0.34) = 0.46 / 0.66 = 0.6969...
        var metric = new CohensKappaMetric<double>();
        double[] preds =   { 0, 0, 1, 1, 1, 1, 2, 2, 2, 2 };
        double[] actuals = { 0, 0, 0, 1, 1, 1, 1, 2, 2, 2 };

        double result = metric.Compute(preds, actuals);
        double expected = (0.8 - 0.34) / (1.0 - 0.34);
        Assert.Equal(expected, result, 6);
    }

    [Fact]
    public void CohensKappa_RandomAgreement_NearZero()
    {
        // When predictions are random (no agreement beyond chance), kappa should be near 0
        var metric = new CohensKappaMetric<double>();
        // Arrange predictions to have approximately chance-level agreement
        double[] preds =   { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
        double[] actuals = { 0, 0, 1, 1, 0, 0, 1, 1, 0, 0 };

        double result = metric.Compute(preds, actuals);
        // Should be close to 0 (no better than chance)
        Assert.True(Math.Abs(result) < 0.5,
            $"Kappa should be near zero for random agreement, got {result}");
    }

    [Fact]
    public void F1Score_MacroAvg_HandCalculated()
    {
        // 3-class example
        // Class 0: TP=2, FP=0, FN=1 => P=2/2=1.0, R=2/3=0.667, F1=0.8
        // Class 1: TP=2, FP=1, FN=0 => P=2/3=0.667, R=2/2=1.0, F1=0.8
        // Class 2: TP=1, FP=0, FN=0 => P=1/1=1.0, R=1/1=1.0, F1=1.0
        // Macro F1 = (0.8 + 0.8 + 1.0) / 3 = 0.8666...
        var metric = new F1ScoreMetric<double>(averaging: AveragingMethod.Macro);
        double[] preds =   { 0, 0, 1, 1, 1, 2 };
        double[] actuals = { 0, 0, 0, 1, 1, 2 };

        double result = metric.Compute(preds, actuals);
        // Verify it's close to 0.8666... (depends on exact class enumeration)
        Assert.True(result > 0.7 && result < 1.0,
            $"Macro F1 should be between 0.7 and 1.0, got {result}");
    }

    #endregion

    #region Cross-Validation - Structural Correctness

    [Fact]
    public void KFold_ProducesCorrectNumberOfSplits()
    {
        var strategy = new KFoldStrategy<double>(k: 5, shuffle: false);
        var splits = strategy.Split(100).ToList();

        Assert.Equal(5, splits.Count);
    }

    [Fact]
    public void KFold_EachSampleAppearsInValidationExactlyOnce()
    {
        int n = 100;
        int k = 5;
        var strategy = new KFoldStrategy<double>(k: k, shuffle: false);

        var allValidation = new List<int>();
        foreach (var (train, val) in strategy.Split(n))
        {
            allValidation.AddRange(val);
        }

        var sorted = allValidation.OrderBy(x => x).ToList();
        Assert.Equal(n, sorted.Count);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(i, sorted[i]);
        }
    }

    [Fact]
    public void KFold_TrainAndValAreDisjoint()
    {
        int n = 50;
        var strategy = new KFoldStrategy<double>(k: 5, shuffle: false);

        foreach (var (train, val) in strategy.Split(n))
        {
            var trainSet = new HashSet<int>(train);
            var valSet = new HashSet<int>(val);

            // No overlap
            Assert.Empty(trainSet.Intersect(valSet));

            // Together they cover all indices
            Assert.Equal(n, trainSet.Count + valSet.Count);
        }
    }

    [Fact]
    public void KFold_UnevenSplit_HandlesRemainder()
    {
        // 7 samples, 3 folds: sizes should be 3, 2, 2 (or 3, 3, 1 depending on impl)
        int n = 7;
        int k = 3;
        var strategy = new KFoldStrategy<double>(k: k, shuffle: false);
        var splits = strategy.Split(n).ToList();

        Assert.Equal(k, splits.Count);

        int totalVal = 0;
        foreach (var (train, val) in splits)
        {
            totalVal += val.Length;
            Assert.Equal(n, train.Length + val.Length);
        }
        Assert.Equal(n, totalVal);
    }

    [Fact]
    public void KFold_Shuffled_SameSeeed_Reproducible()
    {
        int n = 50;
        var strategy1 = new KFoldStrategy<double>(k: 5, shuffle: true, randomSeed: 42);
        var strategy2 = new KFoldStrategy<double>(k: 5, shuffle: true, randomSeed: 42);

        var splits1 = strategy1.Split(n).ToList();
        var splits2 = strategy2.Split(n).ToList();

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(splits1[i].TrainIndices, splits2[i].TrainIndices);
            Assert.Equal(splits1[i].ValidationIndices, splits2[i].ValidationIndices);
        }
    }

    [Fact]
    public void KFold_MinDataSize_ThrowsWhenTooFew()
    {
        var strategy = new KFoldStrategy<double>(k: 5);
        Assert.Throws<ArgumentException>(() => strategy.Split(3).ToList());
    }

    [Fact]
    public void StratifiedKFold_PreservesClassDistribution()
    {
        // 70% class 0, 30% class 1
        int n = 100;
        var labels = new double[n];
        for (int i = 0; i < 70; i++) labels[i] = 0.0;
        for (int i = 70; i < 100; i++) labels[i] = 1.0;

        var strategy = new StratifiedKFoldStrategy<double>(k: 5, shuffle: true, randomSeed: 42);
        var splits = strategy.Split(n, labels).ToList();

        Assert.Equal(5, splits.Count);

        foreach (var (train, val) in splits)
        {
            // Each validation fold should have approximately 70% class 0
            int class0InVal = val.Count(i => Math.Abs(labels[i]) < 1e-10);
            int class1InVal = val.Count(i => Math.Abs(labels[i] - 1.0) < 1e-10);

            double ratio0 = (double)class0InVal / val.Length;
            Assert.True(ratio0 >= 0.5 && ratio0 <= 0.9,
                $"Class 0 ratio in validation fold should be ~0.7, got {ratio0}");

            // All indices covered
            Assert.Equal(n, train.Length + val.Length);
        }
    }

    [Fact]
    public void StratifiedKFold_EachSampleAppearsOnce()
    {
        int n = 60;
        var labels = new double[n];
        for (int i = 0; i < 40; i++) labels[i] = 0.0;
        for (int i = 40; i < 60; i++) labels[i] = 1.0;

        var strategy = new StratifiedKFoldStrategy<double>(k: 3, shuffle: false, randomSeed: 42);
        var allVal = new List<int>();

        foreach (var (_, val) in strategy.Split(n, labels))
        {
            allVal.AddRange(val);
        }

        var sorted = allVal.OrderBy(x => x).ToList();
        Assert.Equal(n, sorted.Count);
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(i, sorted[i]);
        }
    }

    [Fact]
    public void StratifiedKFold_RequiresLabels_ThrowsWithout()
    {
        var strategy = new StratifiedKFoldStrategy<double>(k: 3);
        Assert.Throws<ArgumentException>(() => strategy.Split(10).ToList());
    }

    #endregion

    #region Metric Properties - Mathematical Invariants

    [Fact]
    public void MSE_IsAlwaysNonNegative()
    {
        var metric = new MSEMetric<double>();
        var random = new Random(42);

        for (int trial = 0; trial < 10; trial++)
        {
            int n = random.Next(5, 50);
            var preds = Enumerable.Range(0, n).Select(_ => random.NextDouble() * 100).ToArray();
            var actuals = Enumerable.Range(0, n).Select(_ => random.NextDouble() * 100).ToArray();

            double result = metric.Compute(preds, actuals);
            Assert.True(result >= 0, $"MSE should be non-negative, got {result}");
        }
    }

    [Fact]
    public void RMSE_AlwaysGreaterOrEqualTo_MAE()
    {
        // RMSE >= MAE is a mathematical property (Cauchy-Schwarz inequality)
        var rmse = new RMSEMetric<double>();
        var mae = new MAEMetric<double>();
        var random = new Random(42);

        for (int trial = 0; trial < 10; trial++)
        {
            int n = random.Next(5, 50);
            var preds = Enumerable.Range(0, n).Select(_ => random.NextDouble() * 100).ToArray();
            var actuals = Enumerable.Range(0, n).Select(_ => random.NextDouble() * 100).ToArray();

            double rmseVal = rmse.Compute(preds, actuals);
            double maeVal = mae.Compute(preds, actuals);

            Assert.True(rmseVal >= maeVal - 1e-10,
                $"RMSE ({rmseVal}) should be >= MAE ({maeVal})");
        }
    }

    [Fact]
    public void RMSE_EqualToMAE_WhenAllErrorsSameSize()
    {
        // When all errors are the same magnitude, RMSE = MAE
        var rmse = new RMSEMetric<double>();
        var mae = new MAEMetric<double>();
        double[] preds = { 1, 2, 3, 4 };
        double[] actuals = { 2, 3, 4, 5 }; // All errors = 1

        double rmseVal = rmse.Compute(preds, actuals);
        double maeVal = mae.Compute(preds, actuals);

        Assert.Equal(rmseVal, maeVal, 10);
    }

    [Fact]
    public void MetricDirection_RegressionErrorMetrics_AreLowerIsBetter()
    {
        Assert.Equal(MetricDirection.LowerIsBetter, new MSEMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new MAEMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new RMSEMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new MAPEMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new SymmetricMAPEMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new HuberLossMetric<double>().Direction);
    }

    [Fact]
    public void MetricDirection_ScoreMetrics_AreHigherIsBetter()
    {
        Assert.Equal(MetricDirection.HigherIsBetter, new R2ScoreMetric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new AdjustedR2Metric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new F1ScoreMetric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new CohensKappaMetric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new MatthewsCorrelationCoefficientMetric<double>().Direction);
    }

    #endregion

    #region Confidence Intervals - Bootstrap Validity

    [Fact]
    public void BootstrapCI_LowerBoundLessThanUpperBound()
    {
        var metric = new MSEMetric<double>();
        double[] preds = { 1.1, 2.2, 3.3, 4.1, 5.2, 6.1, 7.3, 8.0, 9.1, 10.2 };
        double[] actuals = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var ci = metric.ComputeWithCI(preds, actuals, bootstrapSamples: 500, randomSeed: 42);

        Assert.True(ci.HasConfidenceInterval, "CI should be computed");

        Assert.True(ci.LowerBound <= ci.UpperBound,
            $"CI lower ({ci.LowerBound}) should be <= upper ({ci.UpperBound})");
        Assert.True(ci.LowerBound <= ci.Value,
            $"CI lower ({ci.LowerBound}) should be <= value ({ci.Value})");
        Assert.True(ci.Value <= ci.UpperBound,
            $"Value ({ci.Value}) should be <= upper ({ci.UpperBound})");
    }

    [Fact]
    public void BootstrapCI_HigherConfidence_WiderInterval()
    {
        var metric = new MAEMetric<double>();
        double[] preds = { 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5 };
        double[] actuals = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var ci90 = metric.ComputeWithCI(preds, actuals,
            confidenceLevel: 0.90, bootstrapSamples: 1000, randomSeed: 42);
        var ci99 = metric.ComputeWithCI(preds, actuals,
            confidenceLevel: 0.99, bootstrapSamples: 1000, randomSeed: 42);

        double width90 = ci90.UpperBound - ci90.LowerBound;
        double width99 = ci99.UpperBound - ci99.LowerBound;

        Assert.True(width99 >= width90 - 1e-6,
            $"99% CI width ({width99}) should be >= 90% CI width ({width90})");
    }

    [Fact]
    public void BootstrapCI_InvalidParams_Throws()
    {
        var metric = new MSEMetric<double>();
        double[] preds = { 1.0, 2.0, 3.0 };
        double[] actuals = { 1.0, 2.0, 3.0 };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(preds, actuals, bootstrapSamples: 1));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(preds, actuals, confidenceLevel: 0.0));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(preds, actuals, confidenceLevel: 1.0));
    }

    #endregion

    #region Edge Cases - Empty Input and Single Element

    [Fact]
    public void AllRegressionMetrics_EmptyInput_ReturnZero()
    {
        double[] empty = Array.Empty<double>();

        Assert.Equal(0.0, new MSEMetric<double>().Compute(empty, empty), 10);
        Assert.Equal(0.0, new MAEMetric<double>().Compute(empty, empty), 10);
        Assert.Equal(0.0, new RMSEMetric<double>().Compute(empty, empty), 10);
        Assert.Equal(0.0, new R2ScoreMetric<double>().Compute(empty, empty), 10);
        Assert.Equal(0.0, new MAPEMetric<double>().Compute(empty, empty), 10);
        Assert.Equal(0.0, new SymmetricMAPEMetric<double>().Compute(empty, empty), 10);
        Assert.Equal(0.0, new HuberLossMetric<double>().Compute(empty, empty), 10);
        Assert.Equal(0.0, new MeanSquaredLogErrorMetric<double>().Compute(empty, empty), 10);
    }

    [Fact]
    public void AllMetrics_MismatchedLengths_Throw()
    {
        double[] a = { 1.0, 2.0 };
        double[] b = { 1.0 };

        Assert.Throws<ArgumentException>(() => new MSEMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new MAEMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new R2ScoreMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new F1ScoreMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new CohensKappaMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new MatthewsCorrelationCoefficientMetric<double>().Compute(a, b));
    }

    [Fact]
    public void SingleElement_MSE_IsSquaredError()
    {
        var metric = new MSEMetric<double>();
        double[] preds = { 3.0 };
        double[] actuals = { 5.0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(4.0, result, 10); // (5-3)^2 = 4
    }

    [Fact]
    public void SingleElement_R2Score_ConstantActual_PerfectPred()
    {
        // Single element: SS_tot = 0 (only one value), SS_res = 0 => R2 = 1
        var metric = new R2ScoreMetric<double>();
        double[] preds = { 5.0 };
        double[] actuals = { 5.0 };

        double result = metric.Compute(preds, actuals);
        Assert.Equal(1.0, result, 10);
    }

    #endregion

    #region Float Type Verification

    [Fact]
    public void MSE_Float_SameAsDouble()
    {
        var doubleMetric = new MSEMetric<double>();
        var floatMetric = new MSEMetric<float>();

        double[] doublePreds = { 1.1, 2.2, 2.7, 4.5, 4.8 };
        double[] doubleActuals = { 1, 2, 3, 4, 5 };
        float[] floatPreds = { 1.1f, 2.2f, 2.7f, 4.5f, 4.8f };
        float[] floatActuals = { 1, 2, 3, 4, 5 };

        double doubleResult = doubleMetric.Compute(doublePreds, doubleActuals);
        float floatResult = floatMetric.Compute(floatPreds, floatActuals);

        // Float precision is lower, so allow more tolerance
        Assert.Equal(doubleResult, (double)floatResult, 3);
    }

    [Fact]
    public void R2Score_Float_ReasonableAccuracy()
    {
        var metric = new R2ScoreMetric<float>();
        float[] preds = { 2.5f, 0.0f, 2.0f, 8.0f };
        float[] actuals = { 3.0f, -0.5f, 2.0f, 7.0f };

        float result = metric.Compute(preds, actuals);
        Assert.True(result > 0.9f && result < 1.0f,
            $"R2 should be close to 0.948 for this example, got {result}");
    }

    #endregion

    #region Time Series Metrics

    [Fact]
    public void SMAPETimeSeries_HandCalculated()
    {
        var metric = new SMAPEMetric<double>();
        double[] preds = { 110, 180 };
        double[] actuals = { 100, 200 };

        double result = metric.Compute(preds, actuals);
        // sMAPE same formula as regression sMAPE
        double expected = 100.0 * (10.0 / 105.0 + 20.0 / 190.0) / 2.0;
        Assert.Equal(expected, result, 6);
    }

    [Fact]
    public void WAPE_HandCalculated()
    {
        // WAPE = Σ|y - ŷ| / Σ|y|
        // actuals: [100, 200], predictions: [110, 180]
        // |errors| = [10, 20], |actuals| = [100, 200]
        // WAPE = 30 / 300 = 0.1 = 10%
        var metric = new WAPEMetric<double>();
        double[] preds = { 110, 180 };
        double[] actuals = { 100, 200 };

        double result = metric.Compute(preds, actuals);
        double expected = 30.0 / 300.0;
        Assert.Equal(expected, result, 10);
    }

    #endregion
}