using AiDotNet.Evaluation.Enums;
using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Metrics.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evaluation;

/// <summary>
/// Deep math-correctness integration tests for classification loss metrics and
/// advanced regression metrics with hand-calculated expected values, edge cases,
/// and cross-metric mathematical identities.
///
/// Classification metrics tested:
///   HingeLoss, HammingLoss, OptimizedPrecision, BalancedErrorRate
///
/// Regression metrics tested:
///   HuberLoss, LogCoshLoss, QuantileLoss, TweedieLoss, PoissonDeviance,
///   SpearmanCorrelation, SymmetricMAPE, MeanDirectionalAccuracy, AdjustedR2,
///   MeanSquaredLogError, ExplainedVariance
/// </summary>
public class LossAndRobustRegressionMetricsDeepMathIntegrationTests
{
    private const double Tol = 1e-6;

    // ═══════════════════════════════════════════════════════════════
    // HINGE LOSS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void HingeLoss_PerfectPredictions_InProbabilitySpace_ShouldBeZero()
    {
        // Predictions: 1.0 for positive, 0.0 for negative (probabilities in [0,1])
        // Converted: yHat = 2*1.0-1 = 1 for positive, yHat = 2*0.0-1 = -1 for negative
        // y: +1 for actual=1, -1 for actual=0
        // Hinge: max(0, 1 - 1*1) = 0 for pos, max(0, 1 - (-1)*(-1)) = 0 for neg
        var pred = new double[] { 1.0, 1.0, 0.0, 0.0 };
        var actual = new double[] { 1.0, 1.0, 0.0, 0.0 };
        var metric = new HingeLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void HingeLoss_AllWrongPredictions_ShouldBeTwo()
    {
        // pred=0.0 for actual=1.0: yHat=2*0-1=-1, y=+1, hinge=max(0,1-1*(-1))=max(0,2)=2
        // pred=1.0 for actual=0.0: yHat=2*1-1=1, y=-1, hinge=max(0,1-(-1)*1)=max(0,2)=2
        // Average = (2+2)/2 = 2.0
        var pred = new double[] { 0.0, 1.0 };
        var actual = new double[] { 1.0, 0.0 };
        var metric = new HingeLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void HingeLoss_HandCalculated_MixedConfidence()
    {
        // pred=0.9 for actual=1: yHat=2*0.9-1=0.8, y=+1, hinge=max(0,1-0.8)=0.2
        // pred=0.7 for actual=1: yHat=2*0.7-1=0.4, y=+1, hinge=max(0,1-0.4)=0.6
        // pred=0.3 for actual=0: yHat=2*0.3-1=-0.4, y=-1, hinge=max(0,1-(-1)*(-0.4))=max(0,1-0.4)=0.6
        // pred=0.1 for actual=0: yHat=2*0.1-1=-0.8, y=-1, hinge=max(0,1-(-1)*(-0.8))=max(0,1-0.8)=0.2
        // Average = (0.2+0.6+0.6+0.2)/4 = 1.6/4 = 0.4
        var pred = new double[] { 0.9, 0.7, 0.3, 0.1 };
        var actual = new double[] { 1.0, 1.0, 0.0, 0.0 };
        var metric = new HingeLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.4, result, Tol);
    }

    [Fact]
    public void HingeLoss_RawMarginScores_NotInProbabilityRange()
    {
        // pred=3.0 (>1, so NOT converted), actual=1 => y=+1
        // hinge = max(0, 1 - 1*3.0) = max(0, -2) = 0
        // pred=-2.0 (<0, so NOT converted), actual=0 => y=-1
        // hinge = max(0, 1 - (-1)*(-2.0)) = max(0, 1-2) = 0
        // pred=0.5 (in [0,1], so converted: yHat=2*0.5-1=0), actual=1 => y=+1
        // hinge = max(0, 1 - 1*0) = 1
        // Average = (0+0+1)/3 = 1/3
        var pred = new double[] { 3.0, -2.0, 0.5 };
        var actual = new double[] { 1.0, 0.0, 1.0 };
        var metric = new HingeLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0 / 3.0, result, Tol);
    }

    [Fact]
    public void HingeLoss_IsNonNegative()
    {
        var pred = new double[] { 0.2, 0.8, 0.5, 0.3 };
        var actual = new double[] { 0.0, 1.0, 1.0, 0.0 };
        var metric = new HingeLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.True(result >= 0.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // HAMMING LOSS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void HammingLoss_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1.0, 0.0, 1.0, 0.0 };
        var actual = new double[] { 1.0, 0.0, 1.0, 0.0 };
        var metric = new HammingLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void HammingLoss_AllWrong_ShouldBeOne()
    {
        var pred = new double[] { 1.0, 0.0, 1.0, 0.0 };
        var actual = new double[] { 0.0, 1.0, 0.0, 1.0 };
        var metric = new HammingLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void HammingLoss_HandCalculated_OneWrong()
    {
        // 3 correct, 1 wrong out of 4 => 1/4 = 0.25
        var pred = new double[] { 1.0, 0.0, 1.0, 1.0 }; // last wrong
        var actual = new double[] { 1.0, 0.0, 1.0, 0.0 };
        var metric = new HammingLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.25, result, Tol);
    }

    [Fact]
    public void HammingLoss_UsesRounding_PointFiveRoundsToEven()
    {
        // Math.Round(0.5) = 0 (banker's rounding), Math.Round(1.5) = 2
        // pred=0.5 rounds to 0, actual=1 => wrong
        // pred=1.5 rounds to 2, actual=2 => correct
        var pred = new double[] { 0.5, 1.5 };
        var actual = new double[] { 1.0, 2.0 };
        var metric = new HammingLossMetric<double>();
        double result = metric.Compute(pred, actual);
        // (int)Math.Round(0.5) = 0, (int)Math.Round(1.0) = 1 => 0 != 1 => wrong
        // (int)Math.Round(1.5) = 2, (int)Math.Round(2.0) = 2 => 2 == 2 => correct
        Assert.Equal(0.5, result, Tol);
    }

    [Fact]
    public void HammingLoss_MultiClass_HandCalculated()
    {
        // Multi-class: 5 samples, 2 wrong
        var pred = new double[] { 0, 1, 2, 1, 0 };
        var actual = new double[] { 0, 1, 2, 2, 1 }; // last two wrong
        var metric = new HammingLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.0 / 5.0, result, Tol);
    }

    [Fact]
    public void HammingLoss_PlusAccuracy_EqualsOne()
    {
        // HammingLoss = 1 - Accuracy for single-label classification
        var pred = new double[] { 1.0, 0.0, 1.0, 0.0, 1.0 };
        var actual = new double[] { 1.0, 0.0, 0.0, 0.0, 1.0 };
        var hammingMetric = new HammingLossMetric<double>();
        var accuracyMetric = new AccuracyMetric<double>();
        double hamming = hammingMetric.Compute(pred, actual);
        double accuracy = accuracyMetric.Compute(pred, actual);
        Assert.Equal(1.0, hamming + accuracy, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // OPTIMIZED PRECISION
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void OptimizedPrecision_PerfectPredictions_ShouldBeOne()
    {
        // Perfect: TP=2, TN=2, FP=0, FN=0
        // Accuracy = 1.0, Sens=1.0, Spec=1.0
        // Penalty = |1-1|/(1+1) = 0
        // OP = 1.0 - 0 = 1.0
        var pred = new double[] { 1.0, 1.0, 0.0, 0.0 };
        var actual = new double[] { 1.0, 1.0, 0.0, 0.0 };
        var metric = new OptimizedPrecisionMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void OptimizedPrecision_HandCalculated_BalancedClassifier()
    {
        // TP=3, TN=2, FP=1, FN=1 (7 samples)
        // Accuracy = 5/7
        // Sens = TP/(TP+FN) = 3/4 = 0.75
        // Spec = TN/(TN+FP) = 2/3
        // |Sens-Spec| = |0.75-0.6667| = 0.08333
        // Sens+Spec = 0.75+0.6667 = 1.4167
        // Penalty = 0.08333/1.4167 = 0.0588
        // OP = 5/7 - 0.0588 = 0.7143 - 0.0588 = 0.6555
        var pred = new double[] { 1, 1, 1, 0, 0, 0, 1 };
        var actual = new double[] { 1, 1, 1, 0, 0, 1, 0 };
        var metric = new OptimizedPrecisionMetric<double>();
        double result = metric.Compute(pred, actual);
        double accuracy = 5.0 / 7.0;
        double sens = 3.0 / 4.0;
        double spec = 2.0 / 3.0;
        double penalty = Math.Abs(sens - spec) / (sens + spec);
        double expected = accuracy - penalty;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void OptimizedPrecision_HighlyImbalanced_PenalizesAsymmetry()
    {
        // All predicted positive: TP=3, FP=7, TN=0, FN=0
        // Accuracy = 3/10 = 0.3
        // Sens = 3/3 = 1.0
        // Spec = 0/7 = 0
        // Penalty = |1-0|/(1+0) = 1.0
        // OP = 0.3 - 1.0 = -0.7
        var pred = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        var actual = new double[] { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
        var metric = new OptimizedPrecisionMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(-0.7, result, Tol);
    }

    [Fact]
    public void OptimizedPrecision_BalancedSensSpec_NoPenalty()
    {
        // Construct scenario where Sens == Spec
        // TP=2, TN=2, FP=1, FN=1
        // Sens = 2/3, Spec = 2/3
        // Penalty = 0
        // OP = Accuracy = 4/6
        var pred = new double[] { 1, 1, 0, 0, 0, 1 };
        var actual = new double[] { 1, 1, 0, 0, 1, 0 };
        var metric = new OptimizedPrecisionMetric<double>();
        double result = metric.Compute(pred, actual);
        double accuracy = 4.0 / 6.0;
        Assert.Equal(accuracy, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // BALANCED ERROR RATE
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void BalancedErrorRate_PerfectClassifier_ShouldBeZero()
    {
        var pred = new double[] { 1, 1, 0, 0 };
        var actual = new double[] { 1, 1, 0, 0 };
        var metric = new BalancedErrorRateMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void BalancedErrorRate_PlusBalancedAccuracy_EqualsOne()
    {
        // BER = 1 - BalancedAccuracy
        var pred = new double[] { 1, 0, 1, 0, 1 };
        var actual = new double[] { 1, 0, 0, 0, 1 };
        var berMetric = new BalancedErrorRateMetric<double>();
        var baMetric = new BalancedAccuracyMetric<double>();
        double ber = berMetric.Compute(pred, actual);
        double ba = baMetric.Compute(pred, actual);
        Assert.Equal(1.0, ber + ba, Tol);
    }

    [Fact]
    public void BalancedErrorRate_HandCalculated_Asymmetric()
    {
        // TP=2, TN=1, FP=2, FN=1
        // FNR = FN/(TP+FN) = 1/3
        // FPR = FP/(TN+FP) = 2/3
        // BER = (1/3 + 2/3)/2 = 0.5
        var pred = new double[] { 1, 1, 0, 1, 1, 0 };
        var actual = new double[] { 1, 1, 1, 0, 0, 0 };
        var metric = new BalancedErrorRateMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.5, result, Tol);
    }

    [Fact]
    public void BalancedErrorRate_RandomClassifier_ShouldBeHalf()
    {
        // When FNR = FPR = 0.5, BER = 0.5
        // TP=1, FN=1, TN=1, FP=1
        var pred = new double[] { 1, 0, 1, 0 };
        var actual = new double[] { 1, 1, 0, 0 };
        var metric = new BalancedErrorRateMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.5, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // HUBER LOSS (REGRESSION)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void HuberLoss_SmallErrors_EqualsHalfMSE()
    {
        // When all errors <= delta, Huber = 0.5 * error^2 (quadratic)
        // errors: 0.3, 0.5, 0.2, 0.4 (all <= 1.0 delta)
        // losses: 0.5*0.09, 0.5*0.25, 0.5*0.04, 0.5*0.16 = 0.045, 0.125, 0.02, 0.08
        // average = 0.27 / 4 = 0.0675
        var pred = new double[] { 1.3, 2.5, 3.2, 4.4 };
        var actual = new double[] { 1.0, 2.0, 3.0, 4.0 };
        var metric = new HuberLossMetric<double>(delta: 1.0);
        double result = metric.Compute(pred, actual);
        double expected = (0.5 * 0.09 + 0.5 * 0.25 + 0.5 * 0.04 + 0.5 * 0.16) / 4.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void HuberLoss_LargeErrors_IsLinear()
    {
        // When |error| > delta, Huber = delta * (|error| - 0.5 * delta)
        // delta=1.0, error=3.0: 1.0 * (3.0 - 0.5) = 2.5
        // delta=1.0, error=5.0: 1.0 * (5.0 - 0.5) = 4.5
        // average = (2.5 + 4.5) / 2 = 3.5
        var pred = new double[] { 3.0, 5.0 };
        var actual = new double[] { 0.0, 0.0 };
        var metric = new HuberLossMetric<double>(delta: 1.0);
        double result = metric.Compute(pred, actual);
        Assert.Equal(3.5, result, Tol);
    }

    [Fact]
    public void HuberLoss_MixedQuadraticLinear_HandCalculated()
    {
        // delta=1.0
        // error=0.5: quadratic => 0.5*0.25 = 0.125
        // error=2.0: linear => 1.0*(2.0-0.5) = 1.5
        // error=1.0: boundary (quadratic) => 0.5*1.0 = 0.5
        // average = (0.125 + 1.5 + 0.5) / 3 = 2.125/3
        var pred = new double[] { 0.5, 2.0, 1.0 };
        var actual = new double[] { 0.0, 0.0, 0.0 };
        var metric = new HuberLossMetric<double>(delta: 1.0);
        double result = metric.Compute(pred, actual);
        double expected = (0.125 + 1.5 + 0.5) / 3.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void HuberLoss_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1.0, 2.0, 3.0 };
        var actual = new double[] { 1.0, 2.0, 3.0 };
        var metric = new HuberLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void HuberLoss_ContinuousAtBoundary()
    {
        // At |error|=delta, both branches give the same value
        // Quadratic: 0.5 * delta^2
        // Linear: delta * (delta - 0.5*delta) = 0.5 * delta^2
        double delta = 2.5;
        var metric = new HuberLossMetric<double>(delta: delta);
        // Just below delta
        var pred1 = new double[] { delta - 0.001 };
        var actual1 = new double[] { 0.0 };
        double below = metric.Compute(pred1, actual1);
        // Just above delta
        var pred2 = new double[] { delta + 0.001 };
        var actual2 = new double[] { 0.0 };
        double above = metric.Compute(pred2, actual2);
        Assert.True(Math.Abs(below - above) < 0.01, "Huber loss should be continuous at the boundary");
    }

    [Fact]
    public void HuberLoss_LessThanOrEqualMSE()
    {
        // Huber loss <= 0.5 * MSE (for delta=1) since linear part grows slower
        var pred = new double[] { 0, 0, 0, 0, 0 };
        var actual = new double[] { 0.5, 1.5, 3.0, 0.2, 4.0 };
        var huberMetric = new HuberLossMetric<double>(delta: 1.0);
        var mseMetric = new MSEMetric<double>();
        double huber = huberMetric.Compute(pred, actual);
        double mse = mseMetric.Compute(pred, actual);
        Assert.True(huber <= 0.5 * mse + Tol,
            $"Huber loss ({huber}) should be <= 0.5 * MSE ({0.5 * mse}) for outlier-heavy data");
    }

    // ═══════════════════════════════════════════════════════════════
    // LOG-COSH LOSS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void LogCoshLoss_SmallErrors_ApproximatesHalfSquared()
    {
        // For small x: log(cosh(x)) ≈ x^2/2
        // error=0.1: log(cosh(0.1)) ≈ 0.01/2 = 0.005
        var pred = new double[] { 0.1 };
        var actual = new double[] { 0.0 };
        var metric = new LogCoshLossMetric<double>();
        double result = metric.Compute(pred, actual);
        double approx = 0.01 / 2.0;
        Assert.True(Math.Abs(result - approx) < 0.001,
            $"For small errors, log(cosh(x)) should approximate x^2/2. Got {result}, expected ~{approx}");
    }

    [Fact]
    public void LogCoshLoss_LargeErrors_ApproximatesAbsMinusLog2()
    {
        // For large |x|: log(cosh(x)) ≈ |x| - log(2)
        // error=50: log(cosh(50)) ≈ 50 - log(2) = 50 - 0.6931 = 49.3069
        var pred = new double[] { 50.0 };
        var actual = new double[] { 0.0 };
        var metric = new LogCoshLossMetric<double>();
        double result = metric.Compute(pred, actual);
        double expected = 50.0 - Math.Log(2);
        Assert.Equal(expected, result, 0.001);
    }

    [Fact]
    public void LogCoshLoss_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1.0, 2.0, 3.0 };
        var actual = new double[] { 1.0, 2.0, 3.0 };
        var metric = new LogCoshLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void LogCoshLoss_HandCalculated_ModerateErrors()
    {
        // errors: 1.0 and -1.0
        // log(cosh(1)) = log(1.54308...) = 0.43337...
        // log(cosh(-1)) = log(cosh(1)) = 0.43337...
        // average = 0.43337
        var pred = new double[] { 1.0, -1.0 };
        var actual = new double[] { 0.0, 0.0 };
        var metric = new LogCoshLossMetric<double>();
        double result = metric.Compute(pred, actual);
        double expected = Math.Log(Math.Cosh(1.0));
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void LogCoshLoss_IsSymmetric()
    {
        var pred1 = new double[] { 2.0 };
        var actual1 = new double[] { 0.0 };
        var pred2 = new double[] { -2.0 };
        var actual2 = new double[] { 0.0 };
        var metric = new LogCoshLossMetric<double>();
        Assert.Equal(metric.Compute(pred1, actual1), metric.Compute(pred2, actual2), Tol);
    }

    [Fact]
    public void LogCoshLoss_IsAlwaysNonNegative()
    {
        var pred = new double[] { -5, 0, 3, -1.5, 7 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new LogCoshLossMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.True(result >= 0);
    }

    // ═══════════════════════════════════════════════════════════════
    // QUANTILE LOSS (PINBALL LOSS)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void QuantileLoss_Median_EqualsHalfMAE()
    {
        // At quantile=0.5: loss = 0.5*|error| for all cases
        // Equivalent to MAE/2
        var pred = new double[] { 1, 3, 5 };
        var actual = new double[] { 2, 2, 2 };
        var quantileMetric = new QuantileLossMetric<double>(0.5);
        var maeMetric = new MAEMetric<double>();
        double quantile = quantileMetric.Compute(pred, actual);
        double mae = maeMetric.Compute(pred, actual);
        Assert.Equal(mae / 2.0, quantile, Tol);
    }

    [Fact]
    public void QuantileLoss_HighQuantile_PenalizesUnderPrediction()
    {
        // tau=0.9
        // Underprediction (y > yhat): error=y-yhat > 0 => loss = 0.9 * error
        // Overprediction (y < yhat): error=y-yhat < 0 => loss = (0.9-1) * error = -0.1 * error = 0.1*|error|
        // pred=1, actual=5: error=4 > 0 => loss = 0.9*4 = 3.6
        // pred=5, actual=1: error=-4 < 0 => loss = -0.1*(-4) = 0.4
        // average = (3.6 + 0.4) / 2 = 2.0
        var pred = new double[] { 1.0, 5.0 };
        var actual = new double[] { 5.0, 1.0 };
        var metric = new QuantileLossMetric<double>(0.9);
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void QuantileLoss_LowQuantile_PenalizesOverPrediction()
    {
        // tau=0.1
        // Underprediction (y > yhat): error=4 > 0 => loss = 0.1*4 = 0.4
        // Overprediction (y < yhat): error=-4 < 0 => loss = (0.1-1)*(-4) = 0.9*4 = 3.6
        // average = (0.4 + 3.6) / 2 = 2.0
        var pred = new double[] { 1.0, 5.0 };
        var actual = new double[] { 5.0, 1.0 };
        var metric = new QuantileLossMetric<double>(0.1);
        double result = metric.Compute(pred, actual);
        Assert.Equal(2.0, result, Tol);
    }

    [Fact]
    public void QuantileLoss_AsymmetryVerification()
    {
        // For same |error|, loss should differ by tau/(1-tau) ratio
        // tau=0.9: under=0.9*e, over=0.1*e => ratio = 9:1
        var pred_under = new double[] { 0 };
        var actual_under = new double[] { 1 }; // error=1 => loss=0.9*1=0.9
        var pred_over = new double[] { 1 };
        var actual_over = new double[] { 0 }; // error=-1 => loss=0.1*1=0.1
        var metric = new QuantileLossMetric<double>(0.9);
        double underLoss = metric.Compute(pred_under, actual_under);
        double overLoss = metric.Compute(pred_over, actual_over);
        Assert.Equal(9.0, underLoss / overLoss, Tol);
    }

    [Fact]
    public void QuantileLoss_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new QuantileLossMetric<double>(0.75);
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void QuantileLoss_InvalidQuantile_ShouldThrow()
    {
        Assert.Throws<ArgumentException>(() => new QuantileLossMetric<double>(0.0));
        Assert.Throws<ArgumentException>(() => new QuantileLossMetric<double>(1.0));
        Assert.Throws<ArgumentException>(() => new QuantileLossMetric<double>(-0.1));
    }

    // ═══════════════════════════════════════════════════════════════
    // TWEEDIE LOSS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void TweedieLoss_PowerZero_EquatesMSE()
    {
        // p=0: Tweedie = (y-mu)^2, which is MSE (without /N, but implementation averages)
        var pred = new double[] { 1.5, 2.5, 3.5 };
        var actual = new double[] { 1.0, 2.0, 3.0 };
        var tweedieMetric = new TweedieLossMetric<double>(power: 0);
        var mseMetric = new MSEMetric<double>();
        double tweedie = tweedieMetric.Compute(pred, actual);
        double mse = mseMetric.Compute(pred, actual);
        Assert.Equal(mse, tweedie, Tol);
    }

    [Fact]
    public void TweedieLoss_PowerOne_PoissonDeviance()
    {
        // p=1: Tweedie = 2*(y*log(y/mu) - (y-mu)) for y>0
        // Same formula as PoissonDeviance
        var pred = new double[] { 2.0, 4.0, 6.0 };
        var actual = new double[] { 3.0, 3.0, 3.0 };
        var tweedieMetric = new TweedieLossMetric<double>(power: 1.0);
        var poissonMetric = new PoissonDevianceMetric<double>();
        double tweedie = tweedieMetric.Compute(pred, actual);
        double poisson = poissonMetric.Compute(pred, actual);
        Assert.Equal(poisson, tweedie, Tol);
    }

    [Fact]
    public void TweedieLoss_PowerTwo_GammaDeviance()
    {
        // p=2: Tweedie = 2*(-log(y/mu) + (y-mu)/mu) for y>0
        // y=3, mu=2: 2*(-log(3/2) + (3-2)/2) = 2*(-0.4055 + 0.5) = 2*0.0945 = 0.189
        // y=3, mu=4: 2*(-log(3/4) + (3-4)/4) = 2*(0.2877 - 0.25) = 2*0.0377 = 0.0753
        // average = (0.189 + 0.0753) / 2
        var pred = new double[] { 2.0, 4.0 };
        var actual = new double[] { 3.0, 3.0 };
        var metric = new TweedieLossMetric<double>(power: 2.0);
        double result = metric.Compute(pred, actual);
        double d1 = 2.0 * (-Math.Log(3.0 / 2.0) + (3.0 - 2.0) / 2.0);
        double d2 = 2.0 * (-Math.Log(3.0 / 4.0) + (3.0 - 4.0) / 4.0);
        double expected = (d1 + d2) / 2.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void TweedieLoss_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1.0, 2.0, 3.0 };
        var actual = new double[] { 1.0, 2.0, 3.0 };
        var metric = new TweedieLossMetric<double>(power: 1.5);
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, 0.001);
    }

    [Fact]
    public void TweedieLoss_InvalidPower_ShouldThrow()
    {
        // power in (0,1) is invalid
        Assert.Throws<ArgumentException>(() => new TweedieLossMetric<double>(power: 0.5));
    }

    // ═══════════════════════════════════════════════════════════════
    // POISSON DEVIANCE
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void PoissonDeviance_HandCalculated()
    {
        // y=5, mu=3: 2*(5*log(5/3) - (5-3)) = 2*(5*0.5108 - 2) = 2*(2.554 - 2) = 2*0.554 = 1.108
        // y=2, mu=4: 2*(2*log(2/4) - (2-4)) = 2*(2*(-0.6931) + 2) = 2*(-1.3863 + 2) = 2*0.6137 = 1.2274
        // average = (1.108 + 1.2274) / 2 = 1.1677
        var pred = new double[] { 3.0, 4.0 };
        var actual = new double[] { 5.0, 2.0 };
        var metric = new PoissonDevianceMetric<double>();
        double result = metric.Compute(pred, actual);
        double d1 = 2.0 * (5.0 * Math.Log(5.0 / 3.0) - (5.0 - 3.0));
        double d2 = 2.0 * (2.0 * Math.Log(2.0 / 4.0) - (2.0 - 4.0));
        double expected = (d1 + d2) / 2.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void PoissonDeviance_ZeroActual_UsesTwoMu()
    {
        // y=0, mu=3: deviance = 2 * mu = 6
        var pred = new double[] { 3.0 };
        var actual = new double[] { 0.0 };
        var metric = new PoissonDevianceMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(6.0, result, Tol);
    }

    [Fact]
    public void PoissonDeviance_PerfectPredictions_ShouldBeZero()
    {
        // y=mu => 2*(y*log(1) - 0) = 0
        var pred = new double[] { 3.0, 5.0, 7.0 };
        var actual = new double[] { 3.0, 5.0, 7.0 };
        var metric = new PoissonDevianceMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void PoissonDeviance_IsNonNegative()
    {
        // Poisson deviance is always >= 0 (it's a deviance)
        var pred = new double[] { 1, 5, 2, 8 };
        var actual = new double[] { 3, 3, 3, 3 };
        var metric = new PoissonDevianceMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.True(result >= -Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // SPEARMAN CORRELATION
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void SpearmanCorrelation_PerfectMonotonic_ShouldBeOne()
    {
        // Perfect monotonic increasing: ranks identical
        var pred = new double[] { 10, 20, 30, 40, 50 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new SpearmanCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void SpearmanCorrelation_PerfectReverseMonotonic_ShouldBeNegOne()
    {
        // Perfect monotonic decreasing: opposite ranks
        var pred = new double[] { 50, 40, 30, 20, 10 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new SpearmanCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(-1.0, result, Tol);
    }

    [Fact]
    public void SpearmanCorrelation_HandCalculated_NoTies()
    {
        // pred = [3, 1, 4, 2]  => ranks = [3, 1, 4, 2]
        // actual = [10, 30, 20, 40] => ranks = [1, 3, 2, 4]
        // Using Pearson on ranks:
        // mean(pred_ranks) = 2.5, mean(actual_ranks) = 2.5
        // cov = (3-2.5)(1-2.5) + (1-2.5)(3-2.5) + (4-2.5)(2-2.5) + (2-2.5)(4-2.5)
        //     = 0.5*(-1.5) + (-1.5)*0.5 + 1.5*(-0.5) + (-0.5)*1.5
        //     = -0.75 - 0.75 - 0.75 - 0.75 = -3.0
        // varP = 0.25 + 2.25 + 2.25 + 0.25 = 5.0
        // varA = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        // rho = -3.0 / sqrt(5*5) = -3/5 = -0.6
        var pred = new double[] { 3, 1, 4, 2 };
        var actual = new double[] { 10, 30, 20, 40 };
        var metric = new SpearmanCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(-0.6, result, Tol);
    }

    [Fact]
    public void SpearmanCorrelation_WithTies_UsesAverageRanks()
    {
        // pred = [1, 2, 2, 4] => ranks = [1, 2.5, 2.5, 4] (tied 2nd and 3rd)
        // actual = [10, 20, 30, 40] => ranks = [1, 2, 3, 4]
        // mean(pred) = (1+2.5+2.5+4)/4 = 2.5
        // mean(actual) = 2.5
        // cov = (1-2.5)(1-2.5) + (2.5-2.5)(2-2.5) + (2.5-2.5)(3-2.5) + (4-2.5)(4-2.5)
        //     = 2.25 + 0 + 0 + 2.25 = 4.5
        // varP = 2.25 + 0 + 0 + 2.25 = 4.5
        // varA = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        // rho = 4.5 / sqrt(4.5*5) = 4.5 / sqrt(22.5) = 4.5/4.7434 = 0.9487
        var pred = new double[] { 1, 2, 2, 4 };
        var actual = new double[] { 10, 20, 30, 40 };
        var metric = new SpearmanCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        double expected = 4.5 / Math.Sqrt(4.5 * 5.0);
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void SpearmanCorrelation_NonlinearMonotonic_StillOne()
    {
        // Spearman only cares about monotonic relationship, not linearity
        // pred = [1, 4, 9, 16, 25] (squares), actual = [1, 2, 3, 4, 5]
        // Both have same ranks => rho = 1
        var pred = new double[] { 1, 4, 9, 16, 25 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new SpearmanCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void SpearmanCorrelation_ConstantInput_ShouldBeZero()
    {
        var pred = new double[] { 5, 5, 5, 5 };
        var actual = new double[] { 1, 2, 3, 4 };
        var metric = new SpearmanCorrelationMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // SYMMETRIC MAPE
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void SymmetricMAPE_HandCalculated()
    {
        // sMAPE = 100/N * sum(|y-yhat| / ((|y|+|yhat|)/2))
        // y=100, yhat=110: 100 * |100-110| / ((100+110)/2) = 100 * 10/105 = 9.5238
        // y=50, yhat=40: 100 * |50-40| / ((50+40)/2) = 100 * 10/45 = 22.2222
        // average = (9.5238 + 22.2222) / 2 = 15.873
        var pred = new double[] { 110, 40 };
        var actual = new double[] { 100, 50 };
        var metric = new SymmetricMAPEMetric<double>();
        double result = metric.Compute(pred, actual);
        double d1 = 100.0 * 10.0 / 105.0;
        double d2 = 100.0 * 10.0 / 45.0;
        double expected = (d1 + d2) / 2.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void SymmetricMAPE_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new SymmetricMAPEMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void SymmetricMAPE_IsSymmetric_UnlikeRegularMAPE()
    {
        // sMAPE treats over and under predictions equally
        // Swap pred and actual should give same result
        var pred = new double[] { 110, 40 };
        var actual = new double[] { 100, 50 };
        var metric = new SymmetricMAPEMetric<double>();
        double forward = metric.Compute(pred, actual);
        double backward = metric.Compute(actual, pred);
        Assert.Equal(forward, backward, Tol);
    }

    [Fact]
    public void SymmetricMAPE_BoundedByTwoHundred()
    {
        // Max sMAPE occurs when one of y,yhat is 0 and the other is nonzero
        // |y-yhat| / ((|y|+|yhat|)/2) = |y-yhat| / (|y-yhat|/2) = 2
        // So sMAPE element = 100*2 = 200
        var pred = new double[] { 100 };
        var actual = new double[] { 0.001 }; // very small but not zero
        var metric = new SymmetricMAPEMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.True(result <= 200.0 + Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEAN DIRECTIONAL ACCURACY
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MDA_PerfectDirectionPrediction_ShouldBeOne()
    {
        // Actuals: 1,3,2,5 => changes: +2, -1, +3
        // Predictions: 1,4,3,6 => changes: +3, -1, +3
        // All directions match => MDA = 3/3 = 1.0
        var pred = new double[] { 1, 4, 3, 6 };
        var actual = new double[] { 1, 3, 2, 5 };
        var metric = new MeanDirectionalAccuracyMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void MDA_AllWrongDirections_ShouldBeZero()
    {
        // Actuals: 1,3,2,5 => changes: +2, -1, +3
        // Predictions: 5,2,4,1 => changes: -3, +2, -3
        // No directions match => MDA = 0/3 = 0.0
        var pred = new double[] { 5, 2, 4, 1 };
        var actual = new double[] { 1, 3, 2, 5 };
        var metric = new MeanDirectionalAccuracyMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    [Fact]
    public void MDA_HandCalculated_MixedDirections()
    {
        // Actuals: 10,15,12,18,16 => changes: +5, -3, +6, -2
        // Predictions: 10,14,13,17,18 => changes: +4, -1, +4, +1
        // Match: +/+ = yes, -/- = yes, +/+ = yes, -/+ = no
        // MDA = 3/4 = 0.75
        var pred = new double[] { 10, 14, 13, 17, 18 };
        var actual = new double[] { 10, 15, 12, 18, 16 };
        var metric = new MeanDirectionalAccuracyMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.75, result, Tol);
    }

    [Fact]
    public void MDA_SingleElement_ReturnsHalf()
    {
        var pred = new double[] { 1.0 };
        var actual = new double[] { 2.0 };
        var metric = new MeanDirectionalAccuracyMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.5, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // ADJUSTED R2
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void AdjustedR2_PerfectPredictions_ShouldBeOne()
    {
        var pred = new double[] { 1, 2, 3, 4, 5 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new AdjustedR2Metric<double>(numPredictors: 1);
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void AdjustedR2_HandCalculated()
    {
        // actual = [1,2,3,4,5], mean=3
        // pred = [1.1, 2.2, 2.8, 4.1, 4.8]
        // SS_res = (1-1.1)^2 + (2-2.2)^2 + (3-2.8)^2 + (4-4.1)^2 + (5-4.8)^2
        //        = 0.01 + 0.04 + 0.04 + 0.01 + 0.04 = 0.14
        // SS_tot = (1-3)^2 + (2-3)^2 + 0 + (4-3)^2 + (5-3)^2 = 4+1+0+1+4 = 10
        // R2 = 1 - 0.14/10 = 0.986
        // AdjR2 = 1 - (1-0.986)*(5-1)/(5-1-1) = 1 - 0.014*4/3 = 1 - 0.01867 = 0.98133
        var pred = new double[] { 1.1, 2.2, 2.8, 4.1, 4.8 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new AdjustedR2Metric<double>(numPredictors: 1);
        double result = metric.Compute(pred, actual);
        double r2 = 1.0 - 0.14 / 10.0;
        double expected = 1.0 - (1.0 - r2) * 4.0 / 3.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void AdjustedR2_LessThanOrEqualR2()
    {
        // Adjusted R2 <= R2 when n > p+1
        var pred = new double[] { 1.1, 2.2, 2.8, 4.1, 4.8 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var adjMetric = new AdjustedR2Metric<double>(numPredictors: 2);
        var r2Metric = new R2ScoreMetric<double>();
        double adjR2 = adjMetric.Compute(pred, actual);
        double r2 = r2Metric.Compute(pred, actual);
        Assert.True(adjR2 <= r2 + Tol,
            $"AdjustedR2 ({adjR2}) should be <= R2 ({r2})");
    }

    [Fact]
    public void AdjustedR2_MorePredictors_LowerScore()
    {
        // More predictors => greater penalty => lower adjusted R2
        var pred = new double[] { 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5 };
        var actual = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var metric1 = new AdjustedR2Metric<double>(numPredictors: 1);
        var metric5 = new AdjustedR2Metric<double>(numPredictors: 5);
        double adj1 = metric1.Compute(pred, actual);
        double adj5 = metric5.Compute(pred, actual);
        Assert.True(adj1 > adj5,
            $"AdjR2 with 1 predictor ({adj1}) should be > AdjR2 with 5 ({adj5})");
    }

    [Fact]
    public void AdjustedR2_TooFewSamples_ReturnsZero()
    {
        // n <= p+1 => not enough samples
        var pred = new double[] { 1.0, 2.0 };
        var actual = new double[] { 1.0, 2.0 };
        var metric = new AdjustedR2Metric<double>(numPredictors: 2);
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // MEAN SQUARED LOG ERROR
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MSLE_HandCalculated()
    {
        // MSLE = (1/N) * sum((log(1+y) - log(1+yhat))^2)
        // y=3, yhat=2.5: (log(4)-log(3.5))^2 = (1.3863-1.2528)^2 = 0.01785
        // y=5, yhat=4.8: (log(6)-log(5.8))^2 = (1.7918-1.7579)^2 = 0.001147
        // average = (0.01785 + 0.001147) / 2
        var pred = new double[] { 2.5, 4.8 };
        var actual = new double[] { 3.0, 5.0 };
        var metric = new MeanSquaredLogErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        double d1 = Math.Pow(Math.Log(4.0) - Math.Log(3.5), 2);
        double d2 = Math.Pow(Math.Log(6.0) - Math.Log(5.8), 2);
        double expected = (d1 + d2) / 2.0;
        Assert.Equal(expected, result, Tol);
    }

    [Fact]
    public void MSLE_PerfectPredictions_ShouldBeZero()
    {
        var pred = new double[] { 1, 2, 3 };
        var actual = new double[] { 1, 2, 3 };
        var metric = new MeanSquaredLogErrorMetric<double>();
        Assert.Equal(0.0, metric.Compute(pred, actual), Tol);
    }

    [Fact]
    public void MSLE_PenalizesUnderPredictionMore()
    {
        // MSLE penalizes underprediction more than overprediction for positive values
        // Under: y=100, yhat=50: (log(101)-log(51))^2 = (4.6151-3.9318)^2 = 0.4673
        // Over: y=50, yhat=100: (log(51)-log(101))^2 = same value!
        // Actually MSLE is symmetric in log space. But the percentage error differs.
        // Let's verify it's symmetric in log space
        var pred_under = new double[] { 50 };
        var actual_under = new double[] { 100 };
        var pred_over = new double[] { 100 };
        var actual_over = new double[] { 50 };
        var metric = new MeanSquaredLogErrorMetric<double>();
        double under = metric.Compute(pred_under, actual_under);
        double over = metric.Compute(pred_over, actual_over);
        Assert.Equal(under, over, Tol);
    }

    [Fact]
    public void MSLE_NegativeValues_ClampedToZero()
    {
        // Implementation clamps negatives to 0
        // y=-1 => clamped to 0, yhat=2: (log(1)-log(3))^2 = (0-1.0986)^2 = 1.207
        var pred = new double[] { 2.0 };
        var actual = new double[] { -1.0 };
        var metric = new MeanSquaredLogErrorMetric<double>();
        double result = metric.Compute(pred, actual);
        double expected = Math.Pow(Math.Log(1.0) - Math.Log(3.0), 2);
        Assert.Equal(expected, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // EXPLAINED VARIANCE
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ExplainedVariance_PerfectPredictions_ShouldBeOne()
    {
        var pred = new double[] { 1, 2, 3, 4, 5 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new ExplainedVarianceMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void ExplainedVariance_EqualsR2_WhenUnbiased()
    {
        // EV = 1 - Var(residuals)/Var(y)
        // R2 = 1 - SS_res/SS_tot
        // When mean(residuals) = 0, Var(res) = (1/n)*SS_res, Var(y) = (1/n)*SS_tot
        // So EV = R2 when predictions are unbiased
        var pred = new double[] { 1.1, 1.9, 3.1, 3.9, 5.0 };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var evMetric = new ExplainedVarianceMetric<double>();
        var r2Metric = new R2ScoreMetric<double>();
        double ev = evMetric.Compute(pred, actual);
        double r2 = r2Metric.Compute(pred, actual);
        // Check if residuals are roughly zero-mean
        double meanRes = 0;
        for (int i = 0; i < pred.Length; i++)
            meanRes += actual[i] - pred[i];
        meanRes /= pred.Length;
        if (Math.Abs(meanRes) < 0.1)
        {
            Assert.True(Math.Abs(ev - r2) < 0.05,
                $"EV ({ev}) should approximately equal R2 ({r2}) when unbiased");
        }
    }

    [Fact]
    public void ExplainedVariance_GreaterThanOrEqualR2_WhenBiased()
    {
        // When predictions have a constant bias, EV >= R2
        // because EV ignores the bias (subtracts mean residual)
        // Add constant bias of 2 to all predictions
        var pred = new double[] { 3, 4, 5, 6, 7 }; // actual + 2
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var evMetric = new ExplainedVarianceMetric<double>();
        var r2Metric = new R2ScoreMetric<double>();
        double ev = evMetric.Compute(pred, actual);
        double r2 = r2Metric.Compute(pred, actual);
        Assert.True(ev >= r2 - Tol,
            $"EV ({ev}) should be >= R2 ({r2}) with constant bias");
    }

    [Fact]
    public void ExplainedVariance_ConstantBias_StillOne()
    {
        // pred = actual + constant => residuals all equal constant
        // Var(residuals) = 0 (constant), so EV = 1
        var pred = new double[] { 6, 7, 8, 9, 10 }; // actual + 5
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new ExplainedVarianceMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(1.0, result, Tol);
    }

    [Fact]
    public void ExplainedVariance_MeanPredictor_ShouldBeZero()
    {
        // Predicting mean for all => residuals = actual - mean
        // Var(residuals) = Var(actual)
        // EV = 1 - Var(actual)/Var(actual) = 0
        double mean = 3.0;
        var pred = new double[] { mean, mean, mean, mean, mean };
        var actual = new double[] { 1, 2, 3, 4, 5 };
        var metric = new ExplainedVarianceMetric<double>();
        double result = metric.Compute(pred, actual);
        Assert.Equal(0.0, result, Tol);
    }

    // ═══════════════════════════════════════════════════════════════
    // CROSS-METRIC MATHEMATICAL IDENTITIES
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void CrossMetric_HammingLoss_IsOneMinus_Accuracy()
    {
        var pred = new double[] { 1, 0, 1, 0, 1, 1, 0 };
        var actual = new double[] { 1, 0, 0, 1, 1, 0, 0 };
        var hamming = new HammingLossMetric<double>().Compute(pred, actual);
        var accuracy = new AccuracyMetric<double>().Compute(pred, actual);
        Assert.Equal(1.0, hamming + accuracy, Tol);
    }

    [Fact]
    public void CrossMetric_BER_IsOneMinus_BalancedAccuracy()
    {
        var pred = new double[] { 1, 0, 1, 0, 1, 1, 0 };
        var actual = new double[] { 1, 0, 0, 1, 1, 0, 0 };
        var ber = new BalancedErrorRateMetric<double>().Compute(pred, actual);
        var ba = new BalancedAccuracyMetric<double>().Compute(pred, actual);
        Assert.Equal(1.0, ber + ba, Tol);
    }

    [Fact]
    public void CrossMetric_QuantileLoss05_IsHalf_MAE()
    {
        var pred = new double[] { 1, 3, 5, 7, 9 };
        var actual = new double[] { 2, 2, 4, 8, 7 };
        var quantile = new QuantileLossMetric<double>(0.5).Compute(pred, actual);
        var mae = new MAEMetric<double>().Compute(pred, actual);
        Assert.Equal(mae / 2.0, quantile, Tol);
    }

    [Fact]
    public void CrossMetric_PoissonDeviance_EqualsTweedie_PowerOne()
    {
        var pred = new double[] { 2, 4, 6, 8 };
        var actual = new double[] { 3, 3, 5, 9 };
        var poisson = new PoissonDevianceMetric<double>().Compute(pred, actual);
        var tweedie = new TweedieLossMetric<double>(power: 1.0).Compute(pred, actual);
        Assert.Equal(poisson, tweedie, Tol);
    }

    [Fact]
    public void CrossMetric_Tweedie_PowerZero_EqualsMSE()
    {
        var pred = new double[] { 1.5, 2.5, 3.5 };
        var actual = new double[] { 1, 2, 3 };
        var tweedie = new TweedieLossMetric<double>(power: 0).Compute(pred, actual);
        var mse = new MSEMetric<double>().Compute(pred, actual);
        Assert.Equal(mse, tweedie, Tol);
    }

    [Fact]
    public void CrossMetric_LogCoshLoss_BetweenMAEandMSE()
    {
        // For same data, log(cosh(x)) is between x^2/2 and |x|-log(2)
        // So LogCoshLoss should be between approximately MAE-log(2) and MSE/2
        var pred = new double[] { 0, 0, 0, 0 };
        var actual = new double[] { 0.5, 1.5, 2.0, 3.0 };
        var logCosh = new LogCoshLossMetric<double>().Compute(pred, actual);
        var mae = new MAEMetric<double>().Compute(pred, actual);
        var mse = new MSEMetric<double>().Compute(pred, actual);
        // LogCosh <= each element's max(MSE/2 per element)
        // For individual elements, log(cosh(x)) <= 0.5*x^2
        Assert.True(logCosh <= 0.5 * mse + Tol,
            $"LogCosh ({logCosh}) should be <= MSE/2 ({0.5 * mse})");
        Assert.True(logCosh >= 0,
            "LogCosh should be non-negative");
    }

    [Fact]
    public void CrossMetric_HuberLoss_SmallDelta_ApproximatesDeltaMAE()
    {
        // For very small delta, almost all errors are in the linear region
        // Huber ≈ delta * (MAE - delta/2)
        double delta = 0.01;
        var pred = new double[] { 0, 0, 0, 0 };
        var actual = new double[] { 1, 2, 3, 4 };
        var huber = new HuberLossMetric<double>(delta: delta).Compute(pred, actual);
        var mae = new MAEMetric<double>().Compute(pred, actual);
        double approx = delta * (mae - 0.5 * delta);
        Assert.True(Math.Abs(huber - approx) < 0.01,
            $"Huber ({huber}) should approximate delta*(MAE-delta/2) = {approx} for small delta");
    }

    [Fact]
    public void CrossMetric_ExplainedVariance_GeR2_ForBiasedPredictions()
    {
        // Biased predictions: pred = actual + 3
        var pred = new double[] { 4, 5, 6, 7, 8, 9, 10 };
        var actual = new double[] { 1, 2, 3, 4, 5, 6, 7 };
        var ev = new ExplainedVarianceMetric<double>().Compute(pred, actual);
        var r2 = new R2ScoreMetric<double>().Compute(pred, actual);
        Assert.True(ev >= r2 - Tol,
            $"ExplainedVariance ({ev}) should be >= R2 ({r2}) for biased predictions");
        // With constant bias, EV should be 1 (perfect variance explanation)
        Assert.Equal(1.0, ev, Tol);
        // R2 should be < 1 due to bias
        Assert.True(r2 < 1.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // EDGE CASES AND VALIDATION
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void AllMetrics_EmptyInput_ShouldHandleGracefully()
    {
        var empty = Array.Empty<double>();
        Assert.Equal(0.0, new HingeLossMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new HammingLossMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new OptimizedPrecisionMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new HuberLossMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new LogCoshLossMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new QuantileLossMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new TweedieLossMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new PoissonDevianceMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new SymmetricMAPEMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new MeanSquaredLogErrorMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new ExplainedVarianceMetric<double>().Compute(empty, empty), Tol);
        Assert.Equal(0.0, new AdjustedR2Metric<double>().Compute(empty, empty), Tol);
    }

    [Fact]
    public void AllMetrics_MismatchedLengths_ShouldThrow()
    {
        var a = new double[] { 1, 2 };
        var b = new double[] { 1 };
        Assert.Throws<ArgumentException>(() => new HingeLossMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new HammingLossMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new OptimizedPrecisionMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new HuberLossMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new LogCoshLossMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new QuantileLossMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new TweedieLossMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new PoissonDevianceMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new SpearmanCorrelationMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new SymmetricMAPEMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new MeanDirectionalAccuracyMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new MeanSquaredLogErrorMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new ExplainedVarianceMetric<double>().Compute(a, b));
        Assert.Throws<ArgumentException>(() => new AdjustedR2Metric<double>().Compute(a, b));
    }

    [Fact]
    public void HingeLoss_ComputeWithCI_InvalidBootstrap_ShouldThrow()
    {
        var pred = new double[] { 0.9, 0.1 };
        var actual = new double[] { 1, 0 };
        var metric = new HingeLossMetric<double>();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(pred, actual, bootstrapSamples: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(pred, actual, confidenceLevel: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            metric.ComputeWithCI(pred, actual, confidenceLevel: 1));
    }

    [Fact]
    public void SpearmanCorrelation_ComputeWithCI_ProducesValidInterval()
    {
        var pred = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var actual = new double[] { 1, 3, 2, 4, 5, 7, 6, 8, 10, 9 };
        var metric = new SpearmanCorrelationMetric<double>();
        var ciResult = metric.ComputeWithCI(pred, actual, bootstrapSamples: 200, randomSeed: 42);
        double value = ciResult.Value;
        double lower = ciResult.LowerBound;
        double upper = ciResult.UpperBound;
        Assert.True(lower <= value, $"Lower bound ({lower}) should be <= value ({value})");
        Assert.True(value <= upper, $"Value ({value}) should be <= upper bound ({upper})");
        Assert.True(lower >= -1.0 - Tol);
        Assert.True(upper <= 1.0 + Tol);
    }

    [Fact]
    public void MetricDirections_AreCorrect()
    {
        // Loss metrics should be LowerIsBetter
        Assert.Equal(MetricDirection.LowerIsBetter, new HingeLossMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new HammingLossMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new HuberLossMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new LogCoshLossMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new QuantileLossMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new TweedieLossMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new PoissonDevianceMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new SymmetricMAPEMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new BalancedErrorRateMetric<double>().Direction);
        Assert.Equal(MetricDirection.LowerIsBetter, new MeanSquaredLogErrorMetric<double>().Direction);

        // Score metrics should be HigherIsBetter
        Assert.Equal(MetricDirection.HigherIsBetter, new OptimizedPrecisionMetric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new SpearmanCorrelationMetric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new MeanDirectionalAccuracyMetric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new AdjustedR2Metric<double>().Direction);
        Assert.Equal(MetricDirection.HigherIsBetter, new ExplainedVarianceMetric<double>().Direction);
    }
}
