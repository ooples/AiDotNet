using AiDotNet.UncertaintyQuantification.Calibration;
using AiDotNet.UncertaintyQuantification.ConformalPrediction;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.UncertaintyQuantification;

/// <summary>
/// Reference values taken from the papers the calibration and conformal classes implement.
/// </summary>
public class UncertaintyQuantificationPaperReferenceTests
{
    /// <summary>
    /// Guo et al. 2017 (Sec. 2) bin m holds confidences in ((m-1)/M, m/M]. With M = 10, 0.2 and 0.15 share
    /// bin (0.1, 0.2]: conf 0.175, acc 0.5, ECE 0.325. Master put 0.2 in [0.2, 0.3) and returned 0.475.
    /// </summary>
    [Fact]
    public void Ece_BinsAreRightClosed()
    {
        var ece = new ExpectedCalibrationError<double>(numBins: 10);
        var probs = new Vector<double>(new[] { 0.2, 0.15 });
        var preds = new Vector<int>(new[] { 1, 1 });
        var labels = new Vector<int>(new[] { 1, 0 });

        Assert.Equal(0.325, ece.Compute(probs, preds, labels), 12);

        var diagram = ece.GetReliabilityDiagram(probs, preds, labels);
        Assert.Single(diagram);
        Assert.Equal(0.175, diagram[0].confidence, 12);
        Assert.Equal(0.5, diagram[0].accuracy, 12);
        Assert.Equal(2, diagram[0].count);
    }

    /// <summary>
    /// Angelopoulos and Bates 2021, Sec. 1: q = the ceil((n+1)(1-alpha))-th smallest score. For n = 100,
    /// alpha = 0.1 that is the 91st smallest non-conformity 1 - p, i.e. the 10th smallest conformity p
    /// (0.10 for scores 0.01..1.00). Master used floor(n alpha) = index 10 (0.11), under-covering.
    /// </summary>
    [Fact]
    public void ConformalClassifier_Threshold_IsFiniteSampleQuantile()
    {
        var scores = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            scores[i] = (i + 1) / 100.0;
        }

        Assert.Equal(0.10, ConformalClassifier<double>.ConformalThreshold(scores, 0.9), 12);
    }

    /// <summary>When ceil((n+1)(1-alpha)) &gt; n the set is every class; master returned the minimum score.</summary>
    [Fact]
    public void ConformalClassifier_TooFewCalibrationSamples_IncludesEveryClass()
    {
        var scores = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
        Assert.True(double.IsNegativeInfinity(ConformalClassifier<double>.ConformalThreshold(scores, 0.9)));
    }

    /// <summary>Split conformal: k = ceil(6 * 0.9) = 6 &gt; n = 5, so the interval is unbounded (master used the max).</summary>
    [Fact]
    public void SplitConformal_TooFewCalibrationSamples_IsInfinite()
    {
        var scores = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4, 0.5 });
        Assert.True(double.IsPositiveInfinity(SplitConformalPredictor<double>.ComputeQuantile(scores, 0.9)));
    }

    [Fact]
    public void SplitConformal_Quantile_IsKthSmallest()
    {
        var scores = new Vector<double>(19);
        for (int i = 0; i < 19; i++)
        {
            scores[i] = i + 1;
        }

        // k = ceil(20 * 0.9) = 18
        Assert.Equal(18.0, SplitConformalPredictor<double>.ComputeQuantile(scores, 0.9), 12);
    }

    /// <summary>One gradient step from T = 1 on logits [5, 0, 0], label 1: dNLL/dT = z_y - sum p z.</summary>
    [Fact]
    public void TemperatureScaling_OneStep_MatchesAnalyticGradient()
    {
        var ts = new TemperatureScaling<double>(initialTemperature: 1.0);
        var logits = new Matrix<double>(1, 3);
        logits[0, 0] = 5.0;
        var labels = new Vector<int>(new[] { 1 });

        double e5 = System.Math.Exp(5.0);
        double p0 = e5 / (e5 + 2.0);
        double gradient = 0.0 - p0 * 5.0;

        ts.Calibrate(logits, labels, learningRate: 0.01, maxIterations: 1);
        Assert.Equal(1.0 - 0.01 * gradient, ts.Temperature, 10);
    }
}
