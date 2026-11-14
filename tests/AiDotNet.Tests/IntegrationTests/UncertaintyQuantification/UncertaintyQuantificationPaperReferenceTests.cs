using AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;
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

    /// <summary>
    /// Coverage is the fraction of the test rows whose label is in its set, so every row needs exactly one
    /// label. Master gathered only the first two rows' sets for two labels and divided by three.
    /// </summary>
    [Fact]
    public void ConformalClassifier_EvaluateCoverage_RejectsLabelCountMismatch()
    {
        var classifier = new ConformalClassifier<double>(new AiDotNet.Tests.Helpers.MockNeuralNetwork(outputSize: 3), numClasses: 3);
        classifier.Calibrate(new Matrix<double>(4, 1), new Vector<int>(new[] { 0, 1, 2, 0 }));

        var ex = Assert.Throws<ArgumentException>(
            () => classifier.EvaluateCoverage(new Matrix<double>(3, 1), new Vector<int>(new[] { 0, 1 })));
        Assert.Equal("testLabels", ex.ParamName);
    }

    /// <summary>Split conformal coverage needs one target per test row.</summary>
    [Fact]
    public void SplitConformal_EvaluateCoverage_RejectsTargetCountMismatch()
    {
        var predictor = new SplitConformalPredictor<double>(new AiDotNet.Tests.Helpers.MockNeuralNetwork(outputSize: 1));
        predictor.Calibrate(new Matrix<double>(4, 1), new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 }));

        var ex = Assert.Throws<ArgumentException>(
            () => predictor.EvaluateCoverage(new Matrix<double>(3, 1), new Vector<double>(new[] { 0.0, 1.0 })));
        Assert.Equal("testTargets", ex.ParamName);
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

    /// <summary>
    /// Two MC samples [0.9, 0.1] and [0.1, 0.9]: H[mean] = ln 2, E H[p_s] = H(0.9, 0.1), BALD MI is the
    /// difference (Houlsby et al. 2011); aleatoric E[p(1-p)] = 0.09, epistemic Var[p] = 0.16
    /// (Kwon et al. 2020). Master reported zero entropy and MI and a fixed 30/70 split of the variance.
    /// </summary>
    [Fact]
    public void PredictiveStatistics_ClassProbabilities_MatchBaldDecomposition()
    {
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>([1, 2], new Vector<double>(new[] { 0.9, 0.1 })),
            new Tensor<double>([1, 2], new Vector<double>(new[] { 0.1, 0.9 })),
        };

        var mean = PredictiveSampleStatistics<double>.Mean(samples);
        var variance = PredictiveSampleStatistics<double>.Variance(samples, mean);
        var aleatoric = PredictiveSampleStatistics<double>.AleatoricVariance(samples, mean);
        var metrics = PredictiveSampleStatistics<double>.Metrics(samples, mean);

        double sampleEntropy = -(0.9 * System.Math.Log(0.9) + 0.1 * System.Math.Log(0.1));
        Assert.Equal(0.5, mean[0, 0], 12);
        Assert.Equal(0.16, variance[0, 0], 12);
        Assert.Equal(0.09, aleatoric[0, 1], 12);
        Assert.Equal(System.Math.Log(2.0), metrics["predictive_entropy"][0], 12);
        Assert.Equal(System.Math.Log(2.0) - sampleEntropy, metrics["mutual_information"][0], 12);
    }

    /// <summary>Regression outputs carry no noise head: all sample variance is epistemic, aleatoric is zero.</summary>
    [Fact]
    public void PredictiveStatistics_Regression_AllVarianceIsEpistemic()
    {
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>([1, 1], new Vector<double>(new[] { 3.0 })),
            new Tensor<double>([1, 1], new Vector<double>(new[] { 5.0 })),
        };

        var mean = PredictiveSampleStatistics<double>.Mean(samples);
        Assert.Equal(4.0, mean[0, 0], 12);
        Assert.Equal(1.0, PredictiveSampleStatistics<double>.Variance(samples, mean)[0, 0], 12);
        Assert.Equal(0.0, PredictiveSampleStatistics<double>.AleatoricVariance(samples, mean)[0, 0], 12);
    }

    /// <summary>
    /// Two-output regression samples [2, -1] and [-1, 2] average to [0.5, 0.5], which looks like a class
    /// distribution. The layout check ran on the mean alone, so E[p (1 - p)] = -2 was reported as aleatoric
    /// variance. Each sample is checked, so these stay on the regression path: zero aleatoric part and zero
    /// entropy metrics, with all of Var = 2.25 epistemic.
    /// </summary>
    [Fact]
    public void PredictiveStatistics_RegressionSamplesWithProbabilityLikeMean_StayRegression()
    {
        var samples = new List<Tensor<double>>
        {
            new Tensor<double>([1, 2], new Vector<double>(new[] { 2.0, -1.0 })),
            new Tensor<double>([1, 2], new Vector<double>(new[] { -1.0, 2.0 })),
        };

        var mean = PredictiveSampleStatistics<double>.Mean(samples);
        var aleatoric = PredictiveSampleStatistics<double>.AleatoricVariance(samples, mean);
        var metrics = PredictiveSampleStatistics<double>.Metrics(samples, mean);

        Assert.Equal(0.5, mean[0, 0], 12);
        Assert.Equal(2.25, PredictiveSampleStatistics<double>.Variance(samples, mean)[0, 0], 12);
        Assert.Equal(0.0, aleatoric[0, 0], 12);
        Assert.Equal(0.0, aleatoric[0, 1], 12);
        Assert.Equal(0.0, metrics["predictive_entropy"][0], 12);
        Assert.Equal(0.0, metrics["mutual_information"][0], 12);
    }

    /// <summary>
    /// Bin m holds ((m-1)/M, m/M], the boundaries np.linspace(0, 1, M + 1) (Guo et al. 2017, Sec. 2). With
    /// M = 100, ceil(p M) - 1 misplaces 0.07, 0.14, 0.28, 0.55 and 0.56 one bin up because p M rounds above
    /// the integer (0.07 * 100 = 7.000000000000001). Each k / 100 must land alone in bin k, and 0.0700001,
    /// just above the boundary 0.07, in the next bin.
    /// </summary>
    [Fact]
    public void Ece_BoundaryProbabilities_StayInTheirRightClosedBin()
    {
        var ece = new ExpectedCalibrationError<double>(numBins: 100);
        var probs = new Vector<double>(Enumerable.Range(1, 100).Select(k => k / 100.0).ToArray());
        var ones = new Vector<int>(Enumerable.Repeat(1, 100).ToArray());

        var diagram = ece.GetReliabilityDiagram(probs, ones, ones);

        Assert.Equal(100, diagram.Count);
        for (int k = 1; k <= 100; k++)
        {
            Assert.Equal(k / 100.0, diagram[k - 1].confidence, 12);
            Assert.Equal(1, diagram[k - 1].count);
        }

        var nearBoundary = ece.GetReliabilityDiagram(
            new Vector<double>(new[] { 0.07, 0.0700001 }), new Vector<int>(new[] { 1, 1 }), new Vector<int>(new[] { 1, 0 }));
        Assert.Equal(2, nearBoundary.Count);
        Assert.Equal(0.07, nearBoundary[0].confidence, 12);
        Assert.Equal(1.0, nearBoundary[0].accuracy, 12);
        Assert.Equal(0.0700001, nearBoundary[1].confidence, 12);
        Assert.Equal(0.0, nearBoundary[1].accuracy, 12);
    }

    /// <summary>
    /// ECE over 2,200,000 samples and 1,000 bins: p = (j + 0.5) / 1000 with j = i mod 1000, all correct, so
    /// every bin holds 2,200 samples and ECE = 1 - mean(p) = 0.5. The per-bin sums must stay linear in the
    /// samples; the one-hot [samples x bins] membership had 2.2 * 10^9 entries, past int range.
    /// </summary>
    [Fact]
    public void Ece_LargeSampleManyBins_UsesLinearMemory()
    {
        const int n = 2200000;
        const int bins = 1000;
        var probs = new Vector<double>(Enumerable.Range(0, n).Select(i => (i % bins + 0.5) / bins).ToArray());
        var labels = new Vector<int>(new int[n]);

        var ece = new ExpectedCalibrationError<double>(numBins: bins);

        Assert.Equal(0.5, ece.Compute(probs, labels, labels), 9);
        var diagram = ece.GetReliabilityDiagram(probs, labels, labels);
        Assert.Equal(bins, diagram.Count);
        Assert.All(diagram, bin => Assert.Equal(n / bins, bin.count));
    }
}
