using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.Probabilistic;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for probabilistic anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class ProbabilisticAnomalyDetectionTests
{
    private const int OutlierIndex = 29; // Last row

    private static Matrix<double> CreateTestData()
    {
        int n = 30;
        var data = new double[n, 2];
        for (int i = 0; i < n - 1; i++)
        {
            data[i, 0] = 1.0 + 0.1 * (i % 5);
            data[i, 1] = 2.0 + 0.1 * (i % 7);
        }

        data[n - 1, 0] = 100.0;
        data[n - 1, 1] = 100.0;

        return new Matrix<double>(data);
    }

    private static void AssertOutlierScoresHighest(Vector<double> scores, int outlierIdx)
    {
        double outlierScore = scores[outlierIdx];
        for (int i = 0; i < scores.Length; i++)
        {
            if (i == outlierIdx) continue;
            Assert.True(outlierScore > scores[i],
                $"Outlier score ({outlierScore:F4}) at index {outlierIdx} should be higher than " +
                $"inlier score ({scores[i]:F4}) at index {i}");
        }
    }

    private static void AssertPredictClassifiesCorrectly(Vector<double> predictions, int outlierIdx)
    {
        Assert.Equal(-1.0, predictions[outlierIdx]);

        int normalCount = 0;
        int inlierCount = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            if (i == outlierIdx) continue;
            inlierCount++;
            if (predictions[i] == 1.0) normalCount++;
        }

        Assert.True(normalCount >= inlierCount * 0.8,
            $"Expected at least {inlierCount * 0.8} inliers classified as normal, got {normalCount}/{inlierCount}");
    }

    #region BayesianDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Bayesian_Construction_NotFittedByDefault()
    {
        var detector = new BayesianDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact(Timeout = 120000)]
    public async Task Bayesian_OutlierGetsHighestScore()
    {
        var detector = new BayesianDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Bayesian_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new BayesianDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GMMDetector Tests

    [Fact(Timeout = 120000)]
    public async Task GMM_OutlierGetsHighestScore()
    {
        var detector = new GMMDetector<double>(nComponents: 2);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task GMM_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GMMDetector<double>(nComponents: 2);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region COPODDetector Tests

    [Fact(Timeout = 120000)]
    public async Task COPOD_OutlierGetsHighestScore()
    {
        var detector = new COPODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task COPOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new COPODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ECODDetector Tests

    [Fact(Timeout = 120000)]
    public async Task ECOD_OutlierGetsHighestScore()
    {
        var detector = new ECODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task ECOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ECODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact(Timeout = 120000)]
    public async Task AllProbabilisticDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new BayesianDetector<double>(),
            new GMMDetector<double>(),
            new COPODDetector<double>(),
            new ECODDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion

    #region ECOD / COPOD Reference Value Tests

    // Data 1, 2, 3, 4, 10 (n = 5). F_left = #{v <= x} / 5 = .2, .4, .6, .8, 1 and F_right = #{v >= x} / 5
    // = 1, .8, .6, .4, .2, so U_left = ln 5, ln 2.5, ln(5/3), ln 1.25, 0 and U_right = 0, ln 1.25, ln(5/3),
    // ln 2.5, ln 5. Deviations from the mean 4 are -3, -2, -1, 0, 6, so m3 = 36 / 5 > 0: right-skewed, and
    // PyOD's U_skew = U_right. Scoring the training rows after Fit uses X_train and X together (PyOD
    // decision_function), which here is every value twice: the same fractions and the same skewness.
    private static Matrix<double> EcdfReferenceData()
        => new Matrix<double>(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 10 } });

    [Fact(Timeout = 120000)]
    public async Task ECOD_MatchesPyodTailAndSkewnessRule()
    {
        await Task.Yield();
        // PyOD ECOD: O = max(U_left, U_right, U_skew) per feature, summed
        var data = EcdfReferenceData();
        var detector = new ECODDetector<double>();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);

        double[] expected = { Math.Log(5), Math.Log(2.5), Math.Log(5.0 / 3.0), Math.Log(2.5), Math.Log(5) };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], scores[i], 1e-9);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task COPOD_MatchesPyodTailAndSkewnessRule()
    {
        await Task.Yield();
        // PyOD COPOD: O = max(U_skew, (U_left + U_right) / 2) per feature, summed
        var data = EcdfReferenceData();
        var detector = new COPODDetector<double>();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);

        double[] expected =
        {
            Math.Log(5) / 2, (Math.Log(2.5) + Math.Log(1.25)) / 2, Math.Log(5.0 / 3.0), Math.Log(2.5), Math.Log(5)
        };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], scores[i], 1e-9);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task COPOD_LeftSkewedFeatureUsesTheLeftTail()
    {
        await Task.Yield();
        // Mirror image -1, -2, -3, -4, -10: m3 < 0 (left-skewed), U_skew = U_left, and the max per point is
        // the mirror of the right-skewed case: ln 5, ln 2.5, ln(5/3), ln 2.5, ln 5 for -10, -4, -3, -2, -1.
        var data = new Matrix<double>(new double[,] { { -1 }, { -2 }, { -3 }, { -4 }, { -10 } });
        var detector = new COPODDetector<double>();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);

        // COPOD with U_skew = U_left: max(U_left, (U_left + U_right) / 2)
        Assert.Equal(Math.Log(5), scores[4], 1e-9);
        Assert.Equal(Math.Log(2.5), scores[3], 1e-9);
        Assert.Equal(Math.Log(5) / 2, scores[0], 1e-9);
    }

    #endregion
}
