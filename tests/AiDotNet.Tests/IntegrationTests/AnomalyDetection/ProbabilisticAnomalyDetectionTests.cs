using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.Probabilistic;
using AiDotNet.LinearAlgebra;
using Xunit;

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

    [Fact]
    public void Bayesian_Construction_NotFittedByDefault()
    {
        var detector = new BayesianDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void Bayesian_OutlierGetsHighestScore()
    {
        var detector = new BayesianDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void Bayesian_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new BayesianDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GMMDetector Tests

    [Fact]
    public void GMM_OutlierGetsHighestScore()
    {
        var detector = new GMMDetector<double>(nComponents: 2);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void GMM_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GMMDetector<double>(nComponents: 2);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region COPODDetector Tests

    [Fact]
    public void COPOD_OutlierGetsHighestScore()
    {
        var detector = new COPODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void COPOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new COPODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ECODDetector Tests

    [Fact]
    public void ECOD_OutlierGetsHighestScore()
    {
        var detector = new ECODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void ECOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ECODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllProbabilisticDetectors_PredictBeforeFit_Throws()
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
}
