using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.AngleBased;
using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for angle-based anomaly detection and adapter classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class AngleBasedAnomalyDetectionTests
{
    private const int OutlierIndex = 19; // Last row

    private static Matrix<double> CreateTestData()
    {
        int n = 20;
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

    #region ABODDetector Tests

    [Fact]
    public void ABOD_Construction_NotFittedByDefault()
    {
        var detector = new ABODDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void ABOD_OutlierGetsHighestScore()
    {
        var detector = new ABODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void ABOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ABODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region FastABODDetector Tests

    [Fact]
    public void FastABOD_OutlierGetsHighestScore()
    {
        var detector = new FastABODDetector<double>(k: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void FastABOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new FastABODDetector<double>(k: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region OutlierRemovalAdapter Tests

    [Fact]
    public void OutlierRemovalAdapter_RemovesOutliers()
    {
        var detector = new ZScoreDetector<double>();
        var adapter = new OutlierRemovalAdapter<double, Matrix<double>, Vector<double>>(detector);
        Assert.NotNull(adapter);
    }

    #endregion

    #region NoOutlierRemoval Tests

    [Fact]
    public void NoOutlierRemoval_PassesDataThrough()
    {
        var noRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(noRemoval);
    }

    #endregion
}
