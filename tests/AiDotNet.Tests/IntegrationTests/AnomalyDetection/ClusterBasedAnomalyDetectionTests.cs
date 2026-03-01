using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.ClusterBased;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for cluster-based anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class ClusterBasedAnomalyDetectionTests
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

    #region KMeansDetector Tests

    [Fact]
    public void KMeans_Construction_NotFittedByDefault()
    {
        var detector = new KMeansDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void KMeans_OutlierGetsHighestScore()
    {
        var detector = new KMeansDetector<double>(k: 3);
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void KMeans_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new KMeansDetector<double>(k: 3);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DBSCANDetector Tests

    [Fact]
    public void DBSCAN_OutlierGetsHighestScore()
    {
        var detector = new DBSCANDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void DBSCAN_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DBSCANDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region HDBSCANDetector Tests

    [Fact]
    public void HDBSCAN_OutlierGetsHighestScore()
    {
        var detector = new HDBSCANDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void HDBSCAN_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new HDBSCANDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region CBLOFDetector Tests

    [Fact]
    public void CBLOF_OutlierGetsHighestScore()
    {
        var detector = new CBLOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void CBLOF_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new CBLOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllClusterDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new KMeansDetector<double>(),
            new DBSCANDetector<double>(),
            new HDBSCANDetector<double>(),
            new CBLOFDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
