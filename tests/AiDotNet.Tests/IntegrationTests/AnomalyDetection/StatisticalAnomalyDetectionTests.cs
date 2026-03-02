using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for statistical anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class StatisticalAnomalyDetectionTests
{
    private const int OutlierIndex = 19; // Last row

    /// <summary>
    /// Creates a 20x2 matrix with normal inliers near (1,2) and one clear outlier at row 19 = (100,100).
    /// </summary>
    private static Matrix<double> CreateTestData()
    {
        int n = 20;
        var data = new double[n, 2];
        for (int i = 0; i < n - 1; i++)
        {
            data[i, 0] = 1.0 + 0.1 * (i % 5);
            data[i, 1] = 2.0 + 0.1 * (i % 7);
        }

        // Outlier far from the cluster
        data[n - 1, 0] = 100.0;
        data[n - 1, 1] = 100.0;

        return new Matrix<double>(data);
    }

    /// <summary>
    /// Asserts the outlier gets a higher anomaly score than all inliers.
    /// </summary>
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

    /// <summary>
    /// Asserts Predict flags the outlier as anomaly (-1) and most inliers as normal (1).
    /// </summary>
    private static void AssertPredictClassifiesCorrectly(Vector<double> predictions, int outlierIdx)
    {
        // Outlier must be classified as anomaly (-1)
        Assert.Equal(-1.0, predictions[outlierIdx]);

        // At least 80% of inliers should be classified as normal (1)
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

    #region ZScoreDetector Tests

    [Fact]
    public void ZScore_Construction_NotFittedByDefault()
    {
        var detector = new ZScoreDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void ZScore_OutlierGetsHighestScore()
    {
        var detector = new ZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void ZScore_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ModifiedZScoreDetector Tests

    [Fact]
    public void ModifiedZScore_OutlierGetsHighestScore()
    {
        var detector = new ModifiedZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void ModifiedZScore_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ModifiedZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region IQRDetector Tests

    [Fact]
    public void IQR_OutlierGetsHighestScore()
    {
        var detector = new IQRDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void IQR_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new IQRDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region MADDetector Tests

    [Fact]
    public void MAD_OutlierGetsHighestScore()
    {
        var detector = new MADDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void MAD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new MADDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region HampelDetector Tests

    [Fact]
    public void Hampel_OutlierGetsHighestScore()
    {
        var detector = new HampelDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void Hampel_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new HampelDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GrubbsTestDetector Tests

    [Fact]
    public void Grubbs_OutlierGetsHighestScore()
    {
        var detector = new GrubbsTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void Grubbs_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GrubbsTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ESDDetector Tests

    [Fact]
    public void ESD_OutlierGetsHighestScore()
    {
        var detector = new ESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void ESD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GESDDetector Tests

    [Fact]
    public void GESD_OutlierGetsHighestScore()
    {
        var detector = new GESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void GESD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ChiSquareDetector Tests

    [Fact]
    public void ChiSquare_OutlierGetsHighestScore()
    {
        var detector = new ChiSquareDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void ChiSquare_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ChiSquareDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DixonQTestDetector Tests

    [Fact]
    public void DixonQ_OutlierGetsHighestScore()
    {
        var detector = new DixonQTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void DixonQ_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DixonQTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region PercentileDetector Tests

    [Fact]
    public void Percentile_OutlierGetsHighestScore()
    {
        var detector = new PercentileDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void Percentile_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new PercentileDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllStatisticalDetectors_NotFittedByDefault()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new ZScoreDetector<double>(),
            new ModifiedZScoreDetector<double>(),
            new IQRDetector<double>(),
            new MADDetector<double>(),
            new HampelDetector<double>(),
            new GrubbsTestDetector<double>(),
            new ESDDetector<double>(),
            new GESDDetector<double>(),
            new ChiSquareDetector<double>(),
            new DixonQTestDetector<double>(),
            new PercentileDetector<double>(),
        };

        foreach (var detector in detectors)
        {
            Assert.False(detector.IsFitted);
        }
    }

    [Fact]
    public void AllStatisticalDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new ZScoreDetector<double>(),
            new ModifiedZScoreDetector<double>(),
            new IQRDetector<double>(),
            new MADDetector<double>(),
            new HampelDetector<double>(),
            new GrubbsTestDetector<double>(),
            new ESDDetector<double>(),
            new GESDDetector<double>(),
            new ChiSquareDetector<double>(),
            new DixonQTestDetector<double>(),
            new PercentileDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    [Fact]
    public void AllStatisticalDetectors_ScoreAnomaliesBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new ZScoreDetector<double>(),
            new ModifiedZScoreDetector<double>(),
            new IQRDetector<double>(),
            new MADDetector<double>(),
            new HampelDetector<double>(),
            new GrubbsTestDetector<double>(),
            new ESDDetector<double>(),
            new GESDDetector<double>(),
            new ChiSquareDetector<double>(),
            new DixonQTestDetector<double>(),
            new PercentileDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.ScoreAnomalies(data));
        }
    }

    #endregion
}
