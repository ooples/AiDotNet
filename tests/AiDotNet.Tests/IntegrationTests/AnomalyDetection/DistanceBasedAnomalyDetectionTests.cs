using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.DistanceBased;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for distance-based anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class DistanceBasedAnomalyDetectionTests
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

    #region LocalOutlierFactor Tests

    [Fact]
    public void LOF_Construction_NotFittedByDefault()
    {
        var detector = new LocalOutlierFactor<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void LOF_OutlierGetsHighestScore()
    {
        var detector = new LocalOutlierFactor<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void LOF_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LocalOutlierFactor<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region KNNDetector Tests

    [Fact]
    public void KNN_OutlierGetsHighestScore()
    {
        var detector = new KNNDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void KNN_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new KNNDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region COFDetector Tests

    [Fact]
    public void COF_OutlierGetsHighestScore()
    {
        var detector = new COFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void COF_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new COFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region INFLODetector Tests

    [Fact]
    public void INFLO_OutlierGetsHighestScore()
    {
        var detector = new INFLODetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void INFLO_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new INFLODetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region LoOPDetector Tests

    [Fact]
    public void LoOP_OutlierGetsHighestScore()
    {
        var detector = new LoOPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void LoOP_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LoOPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region LOCIDetector Tests

    [Fact]
    public void LOCI_OutlierGetsHighestScore()
    {
        var detector = new LOCIDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void LOCI_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LOCIDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region LDCOFDetector Tests

    [Fact]
    public void LDCOF_OutlierGetsHighestScore()
    {
        var detector = new LDCOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void LDCOF_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LDCOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region OCSVMDetector Tests

    [Fact]
    public void OCSVM_OutlierGetsHighestScore()
    {
        var detector = new OCSVMDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void OCSVM_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new OCSVMDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region SOSDetector Tests

    [Fact]
    public void SOS_OutlierGetsHighestScore()
    {
        var detector = new SOSDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void SOS_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new SOSDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllDistanceDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new LocalOutlierFactor<double>(),
            new KNNDetector<double>(),
            new COFDetector<double>(),
            new INFLODetector<double>(),
            new LoOPDetector<double>(),
            new LOCIDetector<double>(),
            new LDCOFDetector<double>(),
            new OCSVMDetector<double>(),
            new SOSDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
