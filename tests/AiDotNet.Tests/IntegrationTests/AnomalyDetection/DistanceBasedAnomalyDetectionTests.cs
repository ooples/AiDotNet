using AiDotNet.AnomalyDetection.DistanceBased;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for distance-based anomaly detection classes.
/// </summary>
public class DistanceBasedAnomalyDetectionTests
{
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

    #region LocalOutlierFactor Tests

    [Fact]
    public void LOF_Construction_DoesNotThrow()
    {
        var detector = new LocalOutlierFactor<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void LOF_FitAndPredict_DetectsOutlier()
    {
        var detector = new LocalOutlierFactor<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // LOF should detect (100,100) as outlier (-1) vs cluster at (1,2)
        Assert.Equal(-1.0, predictions[data.Rows - 1]);

        // Verify anomaly scores differ for outlier vs inlier
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        Assert.NotEqual(scores[0], scores[data.Rows - 1]);
    }

    #endregion

    #region KNNDetector Tests

    [Fact]
    public void KNN_Construction_DoesNotThrow()
    {
        var detector = new KNNDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void KNN_FitAndPredict_Works()
    {
        var detector = new KNNDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region COFDetector Tests

    [Fact]
    public void COF_Construction_DoesNotThrow()
    {
        var detector = new COFDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void COF_FitAndPredict_Works()
    {
        var detector = new COFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region INFLODetector Tests

    [Fact]
    public void INFLO_Construction_DoesNotThrow()
    {
        var detector = new INFLODetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void INFLO_FitAndPredict_Works()
    {
        var detector = new INFLODetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region LoOPDetector Tests

    [Fact]
    public void LoOP_Construction_DoesNotThrow()
    {
        var detector = new LoOPDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void LoOP_FitAndPredict_Works()
    {
        var detector = new LoOPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region LOCIDetector Tests

    [Fact]
    public void LOCI_Construction_DoesNotThrow()
    {
        var detector = new LOCIDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void LOCI_FitAndPredict_Works()
    {
        var detector = new LOCIDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region LDCOFDetector Tests

    [Fact]
    public void LDCOF_Construction_DoesNotThrow()
    {
        var detector = new LDCOFDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void LDCOF_FitAndPredict_Works()
    {
        var detector = new LDCOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region OCSVMDetector Tests

    [Fact]
    public void OCSVM_Construction_DoesNotThrow()
    {
        var detector = new OCSVMDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void OCSVM_FitAndPredict_Works()
    {
        var detector = new OCSVMDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region SOSDetector Tests

    [Fact]
    public void SOS_Construction_DoesNotThrow()
    {
        var detector = new SOSDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void SOS_FitAndPredict_Works()
    {
        var detector = new SOSDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
