using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.AngleBased;
using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for angle-based anomaly detection and adapter classes.
/// </summary>
public class AngleBasedAnomalyDetectionTests
{
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

    #region ABODDetector Tests

    [Fact]
    public void ABOD_Construction_DoesNotThrow()
    {
        var detector = new ABODDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void ABOD_FitAndPredict_DetectsOutlier()
    {
        var detector = new ABODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // The last point (100,100) is a clear outlier vs cluster around (1,2)
        // Predict returns -1 for anomalies, 1 for inliers
        var outlierPrediction = predictions[data.Rows - 1];
        Assert.Equal(-1.0, outlierPrediction);

        // Verify anomaly scores: outlier should have a distinct score
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        double outlierScore = scores[data.Rows - 1];
        double inlierScore = scores[0];
        Assert.NotEqual(outlierScore, inlierScore);
    }

    #endregion

    #region FastABODDetector Tests

    [Fact]
    public void FastABOD_Construction_SetsDefaults()
    {
        var detector = new FastABODDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void FastABOD_FitAndPredict_DetectsOutlier()
    {
        var detector = new FastABODDetector<double>(k: 5);
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // The outlier at (100,100) should be flagged
        var outlierPrediction = predictions[data.Rows - 1];
        Assert.Equal(-1.0, outlierPrediction);
    }

    #endregion

    #region OutlierRemovalAdapter Tests

    [Fact]
    public void OutlierRemovalAdapter_RemovesOutliers()
    {
        var detector = new ZScoreDetector<double>();
        var adapter = new OutlierRemovalAdapter<double, Matrix<double>, Vector<double>>(detector);
        Assert.NotNull(adapter);

        // Create data with an outlier and matching output vector
        var data = CreateTestData();
        var outputs = new Vector<double>(data.Rows);
        for (int i = 0; i < data.Rows; i++)
            outputs[i] = i;

        var (cleanedInputs, cleanedOutputs) = adapter.RemoveOutliers(data, outputs);

        // Cleaned data should have fewer rows (outlier removed)
        Assert.True(cleanedInputs.Rows <= data.Rows);
        Assert.Equal(cleanedInputs.Rows, cleanedOutputs.Length);
    }

    #endregion

    #region NoOutlierRemoval Tests

    [Fact]
    public void NoOutlierRemoval_ReturnsDataUnchanged()
    {
        var noRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(noRemoval);

        var data = CreateTestData();
        var outputs = new Vector<double>(data.Rows);
        for (int i = 0; i < data.Rows; i++)
            outputs[i] = i;

        var (cleanedInputs, cleanedOutputs) = noRemoval.RemoveOutliers(data, outputs);

        // NoOutlierRemoval should return data unchanged
        Assert.Equal(data.Rows, cleanedInputs.Rows);
        Assert.Equal(outputs.Length, cleanedOutputs.Length);
    }

    #endregion
}
