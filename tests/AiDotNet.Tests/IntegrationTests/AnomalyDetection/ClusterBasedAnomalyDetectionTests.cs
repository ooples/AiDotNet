using AiDotNet.AnomalyDetection.ClusterBased;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for cluster-based anomaly detection classes.
/// </summary>
public class ClusterBasedAnomalyDetectionTests
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

    #region KMeansDetector Tests

    [Fact]
    public void KMeans_Construction_DoesNotThrow()
    {
        var detector = new KMeansDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void KMeans_FitAndPredict_Works()
    {
        var detector = new KMeansDetector<double>(k: 3);
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region DBSCANDetector Tests

    [Fact]
    public void DBSCAN_Construction_DoesNotThrow()
    {
        var detector = new DBSCANDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void DBSCAN_FitAndPredict_Works()
    {
        var detector = new DBSCANDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region HDBSCANDetector Tests

    [Fact]
    public void HDBSCAN_Construction_DoesNotThrow()
    {
        var detector = new HDBSCANDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void HDBSCAN_FitAndPredict_Works()
    {
        var detector = new HDBSCANDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region CBLOFDetector Tests

    [Fact]
    public void CBLOF_Construction_DoesNotThrow()
    {
        var detector = new CBLOFDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void CBLOF_FitAndPredict_Works()
    {
        var detector = new CBLOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
