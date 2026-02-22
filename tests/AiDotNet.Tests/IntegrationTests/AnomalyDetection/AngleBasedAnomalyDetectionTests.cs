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
    public void ABOD_FitAndPredict_Works()
    {
        var detector = new ABODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region FastABODDetector Tests

    [Fact]
    public void FastABOD_Construction_DoesNotThrow()
    {
        var detector = new FastABODDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void FastABOD_FitAndPredict_Works()
    {
        var detector = new FastABODDetector<double>(k: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region OutlierRemovalAdapter Tests

    [Fact]
    public void OutlierRemovalAdapter_Construction_DoesNotThrow()
    {
        var detector = new ZScoreDetector<double>();
        var adapter = new OutlierRemovalAdapter<double, Matrix<double>, Vector<double>>(detector);
        Assert.NotNull(adapter);
    }

    #endregion

    #region NoOutlierRemoval Tests

    [Fact]
    public void NoOutlierRemoval_Construction_DoesNotThrow()
    {
        var noRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
        Assert.NotNull(noRemoval);
    }

    #endregion
}
