using AiDotNet.AnomalyDetection.TreeBased;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for tree-based anomaly detection classes.
/// </summary>
public class TreeBasedAnomalyDetectionTests
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

    #region IsolationForest Tests

    [Fact]
    public void IsolationForest_Construction_DoesNotThrow()
    {
        var detector = new IsolationForest<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void IsolationForest_FitAndPredict_Works()
    {
        var detector = new IsolationForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    [Fact]
    public void IsolationForest_ScoreAnomalies_ReturnsScoresForEachRow()
    {
        var detector = new IsolationForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
    }

    #endregion

    #region ExtendedIsolationForest Tests

    [Fact]
    public void ExtendedIsolationForest_Construction_DoesNotThrow()
    {
        var detector = new ExtendedIsolationForest<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void ExtendedIsolationForest_FitAndPredict_Works()
    {
        var detector = new ExtendedIsolationForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region SCiForest Tests

    [Fact]
    public void SCiForest_Construction_DoesNotThrow()
    {
        var detector = new SCiForest<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void SCiForest_FitAndPredict_Works()
    {
        var detector = new SCiForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region FairCutForest Tests

    [Fact]
    public void FairCutForest_Construction_DoesNotThrow()
    {
        var detector = new FairCutForest<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void FairCutForest_FitAndPredict_Works()
    {
        var detector = new FairCutForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
