using AiDotNet.AnomalyDetection.Probabilistic;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for probabilistic anomaly detection classes.
/// </summary>
public class ProbabilisticAnomalyDetectionTests
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

    #region BayesianDetector Tests

    [Fact]
    public void Bayesian_Construction_DoesNotThrow()
    {
        var detector = new BayesianDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void Bayesian_FitAndPredict_DetectsOutlier()
    {
        var detector = new BayesianDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // Outlier at (100,100) should be detected
        Assert.Equal(-1.0, predictions[data.Rows - 1]);

        // Scores should differentiate outlier
        var scores = detector.ScoreAnomalies(data);
        Assert.NotEqual(scores[0], scores[data.Rows - 1]);
    }

    #endregion

    #region GMMDetector Tests

    [Fact]
    public void GMM_Construction_DoesNotThrow()
    {
        var detector = new GMMDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void GMM_FitAndPredict_Works()
    {
        var detector = new GMMDetector<double>(nComponents: 2);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region COPODDetector Tests

    [Fact]
    public void COPOD_Construction_DoesNotThrow()
    {
        var detector = new COPODDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void COPOD_FitAndPredict_Works()
    {
        var detector = new COPODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region ECODDetector Tests

    [Fact]
    public void ECOD_Construction_DoesNotThrow()
    {
        var detector = new ECODDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void ECOD_FitAndPredict_Works()
    {
        var detector = new ECODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
