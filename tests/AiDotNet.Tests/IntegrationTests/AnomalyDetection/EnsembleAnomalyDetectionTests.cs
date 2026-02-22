using AiDotNet.AnomalyDetection.Ensemble;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for ensemble anomaly detection classes.
/// </summary>
public class EnsembleAnomalyDetectionTests
{
    private static Matrix<double> CreateTestData()
    {
        int n = 30;
        var data = new double[n, 3];
        for (int i = 0; i < n - 1; i++)
        {
            data[i, 0] = 1.0 + 0.1 * (i % 5);
            data[i, 1] = 2.0 + 0.1 * (i % 7);
            data[i, 2] = 0.5 + 0.05 * (i % 3);
        }

        data[n - 1, 0] = 100.0;
        data[n - 1, 1] = 100.0;
        data[n - 1, 2] = 100.0;

        return new Matrix<double>(data);
    }

    #region AveragingDetector Tests

    [Fact]
    public void Averaging_Construction_DoesNotThrow()
    {
        var detector = new AveragingDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void Averaging_FitAndPredict_DetectsOutlier()
    {
        var detector = new AveragingDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // Ensemble should detect the extreme outlier at (100,100,100)
        Assert.Equal(-1.0, predictions[data.Rows - 1]);

        // Verify scores differentiate outlier from inlier
        var scores = detector.ScoreAnomalies(data);
        Assert.NotEqual(scores[0], scores[data.Rows - 1]);
    }

    #endregion

    #region MaximumDetector Tests

    [Fact]
    public void Maximum_Construction_DoesNotThrow()
    {
        var detector = new MaximumDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void Maximum_FitAndPredict_Works()
    {
        var detector = new MaximumDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region FeatureBaggingDetector Tests

    [Fact]
    public void FeatureBagging_Construction_DoesNotThrow()
    {
        var detector = new FeatureBaggingDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void FeatureBagging_FitAndPredict_Works()
    {
        var detector = new FeatureBaggingDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region RandomSubspaceDetector Tests

    [Fact]
    public void RandomSubspace_Construction_DoesNotThrow()
    {
        var detector = new RandomSubspaceDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void RandomSubspace_FitAndPredict_Works()
    {
        var detector = new RandomSubspaceDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region LSCPDetector Tests

    [Fact]
    public void LSCP_Construction_DoesNotThrow()
    {
        var detector = new LSCPDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void LSCP_FitAndPredict_Works()
    {
        var detector = new LSCPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region SUODDetector Tests

    [Fact]
    public void SUOD_Construction_DoesNotThrow()
    {
        var detector = new SUODDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void SUOD_FitAndPredict_Works()
    {
        var detector = new SUODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region XGBODDetector Tests

    [Fact]
    public void XGBOD_Construction_DoesNotThrow()
    {
        var detector = new XGBODDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void XGBOD_FitAndPredict_Works()
    {
        var detector = new XGBODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
