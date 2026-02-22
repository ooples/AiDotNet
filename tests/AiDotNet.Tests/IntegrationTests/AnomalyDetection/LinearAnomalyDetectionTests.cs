using AiDotNet.AnomalyDetection.Linear;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for linear anomaly detection classes.
/// </summary>
public class LinearAnomalyDetectionTests
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

    #region PCADetector Tests

    [Fact]
    public void PCA_Construction_DoesNotThrow()
    {
        var detector = new PCADetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void PCA_FitAndPredict_Works()
    {
        var detector = new PCADetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region OneClassSVM Tests

    [Fact]
    public void OneClassSVM_Construction_DoesNotThrow()
    {
        var detector = new OneClassSVM<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void OneClassSVM_FitAndPredict_Works()
    {
        var detector = new OneClassSVM<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region EllipticEnvelopeDetector Tests

    [Fact]
    public void EllipticEnvelope_Construction_DoesNotThrow()
    {
        var detector = new EllipticEnvelopeDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void EllipticEnvelope_FitAndPredict_Works()
    {
        var detector = new EllipticEnvelopeDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region MCDDetector Tests

    [Fact]
    public void MCD_Construction_DoesNotThrow()
    {
        var detector = new MCDDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void MCD_FitAndPredict_Works()
    {
        var detector = new MCDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region KernelPCADetector Tests

    [Fact]
    public void KernelPCA_Construction_DoesNotThrow()
    {
        var detector = new KernelPCADetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void KernelPCA_FitAndPredict_Works()
    {
        var detector = new KernelPCADetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region RobustPCADetector Tests

    [Fact]
    public void RobustPCA_Construction_DoesNotThrow()
    {
        var detector = new RobustPCADetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void RobustPCA_FitAndPredict_Works()
    {
        var detector = new RobustPCADetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
