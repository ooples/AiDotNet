using AiDotNet.AnomalyDetection.NeuralNetwork;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for neural network anomaly detection classes.
/// </summary>
public class NeuralNetworkAnomalyDetectionTests
{
    private static Matrix<double> CreateTestData()
    {
        int n = 30;
        var data = new double[n, 4];
        for (int i = 0; i < n - 1; i++)
        {
            data[i, 0] = 1.0 + 0.1 * (i % 5);
            data[i, 1] = 2.0 + 0.1 * (i % 7);
            data[i, 2] = 0.5 + 0.05 * (i % 3);
            data[i, 3] = 1.5 + 0.02 * i;
        }

        data[n - 1, 0] = 100.0;
        data[n - 1, 1] = 100.0;
        data[n - 1, 2] = 100.0;
        data[n - 1, 3] = 100.0;

        return new Matrix<double>(data);
    }

    #region AutoencoderDetector Tests

    [Fact]
    public void Autoencoder_Construction_SetsDefaults()
    {
        var detector = new AutoencoderDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void Autoencoder_FitAndPredict_ProducesValidPredictions()
    {
        var detector = new AutoencoderDetector<double>(epochs: 5);
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // Every prediction should be either -1 (anomaly) or 1 (inlier)
        for (int i = 0; i < predictions.Length; i++)
            Assert.True(predictions[i] == -1.0 || predictions[i] == 1.0,
                $"Prediction at index {i} should be -1 or 1, got {predictions[i]}");

        // Scores should differentiate outlier from inlier
        var scores = detector.ScoreAnomalies(data);
        Assert.NotEqual(scores[0], scores[data.Rows - 1]);
    }

    #endregion

    #region VAEDetector Tests

    [Fact]
    public void VAE_Construction_DoesNotThrow()
    {
        var detector = new VAEDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void VAE_FitAndPredict_Works()
    {
        var detector = new VAEDetector<double>(epochs: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region DAGMMDetector Tests

    [Fact]
    public void DAGMM_Construction_DoesNotThrow()
    {
        var detector = new DAGMMDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void DAGMM_FitAndPredict_Works()
    {
        var detector = new DAGMMDetector<double>(epochs: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region DeepSVDDDetector Tests

    [Fact]
    public void DeepSVDD_Construction_DoesNotThrow()
    {
        var detector = new DeepSVDDDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void DeepSVDD_FitAndPredict_Works()
    {
        var detector = new DeepSVDDDetector<double>(epochs: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region DevNetDetector Tests

    [Fact]
    public void DevNet_Construction_DoesNotThrow()
    {
        var detector = new DevNetDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void DevNet_FitAndPredict_Works()
    {
        var detector = new DevNetDetector<double>(epochs: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region GANomalyDetector Tests

    [Fact]
    public void GAnomaly_Construction_DoesNotThrow()
    {
        var detector = new GANomalyDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void GAnomaly_FitAndPredict_Works()
    {
        var detector = new GANomalyDetector<double>(epochs: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region AnoGANDetector Tests

    [Fact]
    public void AnoGAN_Construction_DoesNotThrow()
    {
        var detector = new AnoGANDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void AnoGAN_FitAndPredict_Works()
    {
        var detector = new AnoGANDetector<double>(epochs: 5);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
