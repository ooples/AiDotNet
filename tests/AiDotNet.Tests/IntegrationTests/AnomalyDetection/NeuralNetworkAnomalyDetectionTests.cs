using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.NeuralNetwork;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for neural network anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class NeuralNetworkAnomalyDetectionTests
{
    private const int OutlierIndex = 29; // Last row

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

    #region AutoencoderDetector Tests

    [Fact]
    public void Autoencoder_Construction_NotFittedByDefault()
    {
        var detector = new AutoencoderDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void Autoencoder_OutlierGetsHighestScore()
    {
        var detector = new AutoencoderDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void Autoencoder_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new AutoencoderDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region VAEDetector Tests

    [Fact]
    public void VAE_OutlierGetsHighestScore()
    {
        var detector = new VAEDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void VAE_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new VAEDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DAGMMDetector Tests

    [Fact]
    public void DAGMM_OutlierGetsHighestScore()
    {
        var detector = new DAGMMDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void DAGMM_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DAGMMDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DeepSVDDDetector Tests

    [Fact]
    public void DeepSVDD_OutlierGetsHighestScore()
    {
        var detector = new DeepSVDDDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void DeepSVDD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DeepSVDDDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DevNetDetector Tests

    [Fact]
    public void DevNet_OutlierGetsHighestScore()
    {
        var detector = new DevNetDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void DevNet_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DevNetDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GANomalyDetector Tests

    [Fact]
    public void GAnomaly_OutlierGetsHighestScore()
    {
        var detector = new GANomalyDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void GAnomaly_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GANomalyDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region AnoGANDetector Tests

    [Fact]
    public void AnoGAN_OutlierGetsHighestScore()
    {
        var detector = new AnoGANDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact]
    public void AnoGAN_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new AnoGANDetector<double>(epochs: 10);
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllNeuralNetworkDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new AutoencoderDetector<double>(),
            new VAEDetector<double>(),
            new DAGMMDetector<double>(),
            new DeepSVDDDetector<double>(),
            new DevNetDetector<double>(),
            new GANomalyDetector<double>(),
            new AnoGANDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
