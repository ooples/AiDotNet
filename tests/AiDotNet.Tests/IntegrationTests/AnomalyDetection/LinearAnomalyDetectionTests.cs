using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.Linear;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for linear anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class LinearAnomalyDetectionTests
{
    private const int OutlierIndex = 29; // Last row

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

    // Inlier-only training set (the 29 normal rows of CreateTestData, without the
    // outlier at row 29). Kernel PCA novelty detection (Hoffmann 2007) assumes an
    // anomaly-free training set: the detector learns the normal manifold and then
    // flags points that cannot be reconstructed from it. Fitting on the contaminated
    // set instead lets the lone outlier dominate its own principal component and be
    // reconstructed with ~zero residual — outside the method's scope.
    private static Matrix<double> CreateCleanTrainingData()
    {
        int n = 29;
        var data = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = 1.0 + 0.1 * (i % 5);
            data[i, 1] = 2.0 + 0.1 * (i % 7);
            data[i, 2] = 0.5 + 0.05 * (i % 3);
        }
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

    #region PCADetector Tests

    [Fact(Timeout = 120000)]
    public async Task PCA_Construction_NotFittedByDefault()
    {
        var detector = new PCADetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact(Timeout = 120000)]
    public async Task PCA_OutlierGetsHighestScore()
    {
        var detector = new PCADetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task PCA_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new PCADetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region OneClassSVM Tests

    [Fact(Timeout = 120000)]
    public async Task OneClassSVM_OutlierGetsHighestScore()
    {
        var detector = new OneClassSVM<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task OneClassSVM_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new OneClassSVM<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region EllipticEnvelopeDetector Tests

    [Fact(Timeout = 120000)]
    public async Task EllipticEnvelope_OutlierGetsHighestScore()
    {
        var detector = new EllipticEnvelopeDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task EllipticEnvelope_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new EllipticEnvelopeDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region MCDDetector Tests

    [Fact(Timeout = 120000)]
    public async Task MCD_OutlierGetsHighestScore()
    {
        var detector = new MCDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task MCD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new MCDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region KernelPCADetector Tests

    [Fact(Timeout = 120000)]
    public async Task KernelPCA_OutlierGetsHighestScore()
    {
        var detector = new KernelPCADetector<double>();
        // Novelty-detection protocol (Hoffmann 2007): fit on the anomaly-free inliers,
        // then score a set that includes the held-out outlier as an unseen point.
        detector.Fit(CreateCleanTrainingData());
        var data = CreateTestData();
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task KernelPCA_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new KernelPCADetector<double>();
        detector.Fit(CreateCleanTrainingData());
        var data = CreateTestData();
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region RobustPCADetector Tests

    [Fact(Timeout = 120000)]
    public async Task RobustPCA_OutlierGetsHighestScore()
    {
        var detector = new RobustPCADetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task RobustPCA_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new RobustPCADetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact(Timeout = 120000)]
    public async Task AllLinearDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new PCADetector<double>(),
            new OneClassSVM<double>(),
            new EllipticEnvelopeDetector<double>(),
            new MCDDetector<double>(),
            new KernelPCADetector<double>(),
            new RobustPCADetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
