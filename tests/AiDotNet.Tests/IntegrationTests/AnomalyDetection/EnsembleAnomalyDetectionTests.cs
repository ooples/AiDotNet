using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.Ensemble;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for ensemble anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class EnsembleAnomalyDetectionTests
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

    private static Matrix<double> CreateNormalTrainingData()
    {
        var data = new double[30, 3];
        for (int i = 0; i < data.GetLength(0); i++)
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

    #region AveragingDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Averaging_Construction_NotFittedByDefault()
    {
        var detector = new AveragingDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact(Timeout = 120000)]
    public async Task Averaging_OutlierGetsHighestScore()
    {
        var detector = new AveragingDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Averaging_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new AveragingDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region MaximumDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Maximum_OutlierGetsHighestScore()
    {
        var detector = new MaximumDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Maximum_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new MaximumDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Maximum_ScoreDoesNotDependOnQueryBatchComposition()
    {
        await Task.Yield();
        var detector = new MaximumDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);

        var one = new Matrix<double>(1, data.Columns);
        var mixed = new Matrix<double>(2, data.Columns);
        for (int column = 0; column < data.Columns; column++)
        {
            one[0, column] = data[OutlierIndex, column];
            mixed[0, column] = data[OutlierIndex, column];
            mixed[1, column] = data[0, column];
        }

        double isolatedScore = detector.ScoreAnomalies(one)[0];
        double mixedScore = detector.ScoreAnomalies(mixed)[0];

        Assert.Equal(isolatedScore, mixedScore, precision: 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Maximum_SeparateSingleRowBatchesUseTrainingScale()
    {
        var detector = new MaximumDetector<double>();
        detector.Fit(CreateNormalTrainingData());

        var normalScore = detector.ScoreAnomalies(
            new Matrix<double>(new[,] { { 1.2, 2.1, 0.55 } }))[0];
        var outlierScore = detector.ScoreAnomalies(
            new Matrix<double>(new[,] { { 100.0, 100.0, 100.0 } }))[0];

        Assert.True(outlierScore > normalScore,
            $"Separately scored outlier ({outlierScore:F4}) should exceed normal ({normalScore:F4}).");
    }

    #endregion

    #region FeatureBaggingDetector Tests

    [Fact(Timeout = 120000)]
    public async Task FeatureBagging_OutlierGetsHighestScore()
    {
        var detector = new FeatureBaggingDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task FeatureBagging_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new FeatureBaggingDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region RandomSubspaceDetector Tests

    [Fact(Timeout = 120000)]
    public async Task RandomSubspace_OutlierGetsHighestScore()
    {
        var detector = new RandomSubspaceDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task RandomSubspace_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new RandomSubspaceDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region LSCPDetector Tests

    [Fact(Timeout = 120000)]
    public async Task LSCP_OutlierGetsHighestScore()
    {
        var detector = new LSCPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task LSCP_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LSCPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region SUODDetector Tests

    [Fact(Timeout = 120000)]
    public async Task SUOD_OutlierGetsHighestScore()
    {
        var detector = new SUODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task SUOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new SUODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task SUOD_SeparateSingleRowBatchesUseTrainingScale()
    {
        var detector = new SUODDetector<double>();
        detector.Fit(CreateNormalTrainingData());

        var normalScore = detector.ScoreAnomalies(
            new Matrix<double>(new[,] { { 1.2, 2.1, 0.55 } }))[0];
        var outlierScore = detector.ScoreAnomalies(
            new Matrix<double>(new[,] { { 100.0, 100.0, 100.0 } }))[0];

        Assert.True(outlierScore > normalScore,
            $"Separately scored outlier ({outlierScore:F4}) should exceed normal ({normalScore:F4}).");
    }

    [Fact(Timeout = 120000)]
    public async Task SUOD_NoProjectionFitHasResolvedParameterLayout()
    {
        var detector = new SUODDetector<double>(useRandomProjection: true, nProjectedFeatures: 10);
        detector.Fit(CreateNormalTrainingData());

        Assert.NotEmpty(detector.GetParameters());
        Assert.Equal(detector.ParameterCount, detector.GetParameters().Length);
    }

    #endregion

    #region XGBODDetector Tests

    [Fact(Timeout = 120000)]
    public async Task XGBOD_OutlierGetsHighestScore()
    {
        var detector = new XGBODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task XGBOD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new XGBODDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact(Timeout = 120000)]
    public async Task AllEnsembleDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new AveragingDetector<double>(),
            new MaximumDetector<double>(),
            new FeatureBaggingDetector<double>(),
            new RandomSubspaceDetector<double>(),
            new LSCPDetector<double>(),
            new SUODDetector<double>(),
            new XGBODDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
