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

    #region LSCP Reference Tests (Zhao et al. 2019, PyOD lscp.py)

    [Fact(Timeout = 120000)]
    public async Task LSCP_CompetentDetectorsAreTheMostPopulatedHistogramBin()
    {
        // Row 1: correlations .1, .2, .25, .9 with min(10, 4) = 4 bins over [.1, .9]: edges .1, .3, .5, .7, .9;
        // counts 3, 0, 0, 1, so the first bin wins and the detectors with .1 <= r <= .3 are competent.
        // Row 2: .5, .5 is flat, so the range is [0, 1] with 2 bins (edges 0, .5, 1); both fall in the last
        // bin (numpy closes the last bin) and both are competent.
        // Row 3: 0, 1 with edges 0, .5, 1: one per bin, the tie goes to the last bin, only r = 1 is competent.
        var detector = new LSCPDetector<double>(nEstimators: 4);
        var correlations = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(
            new[] { 0.1, 0.2, 0.25, 0.9 }, new[] { 1, 4 });
        Assert.Equal(new[] { 1.0, 1.0, 1.0, 0.0 }, detector.CompetentDetectors(correlations).ToArray());

        var flat = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 0.5, 0.5, 0.0, 1.0 }, new[] { 2, 2 });
        Assert.Equal(new[] { 1.0, 1.0, 0.0, 1.0 }, detector.CompetentDetectors(flat).ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task LSCP_LocalRegionIsTheConsistentNearestNeighbours()
    {
        // One feature, so every one of the 20 iterations uses all features and finds the same 3 nearest
        // training rows; each is counted 20 > 10 times and forms the region. Query .1 against 0..9: rows 0,
        // 1, 2. Query .5: rows 0 and 1 tie at .5, then row 2 at 1.5: rows 0, 1, 2. Query 8.8: rows 9, 8, 7.
        var training = new Matrix<double>(10, 1);
        for (int i = 0; i < 10; i++)
        {
            training[i, 0] = i;
        }

        var query = new Matrix<double>(new double[,] { { 0.1 }, { 0.5 }, { 8.8 } });
        var detector = new LSCPDetector<double>(localRegionSize: 3);
        var region = detector.LocalRegions(query, training).ToArray();

        var expected = new double[30];
        foreach (var (row, column) in new[] { (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 7), (2, 8), (2, 9) })
        {
            expected[row * 10 + column] = 1.0;
        }

        Assert.Equal(expected, region);
    }

    #endregion
}
