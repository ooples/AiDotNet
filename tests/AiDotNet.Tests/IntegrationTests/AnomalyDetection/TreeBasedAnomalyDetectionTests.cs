using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.TreeBased;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for tree-based anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class TreeBasedAnomalyDetectionTests
{
    private const int OutlierIndex = 29; // Last row

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

    #region IsolationForest Tests

    [Fact(Timeout = 120000)]
    public async Task IsolationForest_Construction_NotFittedByDefault()
    {
        var detector = new IsolationForest<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact(Timeout = 120000)]
    public async Task IsolationForest_OutlierGetsHighestScore()
    {
        var detector = new IsolationForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task IsolationForest_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new IsolationForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ExtendedIsolationForest Tests

    [Fact(Timeout = 120000)]
    public async Task ExtendedIsolationForest_OutlierGetsHighestScore()
    {
        var detector = new ExtendedIsolationForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task ExtendedIsolationForest_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ExtendedIsolationForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region SCiForest Tests

    [Fact(Timeout = 120000)]
    public async Task SCiForest_OutlierGetsHighestScore()
    {
        var detector = new SCiForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task SCiForest_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new SCiForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task SCiForest_SdGainSplitSeparatesTheTightestGroups()
    {
        // Liu et al. (2010) Eq. 3 on 0, .1, .2, 10: sigma(Y) = sqrt(73.5275 / 4); the split after .2 leaves
        // sigma(Y_l) = sqrt(.02 / 3), sigma(Y_r) = 0 and beats every other split; p is midway, 5.1.
        var values = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 10.0, 0.1, 0.0, 0.2 }, new[] { 4 });
        var (gain, split) = SCiForest<double>.BestSdGainSplit(values);

        double sigma = Math.Sqrt(73.5275 / 4);
        Assert.Equal((sigma - Math.Sqrt(0.02 / 3) / 2) / sigma, gain, 1e-12);
        Assert.Equal(5.1, split, 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task SCiForest_PathLengthsFollowAlgorithmsOneAndTwo()
    {
        // Data 0, 1, 10 with psi = 3: every tree uses all three rows, and whatever the random coefficient the
        // root's best Sdgain split is {0, 1} | {10} (avg sigma .25 against 2.25 for {0} | {1, 10}). {0, 1} is a
        // leaf of size 2 (c = 1) and {10} a leaf of size 1 (c = 0), so h = 2, 2, 1 and s = 2^(-h / c(3)) with
        // c(3) = 2 (ln 2 + 0.5772156649) - 4 / 3. A query at 1000 leaves the root's acceptable range [-v, v]
        // (v = 10 |w|), so its edge is not counted and it lands in the size-1 leaf: h = 0, s = 1.
        var data = new Matrix<double>(new double[,] { { 0 }, { 1 }, { 10 } });
        var detector = new SCiForest<double>(numTrees: 5);
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);

        double c3 = 2 * (Math.Log(2) + 0.5772156649) - 4.0 / 3.0;
        Assert.Equal(Math.Pow(2, -2 / c3), scores[0], 1e-12);
        Assert.Equal(Math.Pow(2, -2 / c3), scores[1], 1e-12);
        Assert.Equal(Math.Pow(2, -1 / c3), scores[2], 1e-12);

        var far = detector.ScoreAnomalies(new Matrix<double>(new double[,] { { 1000 } }));
        Assert.Equal(1.0, far[0], 1e-12);
    }

    #endregion

    #region FairCutForest Tests

    [Fact(Timeout = 120000)]
    public async Task FairCutForest_OutlierGetsHighestScore()
    {
        var detector = new FairCutForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task FairCutForest_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new FairCutForest<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact(Timeout = 120000)]
    public async Task AllTreeDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new IsolationForest<double>(),
            new ExtendedIsolationForest<double>(),
            new SCiForest<double>(),
            new FairCutForest<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
