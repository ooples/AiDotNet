using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.DistanceBased;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for distance-based anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class DistanceBasedAnomalyDetectionTests
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

    #region LocalOutlierFactor Tests

    [Fact(Timeout = 120000)]
    public async Task LOF_Construction_NotFittedByDefault()
    {
        var detector = new LocalOutlierFactor<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact(Timeout = 120000)]
    public async Task LOF_OutlierGetsHighestScore()
    {
        var detector = new LocalOutlierFactor<double>();
        var data = CreateTestData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task LOF_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LocalOutlierFactor<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region KNNDetector Tests

    [Fact(Timeout = 120000)]
    public async Task KNN_OutlierGetsHighestScore()
    {
        var detector = new KNNDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new KNNDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region COFDetector Tests

    [Fact(Timeout = 120000)]
    public async Task COF_OutlierGetsHighestScore()
    {
        var detector = new COFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task COF_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new COFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public Task COF_ScoresRectangularQueryAgainstTrainingDistanceMatrix()
    {
        var detector = new COFDetector<double>(k: 5);
        detector.Fit(CreateTestData());
        var query = new Matrix<double>(new[,]
        {
            { 1.1, 2.2 },
            { 1.3, 2.4 },
            { 90.0, 90.0 }
        });

        var scores = detector.ScoreAnomalies(query);

        Assert.Equal(query.Rows, scores.Length);
        for (int i = 0; i < scores.Length; i++)
            Assert.True(!double.IsNaN(scores[i]) && !double.IsInfinity(scores[i]),
                $"COF score {i} was {scores[i]}.");

        // THE ORDERING IS THE BEHAVIOUR. Count and finiteness are satisfied by a detector that
        // returns the same constant for every row, which is the degenerate this test is meant to
        // catch. Rows 0 and 1 sit inside the training cluster and row 2 is at (90, 90), far
        // outside it, so the outlier must score strictly higher than both.
        Assert.True(scores[2] > scores[0] && scores[2] > scores[1],
            $"COF ranked the far-outlier row (90, 90) at {scores[2]} against in-cluster rows at " +
            $"{scores[0]} and {scores[1]}. An outlier must score strictly higher than points drawn " +
            "from the training distribution.");

        return Task.CompletedTask;
    }

    #endregion

    #region INFLODetector Tests

    [Fact(Timeout = 120000)]
    public async Task INFLO_OutlierGetsHighestScore()
    {
        var detector = new INFLODetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task INFLO_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new INFLODetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region LoOPDetector Tests

    [Fact(Timeout = 120000)]
    public async Task LoOP_OutlierGetsHighestScore()
    {
        var detector = new LoOPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task LoOP_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LoOPDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task LoOP_MatchesKriegelReferenceValues()
    {
        // Kriegel et al. (2009), "LoOP: Local Outlier Probabilities": with S(o) the k nearest neighbours,
        // pdist(o) = sqrt(sum d^2 / |S|) (the lambda factor of pdist cancels in the ratio), PLOF(o) =
        // pdist(o) / E_{s in S(o)}[pdist(s)] - 1, the aggregate nPLOF = lambda * sqrt(E[PLOF^2]) with
        // the expectation over ALL objects of the data set (as ELKI computes it once per relation), and
        // LoOP(o) = max(0, erf(PLOF(o) / (nPLOF * sqrt(2)))).
        // Points 0, 1, 2, 3, 10 with k = 2, lambda = 3, worked by hand:
        //   neighbours 0:{1,2} 1:{0,2} 2:{1,3} 3:{2,1} 10:{3,2}
        //   pdist = sqrt(2.5), 1, 1, sqrt(2.5), sqrt(56.5)
        //   E[pdist] over S(o) = 1, c, c, 1, c with c = (sqrt(2.5) + 1) / 2 = 1.290569415042095
        //   PLOF  = sqrt(2.5) - 1, 1/c - 1, 1/c - 1, sqrt(2.5) - 1, sqrt(56.5)/c - 1
        //         = 0.5811388300841898, -0.2251482265544139 (twice), 0.5811388300841898, 4.824288179757677
        //   nPLOF = 3 * sqrt((2 * 0.5811388300841898^2 + 2 * 0.2251482265544139^2 + 4.824288179757677^2) / 5)
        //         = 6.579593621488894
        //   LoOP(0) = LoOP(3) = erf(0.5811388300841898 / (6.579593621488894 * sqrt(2))) = 0.07038117433941146
        //   LoOP(1) = LoOP(2) = 0 (negative PLOF)
        //   LoOP(10) = erf(4.824288179757677 / (6.579593621488894 * sqrt(2))) = 0.5365756061719503
        var data = new Matrix<double>(new double[,] { { 0 }, { 1 }, { 2 }, { 3 }, { 10 } });
        var detector = new LoOPDetector<double>(k: 2, lambda: 3);
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);

        Assert.Equal(0.07038117433941146, scores[0], 1e-6);
        Assert.Equal(0.0, scores[1], 1e-12);
        Assert.Equal(0.0, scores[2], 1e-12);
        Assert.Equal(0.07038117433941146, scores[3], 1e-6);
        Assert.Equal(0.5365756061719503, scores[4], 1e-6);
    }

    #endregion

    #region LOCIDetector Tests

    [Fact(Timeout = 120000)]
    public async Task LOCI_OutlierGetsHighestScore()
    {
        var detector = new LOCIDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task LOCI_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LOCIDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region LDCOFDetector Tests

    [Fact(Timeout = 120000)]
    public async Task LDCOF_OutlierGetsHighestScore()
    {
        var detector = new LDCOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task LDCOF_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new LDCOFDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region OCSVMDetector Tests

    [Fact(Timeout = 120000)]
    public async Task OCSVM_OutlierGetsHighestScore()
    {
        var detector = new OCSVMDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task OCSVM_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new OCSVMDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region SOSDetector Tests

    [Fact(Timeout = 120000)]
    public async Task SOS_OutlierGetsHighestScore()
    {
        var detector = new SOSDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task SOS_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new SOSDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact(Timeout = 120000)]
    public async Task AllDistanceDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new LocalOutlierFactor<double>(),
            new KNNDetector<double>(),
            new COFDetector<double>(),
            new INFLODetector<double>(),
            new LoOPDetector<double>(),
            new LOCIDetector<double>(),
            new LDCOFDetector<double>(),
            new OCSVMDetector<double>(),
            new SOSDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
