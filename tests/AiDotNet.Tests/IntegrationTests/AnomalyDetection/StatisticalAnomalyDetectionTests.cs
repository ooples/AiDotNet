using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for statistical anomaly detection classes.
/// Verifies that each detector correctly identifies known outliers.
/// </summary>
public class StatisticalAnomalyDetectionTests
{
    private const int OutlierIndex = 19; // Last row

    /// <summary>
    /// Creates a 20x2 matrix with normal inliers near (1,2) and one clear outlier at row 19 = (100,100).
    /// </summary>
    private static Matrix<double> CreateTestData()
    {
        int n = 20;
        var data = new double[n, 2];
        for (int i = 0; i < n - 1; i++)
        {
            data[i, 0] = 1.0 + 0.1 * (i % 5);
            data[i, 1] = 2.0 + 0.1 * (i % 7);
        }

        // Outlier far from the cluster
        data[n - 1, 0] = 100.0;
        data[n - 1, 1] = 100.0;

        return new Matrix<double>(data);
    }

    /// <summary>
    /// Asserts the outlier gets a higher anomaly score than all inliers.
    /// </summary>
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

    /// <summary>
    /// Asserts Predict flags the outlier as anomaly (-1) and most inliers as normal (1).
    /// </summary>
    private static void AssertPredictClassifiesCorrectly(Vector<double> predictions, int outlierIdx)
    {
        // Outlier must be classified as anomaly (-1)
        Assert.Equal(-1.0, predictions[outlierIdx]);

        // At least 80% of inliers should be classified as normal (1)
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

    #region ZScoreDetector Tests

    [Fact(Timeout = 120000)]
    public async Task ZScore_Construction_NotFittedByDefault()
    {
        var detector = new ZScoreDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact(Timeout = 120000)]
    public async Task ZScore_OutlierGetsHighestScore()
    {
        var detector = new ZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task ZScore_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ModifiedZScoreDetector Tests

    [Fact(Timeout = 120000)]
    public async Task ModifiedZScore_OutlierGetsHighestScore()
    {
        var detector = new ModifiedZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task ModifiedZScore_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ModifiedZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region IQRDetector Tests

    [Fact(Timeout = 120000)]
    public async Task IQR_OutlierGetsHighestScore()
    {
        var detector = new IQRDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task IQR_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new IQRDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region MADDetector Tests

    [Fact(Timeout = 120000)]
    public async Task MAD_OutlierGetsHighestScore()
    {
        var detector = new MADDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task MAD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new MADDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region HampelDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Hampel_OutlierGetsHighestScore()
    {
        var detector = new HampelDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Hampel_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new HampelDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GrubbsTestDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Grubbs_OutlierGetsHighestScore()
    {
        var detector = new GrubbsTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Grubbs_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GrubbsTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ESDDetector Tests

    [Fact(Timeout = 120000)]
    public async Task ESD_OutlierGetsHighestScore()
    {
        var detector = new ESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task ESD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region GESDDetector Tests

    [Fact(Timeout = 120000)]
    public async Task GESD_OutlierGetsHighestScore()
    {
        var detector = new GESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task GESD_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new GESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region ChiSquareDetector Tests

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_OutlierGetsHighestScore()
    {
        var detector = new ChiSquareDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquare_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new ChiSquareDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region DixonQTestDetector Tests

    [Fact(Timeout = 120000)]
    public async Task DixonQ_OutlierGetsHighestScore()
    {
        var detector = new DixonQTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task DixonQ_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new DixonQTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region PercentileDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Percentile_OutlierGetsHighestScore()
    {
        var detector = new PercentileDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, OutlierIndex);
    }

    [Fact(Timeout = 120000)]
    public async Task Percentile_PredictClassifiesOutlierAsAnomaly()
    {
        var detector = new PercentileDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, OutlierIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact(Timeout = 120000)]
    public async Task AllStatisticalDetectors_NotFittedByDefault()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new ZScoreDetector<double>(),
            new ModifiedZScoreDetector<double>(),
            new IQRDetector<double>(),
            new MADDetector<double>(),
            new HampelDetector<double>(),
            new GrubbsTestDetector<double>(),
            new ESDDetector<double>(),
            new GESDDetector<double>(),
            new ChiSquareDetector<double>(),
            new DixonQTestDetector<double>(),
            new PercentileDetector<double>(),
        };

        foreach (var detector in detectors)
        {
            Assert.False(detector.IsFitted);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AllStatisticalDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new ZScoreDetector<double>(),
            new ModifiedZScoreDetector<double>(),
            new IQRDetector<double>(),
            new MADDetector<double>(),
            new HampelDetector<double>(),
            new GrubbsTestDetector<double>(),
            new ESDDetector<double>(),
            new GESDDetector<double>(),
            new ChiSquareDetector<double>(),
            new DixonQTestDetector<double>(),
            new PercentileDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AllStatisticalDetectors_ScoreAnomaliesBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new ZScoreDetector<double>(),
            new ModifiedZScoreDetector<double>(),
            new IQRDetector<double>(),
            new MADDetector<double>(),
            new HampelDetector<double>(),
            new GrubbsTestDetector<double>(),
            new ESDDetector<double>(),
            new GESDDetector<double>(),
            new ChiSquareDetector<double>(),
            new DixonQTestDetector<double>(),
            new PercentileDetector<double>(),
        };

        var data = CreateTestData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.ScoreAnomalies(data));
        }
    }

    #endregion

    #region Reference Value Tests

    // NIST/SEMATECH e-Handbook 1.3.5.17.3 (Rosner 1983 example data, n = 54).
    private static readonly double[] RosnerData =
    {
        -0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56,
        1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10,
        2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92,
        2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01
    };

    private static Matrix<double> Column(double[] values)
    {
        var data = new double[values.Length, 1];
        for (int i = 0; i < values.Length; i++)
        {
            data[i, 0] = values[i];
        }

        return new Matrix<double>(data);
    }

    [Fact(Timeout = 120000)]
    public async Task Grubbs_UsesSampleStandardDeviationAndExactCriticalValue()
    {
        await Task.Yield();
        // Nine zeros and a ten: mean 1, sample variance (9 * 1 + 81) / 9 = 10, so G(10) = 9 / sqrt(10)
        // = 2.846050 and G(0) = 1 / sqrt(10) = 0.316228 (the population std 3 would give 3 and 1/3).
        // Two-sided 5% critical value for n = 10: t = t_{1 - 0.05/20, 8} = 3.832519, G_crit =
        // (9 / sqrt(10)) sqrt(t^2 / (8 + t^2)) = 2.289954 (Grubbs 1969 table: 2.290).
        var values = new double[10];
        values[9] = 10.0;
        var detector = new GrubbsTestDetector<double>();
        detector.Fit(Column(values));
        var scores = detector.ScoreAnomalies(Column(values));

        Assert.Equal(9.0 / Math.Sqrt(10.0), scores[9], 1e-9);
        Assert.Equal(1.0 / Math.Sqrt(10.0), scores[0], 1e-9);
        var critical = detector.CriticalValues;
        Assert.NotNull(critical);
        Assert.Equal(2.289954, critical[0], 1e-4);
    }

    [Fact(Timeout = 120000)]
    public async Task GESD_MatchesNistRosnerExample()
    {
        await Task.Yield();
        // NIST e-Handbook 1.3.5.17.3: R_1 = 3.118906 (sample std) and, for alpha = 0.05, r = 10,
        // lambda_i = (n - i) t_{p, n-i-1} / sqrt((n - i - 1 + t^2)(n - i + 1)), p = 1 - alpha / (2(n - i + 1)),
        // tabulated (truncated to 3 decimals) as below.
        double[] lambdaTable = { 3.158, 3.151, 3.143, 3.136, 3.128, 3.120, 3.111, 3.103, 3.094, 3.085 };
        var data = Column(RosnerData);
        var detector = new GESDDetector<double>(maxOutliers: 10, alpha: 0.05);
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);

        double maxScore = double.MinValue;
        for (int i = 0; i < scores.Length; i++)
        {
            maxScore = Math.Max(maxScore, scores[i]);
        }

        Assert.Equal(3.118906, maxScore, 1e-5);
        var critical = detector.CriticalValues;
        Assert.NotNull(critical);
        for (int i = 0; i < lambdaTable.Length; i++)
        {
            Assert.InRange(critical[i], lambdaTable[i], lambdaTable[i] + 0.001);
        }
    }

    #endregion
}
