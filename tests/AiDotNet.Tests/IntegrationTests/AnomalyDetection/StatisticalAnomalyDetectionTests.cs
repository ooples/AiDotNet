using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for statistical anomaly detection classes.
/// </summary>
public class StatisticalAnomalyDetectionTests
{
    /// <summary>
    /// Creates a 20x2 matrix with normal inliers and one clear outlier at row 19.
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

        // Outlier
        data[n - 1, 0] = 100.0;
        data[n - 1, 1] = 100.0;

        return new Matrix<double>(data);
    }

    #region ZScoreDetector Tests

    [Fact]
    public void ZScore_Construction_DoesNotThrow()
    {
        var detector = new ZScoreDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void ZScore_Fit_SetsIsFitted()
    {
        var detector = new ZScoreDetector<double>();
        detector.Fit(CreateTestData());
        Assert.True(detector.IsFitted);
    }

    [Fact]
    public void ZScore_Predict_DetectsStatisticalOutlier()
    {
        var detector = new ZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // The outlier at (100,100) should have a z-score far above threshold
        Assert.Equal(-1.0, predictions[data.Rows - 1]);

        // Verify scores: outlier should have much higher anomaly score
        var scores = detector.ScoreAnomalies(data);
        double outlierScore = scores[data.Rows - 1];
        double inlierScore = scores[0];
        Assert.True(outlierScore > inlierScore, "Outlier score should exceed inlier score");
    }

    #endregion

    #region ModifiedZScoreDetector Tests

    [Fact]
    public void ModifiedZScore_Construction_DoesNotThrow()
    {
        var detector = new ModifiedZScoreDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void ModifiedZScore_FitAndPredict_Works()
    {
        var detector = new ModifiedZScoreDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region IQRDetector Tests

    [Fact]
    public void IQR_Construction_DoesNotThrow()
    {
        var detector = new IQRDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void IQR_FitAndPredict_Works()
    {
        var detector = new IQRDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region MADDetector Tests

    [Fact]
    public void MAD_Construction_DoesNotThrow()
    {
        var detector = new MADDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void MAD_FitAndPredict_Works()
    {
        var detector = new MADDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region HampelDetector Tests

    [Fact]
    public void Hampel_Construction_DoesNotThrow()
    {
        var detector = new HampelDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void Hampel_FitAndPredict_Works()
    {
        var detector = new HampelDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region GrubbsTestDetector Tests

    [Fact]
    public void Grubbs_Construction_DoesNotThrow()
    {
        var detector = new GrubbsTestDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void Grubbs_FitAndPredict_Works()
    {
        var detector = new GrubbsTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region ESDDetector Tests

    [Fact]
    public void ESD_Construction_DoesNotThrow()
    {
        var detector = new ESDDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void ESD_FitAndPredict_Works()
    {
        var detector = new ESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region GESDDetector Tests

    [Fact]
    public void GESD_Construction_DoesNotThrow()
    {
        var detector = new GESDDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void GESD_FitAndPredict_Works()
    {
        var detector = new GESDDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region ChiSquareDetector Tests

    [Fact]
    public void ChiSquare_Construction_DoesNotThrow()
    {
        var detector = new ChiSquareDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void ChiSquare_FitAndPredict_Works()
    {
        var detector = new ChiSquareDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region DixonQTestDetector Tests

    [Fact]
    public void DixonQ_Construction_DoesNotThrow()
    {
        var detector = new DixonQTestDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void DixonQ_FitAndPredict_Works()
    {
        var detector = new DixonQTestDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region PercentileDetector Tests

    [Fact]
    public void Percentile_Construction_DoesNotThrow()
    {
        var detector = new PercentileDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void Percentile_FitAndPredict_Works()
    {
        var detector = new PercentileDetector<double>();
        var data = CreateTestData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllStatisticalDetectors_NotFittedByDefault()
    {
        var detectors = new AiDotNet.AnomalyDetection.AnomalyDetectorBase<double>[]
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

    #endregion
}
