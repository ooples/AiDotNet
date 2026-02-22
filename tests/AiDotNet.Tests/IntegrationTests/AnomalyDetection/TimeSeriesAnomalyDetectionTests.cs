using AiDotNet.AnomalyDetection;
using AiDotNet.AnomalyDetection.TimeSeries;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for time series anomaly detection classes.
/// Verifies that each detector correctly identifies anomaly spikes in time series data.
/// </summary>
public class TimeSeriesAnomalyDetectionTests
{
    private const int AnomalyIndex = 25; // Spike injected at this position

    /// <summary>
    /// Creates a time series matrix (Nx1) with a sinusoidal pattern and one clear anomaly spike.
    /// Values range from ~8 to ~12, with a spike of 100.0 at index 25.
    /// </summary>
    private static Matrix<double> CreateTimeSeriesData()
    {
        int n = 50;
        var data = new double[n, 1];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = 10.0 + 2.0 * Math.Sin(2.0 * Math.PI * i / 12.0);
        }

        // Inject anomaly - value of 100 when normal range is [8, 12]
        data[AnomalyIndex, 0] = 100.0;

        return new Matrix<double>(data);
    }

    private static void AssertOutlierScoresHighest(Vector<double> scores, int outlierIdx)
    {
        double outlierScore = scores[outlierIdx];
        for (int i = 0; i < scores.Length; i++)
        {
            if (i == outlierIdx) continue;
            Assert.True(outlierScore > scores[i],
                $"Anomaly score ({outlierScore:F4}) at index {outlierIdx} should be higher than " +
                $"normal score ({scores[i]:F4}) at index {i}");
        }
    }

    private static void AssertPredictClassifiesCorrectly(Vector<double> predictions, int outlierIdx)
    {
        // Anomaly spike should be classified as anomaly (-1)
        Assert.Equal(-1.0, predictions[outlierIdx]);

        // Most normal points should be classified as normal (1)
        int normalCount = 0;
        int inlierCount = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            if (i == outlierIdx) continue;
            inlierCount++;
            if (predictions[i] == 1.0) normalCount++;
        }

        Assert.True(normalCount >= inlierCount * 0.8,
            $"Expected at least {inlierCount * 0.8} normal points classified correctly, got {normalCount}/{inlierCount}");
    }

    #region MovingAverageDetector Tests

    [Fact]
    public void MovingAverage_Construction_NotFittedByDefault()
    {
        var detector = new MovingAverageDetector<double>();
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void MovingAverage_OutlierGetsHighestScore()
    {
        var detector = new MovingAverageDetector<double>(windowSize: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void MovingAverage_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new MovingAverageDetector<double>(windowSize: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region ARIMADetector Tests

    [Fact]
    public void ARIMA_OutlierGetsHighestScore()
    {
        var detector = new ARIMADetector<double>(p: 1, d: 0, q: 0);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void ARIMA_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new ARIMADetector<double>(p: 1, d: 0, q: 0);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region STLDetector Tests

    [Fact]
    public void STL_OutlierGetsHighestScore()
    {
        var detector = new STLDetector<double>(seasonLength: 12);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void STL_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new STLDetector<double>(seasonLength: 12);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region MatrixProfileDetector Tests

    [Fact]
    public void MatrixProfile_OutlierGetsHighestScore()
    {
        var detector = new MatrixProfileDetector<double>(subsequenceLength: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void MatrixProfile_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new MatrixProfileDetector<double>(subsequenceLength: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region SpectralResidualDetector Tests

    [Fact]
    public void SpectralResidual_OutlierGetsHighestScore()
    {
        var detector = new SpectralResidualDetector<double>();
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void SpectralResidual_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new SpectralResidualDetector<double>();
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region SeasonalHybridESDDetector Tests

    [Fact]
    public void SeasonalHybridESD_OutlierGetsHighestScore()
    {
        var detector = new SeasonalHybridESDDetector<double>(seasonLength: 12);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void SeasonalHybridESD_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new SeasonalHybridESDDetector<double>(seasonLength: 12);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region LSTMDetector Tests

    [Fact]
    public void LSTM_OutlierGetsHighestScore()
    {
        var detector = new LSTMDetector<double>(seqLength: 5, epochs: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void LSTM_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new LSTMDetector<double>(seqLength: 5, epochs: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region NBEATSDetector Tests

    [Fact]
    public void NBEATS_OutlierGetsHighestScore()
    {
        var detector = new NBEATSDetector<double>(epochs: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void NBEATS_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new NBEATSDetector<double>(epochs: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region AnomalyTransformerDetector Tests

    [Fact]
    public void AnomalyTransformer_OutlierGetsHighestScore()
    {
        var detector = new AnomalyTransformerDetector<double>(seqLength: 10, epochs: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var scores = detector.ScoreAnomalies(data);
        Assert.Equal(data.Rows, scores.Length);
        AssertOutlierScoresHighest(scores, AnomalyIndex);
    }

    [Fact]
    public void AnomalyTransformer_PredictClassifiesAnomalyCorrectly()
    {
        var detector = new AnomalyTransformerDetector<double>(seqLength: 10, epochs: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        AssertPredictClassifiesCorrectly(predictions, AnomalyIndex);
    }

    #endregion

    #region Cross-Detector Tests

    [Fact]
    public void AllTimeSeriesDetectors_PredictBeforeFit_Throws()
    {
        var detectors = new AnomalyDetectorBase<double>[]
        {
            new MovingAverageDetector<double>(),
            new ARIMADetector<double>(),
            new STLDetector<double>(),
            new MatrixProfileDetector<double>(),
            new SpectralResidualDetector<double>(),
            new SeasonalHybridESDDetector<double>(),
            new LSTMDetector<double>(),
            new NBEATSDetector<double>(),
            new AnomalyTransformerDetector<double>(),
        };

        var data = CreateTimeSeriesData();
        foreach (var detector in detectors)
        {
            Assert.Throws<InvalidOperationException>(() => detector.Predict(data));
        }
    }

    #endregion
}
