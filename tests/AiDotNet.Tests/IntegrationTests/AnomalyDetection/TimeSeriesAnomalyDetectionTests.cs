using AiDotNet.AnomalyDetection.TimeSeries;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AnomalyDetection;

/// <summary>
/// Integration tests for time series anomaly detection classes.
/// </summary>
public class TimeSeriesAnomalyDetectionTests
{
    /// <summary>
    /// Creates a time series matrix (Nx1) with a clear anomaly spike.
    /// </summary>
    private static Matrix<double> CreateTimeSeriesData()
    {
        int n = 50;
        var data = new double[n, 1];
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = 10.0 + 2.0 * Math.Sin(2.0 * Math.PI * i / 12.0);
        }

        // Inject anomaly
        data[25, 0] = 100.0;

        return new Matrix<double>(data);
    }

    #region MovingAverageDetector Tests

    [Fact]
    public void MovingAverage_Construction_DoesNotThrow()
    {
        var detector = new MovingAverageDetector<double>();
        Assert.NotNull(detector);
        Assert.False(detector.IsFitted);
    }

    [Fact]
    public void MovingAverage_FitAndPredict_DetectsSpike()
    {
        var detector = new MovingAverageDetector<double>(windowSize: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        Assert.True(detector.IsFitted);

        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);

        // The spike at index 25 (value=100 vs normal ~10) should be detected
        Assert.Equal(-1.0, predictions[25]);

        // Verify anomaly scores: spike should have highest score
        var scores = detector.ScoreAnomalies(data);
        double spikeScore = scores[25];
        double normalScore = scores[0];
        Assert.True(spikeScore > normalScore, "Spike score should exceed normal score");
    }

    #endregion

    #region ARIMADetector Tests

    [Fact]
    public void ARIMA_Construction_DoesNotThrow()
    {
        var detector = new ARIMADetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void ARIMA_FitAndPredict_Works()
    {
        var detector = new ARIMADetector<double>(p: 1, d: 0, q: 0);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region STLDetector Tests

    [Fact]
    public void STL_Construction_DoesNotThrow()
    {
        var detector = new STLDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void STL_FitAndPredict_Works()
    {
        var detector = new STLDetector<double>(seasonLength: 12);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region MatrixProfileDetector Tests

    [Fact]
    public void MatrixProfile_Construction_DoesNotThrow()
    {
        var detector = new MatrixProfileDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void MatrixProfile_FitAndPredict_Works()
    {
        var detector = new MatrixProfileDetector<double>(subsequenceLength: 5);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region SpectralResidualDetector Tests

    [Fact]
    public void SpectralResidual_Construction_DoesNotThrow()
    {
        var detector = new SpectralResidualDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void SpectralResidual_FitAndPredict_Works()
    {
        var detector = new SpectralResidualDetector<double>();
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region SeasonalHybridESDDetector Tests

    [Fact]
    public void SeasonalHybridESD_Construction_DoesNotThrow()
    {
        var detector = new SeasonalHybridESDDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void SeasonalHybridESD_FitAndPredict_Works()
    {
        var detector = new SeasonalHybridESDDetector<double>(seasonLength: 12);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region LSTMDetector Tests

    [Fact]
    public void LSTM_Construction_DoesNotThrow()
    {
        var detector = new LSTMDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void LSTM_FitAndPredict_Works()
    {
        var detector = new LSTMDetector<double>(seqLength: 5, epochs: 3);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region NBEATSDetector Tests

    [Fact]
    public void NBEATS_Construction_DoesNotThrow()
    {
        var detector = new NBEATSDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void NBEATS_FitAndPredict_Works()
    {
        var detector = new NBEATSDetector<double>(epochs: 3);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion

    #region AnomalyTransformerDetector Tests

    [Fact]
    public void AnomalyTransformer_Construction_DoesNotThrow()
    {
        var detector = new AnomalyTransformerDetector<double>();
        Assert.NotNull(detector);
    }

    [Fact]
    public void AnomalyTransformer_FitAndPredict_Works()
    {
        var detector = new AnomalyTransformerDetector<double>(seqLength: 10, epochs: 3);
        var data = CreateTimeSeriesData();
        detector.Fit(data);
        var predictions = detector.Predict(data);
        Assert.Equal(data.Rows, predictions.Length);
    }

    #endregion
}
