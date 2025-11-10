using AiDotNet.TimeSeries;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Time Series models and forecasting
/// Tests performance of ARIMA, exponential smoothing, and other time series methods
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class TimeSeriesBenchmarks
{
    [Params(100, 500, 1000)]
    public int TimeSeriesLength { get; set; }

    [Params(10, 50)]
    public int ForecastHorizon { get; set; }

    private Vector<double> _timeSeries = null!;
    private Vector<double> _seasonalTimeSeries = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Generate synthetic time series with trend and noise
        _timeSeries = new Vector<double>(TimeSeriesLength);
        double trend = 0;
        for (int i = 0; i < TimeSeriesLength; i++)
        {
            trend += 0.1;
            _timeSeries[i] = trend + random.NextDouble() * 5;
        }

        // Generate seasonal time series
        _seasonalTimeSeries = new Vector<double>(TimeSeriesLength);
        for (int i = 0; i < TimeSeriesLength; i++)
        {
            double seasonal = Math.Sin(2 * Math.PI * i / 12) * 10;
            trend += 0.1;
            _seasonalTimeSeries[i] = trend + seasonal + random.NextDouble() * 2;
        }
    }

    #region ARIMA Models

    [Benchmark(Baseline = true)]
    public Vector<double> ARIMA_Fit_Forecast()
    {
        var model = new ARIMAModel<double>(p: 1, d: 1, q: 1);
        model.Fit(_timeSeries);
        return model.Forecast(ForecastHorizon);
    }

    [Benchmark]
    public Vector<double> AR_Fit_Forecast()
    {
        var model = new ARModel<double>(order: 2);
        model.Fit(_timeSeries);
        return model.Forecast(ForecastHorizon);
    }

    [Benchmark]
    public Vector<double> MA_Fit_Forecast()
    {
        var model = new MAModel<double>(order: 2);
        model.Fit(_timeSeries);
        return model.Forecast(ForecastHorizon);
    }

    [Benchmark]
    public Vector<double> ARMA_Fit_Forecast()
    {
        var model = new ARMAModel<double>(p: 1, q: 1);
        model.Fit(_timeSeries);
        return model.Forecast(ForecastHorizon);
    }

    #endregion

    #region Seasonal Models

    [Benchmark]
    public Vector<double> SARIMA_Fit_Forecast()
    {
        var model = new SARIMAModel<double>(p: 1, d: 1, q: 1, P: 1, D: 1, Q: 1, seasonalPeriod: 12);
        model.Fit(_seasonalTimeSeries);
        return model.Forecast(ForecastHorizon);
    }

    #endregion

    #region Exponential Smoothing

    [Benchmark]
    public Vector<double> ExponentialSmoothing_Fit_Forecast()
    {
        var model = new ExponentialSmoothingModel<double>(alpha: 0.3);
        model.Fit(_timeSeries);
        return model.Forecast(ForecastHorizon);
    }

    #endregion

    #region State Space Models

    [Benchmark]
    public Vector<double> StateSpace_Fit_Forecast()
    {
        var model = new StateSpaceModel<double>();
        model.Fit(_timeSeries);
        return model.Forecast(ForecastHorizon);
    }

    #endregion

    #region Model Evaluation

    [Benchmark]
    public double ARIMA_Fit_Evaluate()
    {
        var model = new ARIMAModel<double>(p: 1, d: 1, q: 1);
        model.Fit(_timeSeries);

        // In-sample prediction
        var predictions = model.Predict(_timeSeries.Length);

        // Calculate MSE
        double mse = 0;
        for (int i = 0; i < _timeSeries.Length; i++)
        {
            double error = _timeSeries[i] - predictions[i];
            mse += error * error;
        }
        return mse / _timeSeries.Length;
    }

    #endregion
}
