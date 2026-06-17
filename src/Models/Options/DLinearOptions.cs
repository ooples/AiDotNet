namespace AiDotNet.Models.Options;

/// <summary>
/// Options for <see cref="AiDotNet.TimeSeries.DLinearModel{T}"/> — the decomposition-linear forecaster
/// (Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023). Despite its
/// simplicity it is a strong, current baseline that frequently matches or beats heavier transformers on
/// standard long-horizon benchmarks, at a fraction of the cost.
/// </summary>
public class DLinearOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>History length the linear maps see (input window).</summary>
    public int LookbackWindow { get; set; } = 24;

    /// <summary>Forecast horizon. The supervised harness uses 1 (predict the next target value).</summary>
    public int ForecastHorizon { get; set; } = 1;

    /// <summary>Moving-average kernel for the trend/seasonal decomposition (odd; clamped to the window).</summary>
    public int MovingAverageKernel { get; set; } = 25;

    public double LearningRate { get; set; } = 0.01;
    public int Epochs { get; set; } = 50;
    public int BatchSize { get; set; } = 32;

    public DLinearOptions() { }

    public DLinearOptions(DLinearOptions<T> other)
    {
        if (other == null) { throw new ArgumentNullException(nameof(other)); }
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        MovingAverageKernel = other.MovingAverageKernel;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }
}
