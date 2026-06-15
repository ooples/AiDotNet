namespace AiDotNet.Models.Options;

/// <summary>
/// Options for <see cref="AiDotNet.TimeSeries.TiDEModel{T}"/> — Time-series Dense Encoder (Das et al.,
/// "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR 2023). A pure-MLP encoder/decoder
/// with a linear residual; on long-horizon benchmarks it matches or beats transformers at a fraction of
/// the cost. Faithful core: a ReLU encoder MLP + decoder projection + a linear skip from the input window.
/// </summary>
public class TiDEOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int LookbackWindow { get; set; } = 24;
    public int ForecastHorizon { get; set; } = 1;
    public int HiddenSize { get; set; } = 64;
    public double LearningRate { get; set; } = 0.01;
    public int Epochs { get; set; } = 50;
    public int BatchSize { get; set; } = 32;

    public TiDEOptions() { }

    public TiDEOptions(TiDEOptions<T> other)
    {
        if (other == null) { throw new ArgumentNullException(nameof(other)); }
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        HiddenSize = other.HiddenSize;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }
}
