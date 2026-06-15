namespace AiDotNet.Models.Options;

/// <summary>
/// Options for <see cref="AiDotNet.TimeSeries.NLinearModel{T}"/> — the normalization-linear forecaster
/// (Zeng et al., AAAI 2023). It subtracts the last observed value from the window, applies a single linear
/// map, then adds the value back — a distribution-shift-robust baseline that, like DLinear, is a strong
/// modern control against heavier models.
/// </summary>
public class NLinearOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int LookbackWindow { get; set; } = 24;
    public int ForecastHorizon { get; set; } = 1;
    public double LearningRate { get; set; } = 0.01;
    public int Epochs { get; set; } = 50;
    public int BatchSize { get; set; } = 32;

    public NLinearOptions() { }

    public NLinearOptions(NLinearOptions<T> other)
    {
        if (other == null) { throw new ArgumentNullException(nameof(other)); }
        LookbackWindow = other.LookbackWindow;
        ForecastHorizon = other.ForecastHorizon;
        LearningRate = other.LearningRate;
        Epochs = other.Epochs;
        BatchSize = other.BatchSize;
    }
}
