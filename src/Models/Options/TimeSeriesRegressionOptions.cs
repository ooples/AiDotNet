namespace AiDotNet.Models.Options;

public class TimeSeriesRegressionOptions<T> : RegressionOptions<T>
{
    public int LagOrder { get; set; } = 1;
    public bool IncludeTrend { get; set; } = true;
    public int SeasonalPeriod { get; set; } = 0; // 0 means no seasonality
    public bool AutocorrelationCorrection { get; set; } = true;
    public TimeSeriesModelType ModelType { get; set; } = TimeSeriesModelType.ARIMA;
}