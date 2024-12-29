namespace AiDotNet.Models.Options;

public class SARIMAOptions<T> : TimeSeriesRegressionOptions<T>
{
    // Non-seasonal components
    public int P { get; set; } = 1;  // Autoregressive order
    public int D { get; set; } = 0;  // Differencing order
    public int Q { get; set; } = 1;  // Moving average order

    // Seasonal components
    public int SeasonalP { get; set; } = 1;  // Seasonal autoregressive order
    public int SeasonalD { get; set; } = 0;  // Seasonal differencing order
    public int SeasonalQ { get; set; } = 1;  // Seasonal moving average order
    public new int SeasonalPeriod { get; set; } = 12;  // Default to monthly seasonality

    // Other parameters
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-5;
}