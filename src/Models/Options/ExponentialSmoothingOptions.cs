namespace AiDotNet.Models.Options;

public class ExponentialSmoothingOptions<T> : TimeSeriesRegressionOptions<T>
{
    public double InitialAlpha { get; set; } = 0.3;
    public double InitialBeta { get; set; } = 0.1;
    public double InitialGamma { get; set; } = 0.1;
    public bool UseTrend { get; set; } = true;
    public bool UseSeasonal { get; set; } = false;
    public double GridSearchStep { get; set; } = 0.1;
}