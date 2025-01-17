namespace AiDotNet.Models.Options;

public class STLDecompositionOptions<T> : TimeSeriesRegressionOptions<T>
{
    public new int SeasonalPeriod { get; set; } = 12; // Default to monthly data
    public int SeasonalDegree { get; set; } = 1;
    public int TrendDegree { get; set; } = 1;
    public int SeasonalJump { get; set; } = 1;
    public int TrendJump { get; set; } = 1;
    public int InnerLoopPasses { get; set; } = 2;
    public int RobustIterations { get; set; } = 1;
    public double RobustWeightThreshold { get; set; } = 0.001;
    public double SeasonalBandwidth { get; set; } = 0.75;
    public double TrendBandwidth { get; set; } = 0.75;
    public double LowPassBandwidth { get; set; } = 0.75;
    public int TrendWindowSize { get; set; } = 18; // 1.5 * default SeasonalPeriod
    public int SeasonalLoessWindow { get; set; } = 121; // 10 * default SeasonalPeriod + 1
    public int LowPassFilterWindowSize { get; set; } = 12; // Same as default SeasonalPeriod
    public int TrendLoessWindow { get; set; } = 12; // Same as default SeasonalPeriod
}