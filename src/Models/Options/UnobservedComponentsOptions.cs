namespace AiDotNet.Models.Options;

public class UnobservedComponentsOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int MaxIterations { get; set; } = 100;
    public bool OptimizeParameters { get; set; } = true;
    public IOptimizer<T>? Optimizer { get; set; }
    public new int SeasonalPeriod { get; set; } = 1; // Default to no seasonality
    public bool IncludeCycle { get; set; } = false;
    public double CycleLambda { get; set; } = 1600; // Default value for quarterly data
    public int CycleMinPeriod { get; set; } = 2;
    public int CycleMaxPeriod { get; set; } = 40;
    public IMatrixDecomposition<T>? Decomposition { get; set; }
}