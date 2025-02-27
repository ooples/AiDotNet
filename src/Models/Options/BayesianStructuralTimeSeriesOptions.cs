namespace AiDotNet.Models.Options;

public class BayesianStructuralTimeSeriesOptions<T> : TimeSeriesRegressionOptions<T>
{
    public double InitialLevelValue { get; set; } = 0.0;
    public double InitialTrendValue { get; set; } = 0.0;
    public List<int> SeasonalPeriods { get; set; } = [];
    public double InitialObservationVariance { get; set; } = 1.0;
    public bool PerformBackwardSmoothing { get; set; } = true;
    public int MaxIterations { get; set; } = 100;
    public double ConvergenceTolerance { get; set; } = 1e-6;
    public bool UseAutomaticPriors { get; set; } = true;
    public double LevelSmoothingPrior { get; set; } = 0.01;
    public double TrendSmoothingPrior { get; set; } = 0.01;
    public double SeasonalSmoothingPrior { get; set; } = 0.01;
    public bool IncludeRegression { get; set; } = false;
    public MatrixDecompositionType RegressionDecompositionType { get; set; } = MatrixDecompositionType.Lu;
    public double RidgeParameter { get; set; } = 0.1;
}