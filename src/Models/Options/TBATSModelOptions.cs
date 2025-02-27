namespace AiDotNet.Models.Options;

public class TBATSModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int BoxCoxLambda { get; set; } = 1;
    public int ARMAOrder { get; set; } = 1;
    public int TrendDampingFactor { get; set; } = 1;
    public int[] SeasonalPeriods { get; set; } = new int[] { 7, 30, 365 };
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}