namespace AiDotNet.Models.Options;

public class GARCHModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int ARCHOrder { get; set; } = 1;
    public int GARCHOrder { get; set; } = 1;
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
    public ITimeSeriesModel<T>? MeanModel { get; set; }
}