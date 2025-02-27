namespace AiDotNet.Models.Options;

public class RobustRegressionOptions<T> : RegressionOptions<T>
{
    public double TuningConstant { get; set; } = 1.345;
    public int MaxIterations { get; set; } = 100;
    public double Tolerance { get; set; } = 1e-6;
    public WeightFunction WeightFunction { get; set; } = WeightFunction.Huber;
    public IRegression<T>? InitialRegression { get; set; }
}