namespace AiDotNet.Models.Options;

public class MultinomialLogisticRegressionOptions<T> : RegressionOptions<T>
{
    public int MaxIterations { get; set; } = 100;
    public double Tolerance { get; set; } = 1e-4;
}