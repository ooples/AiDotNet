namespace AiDotNet.Models.Options;

public class QuantileRegressionOptions<T> : RegressionOptions<T>
{
    public double Quantile { get; set; } = 0.5; // Default to median regression
    public double LearningRate { get; set; } = 0.01;
    public int MaxIterations { get; set; } = 1000;
}