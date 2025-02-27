namespace AiDotNet.Models.Options;

public class LogisticRegressionOptions<T> : RegressionOptions<T>
{
    public int MaxIterations { get; set; } = 1000;
    public double LearningRate { get; set; } = 0.01;
    public double Tolerance { get; set; } = 1e-4;
}