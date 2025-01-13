namespace AiDotNet.Models.Options;

public class MAModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int MAOrder { get; set; } = 1;
    public double LearningRate { get; set; } = 0.01;
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
}