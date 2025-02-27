namespace AiDotNet.Models.Options;

public class ARModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int AROrder { get; set; } = 1;
    public double LearningRate { get; set; } = 0.01;
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
}