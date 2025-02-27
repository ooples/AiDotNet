namespace AiDotNet.Models.Options;

public class StateSpaceModelOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int StateSize { get; set; } = 1;
    public int ObservationSize { get; set; } = 1;
    public double LearningRate { get; set; } = 0.01;
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
}