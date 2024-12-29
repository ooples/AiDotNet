namespace AiDotNet.Models.Options;

public class ARIMAOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int P { get; set; } = 1;  // AR order
    public int D { get; set; } = 1;  // Differencing order
    public int Q { get; set; } = 1;  // MA order
    public double LearningRate { get; set; } = 0.01;
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-5;
    public bool FitIntercept { get; set; } = true;
}