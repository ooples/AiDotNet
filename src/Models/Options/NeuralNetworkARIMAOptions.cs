namespace AiDotNet.Models.Options;

public class NeuralNetworkARIMAOptions<T> : TimeSeriesRegressionOptions<T>
{
    public int AROrder { get; set; } = 1;
    public int MAOrder { get; set; } = 1;
    public int LaggedPredictions { get; set; } = 1;
    public int ExogenousVariables { get; set; } = 0;
    public bool UseVectorActivations { get; set; } = false;
    public INeuralNetwork<T>? NeuralNetwork { get; set; }
    public IOptimizer<T>? Optimizer { get; set; }
}