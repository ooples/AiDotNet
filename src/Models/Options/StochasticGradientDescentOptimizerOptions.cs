namespace AiDotNet.Models.Options;

public class StochasticGradientDescentOptimizerOptions : OptimizationAlgorithmOptions
{
    public double LearningRate { get; set; } = 0.01;
    public new int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
}