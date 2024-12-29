namespace AiDotNet.Models.Options;

public class GradientDescentOptimizerOptions : OptimizationAlgorithmOptions
{
    public double LearningRate { get; set; } = 0.01;
    public double Tolerance { get; set; } = 1e-6;
}