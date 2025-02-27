namespace AiDotNet.Models.Options;

public class ProximalGradientDescentOptimizerOptions : GradientBasedOptimizerOptions
{
    public double RegularizationStrength { get; set; } = 0.01;
    public double ProximalStepSize { get; set; } = 0.1;
    public int InnerIterations { get; set; } = 10;
    public new double InitialLearningRate { get; set; } = 0.01;
    public new int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
    public new bool UseAdaptiveLearningRate { get; set; } = true;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
    public new double MinLearningRate { get; set; } = 1e-6;
    public new double MaxLearningRate { get; set; } = 1.0;
}