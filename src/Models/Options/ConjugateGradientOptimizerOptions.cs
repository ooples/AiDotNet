namespace AiDotNet.Models.Options;

public class ConjugateGradientOptimizerOptions : GradientBasedOptimizerOptions
{
    public new double InitialLearningRate { get; set; } = 0.1;
    public new double MinLearningRate { get; set; } = 1e-6;
    public new double MaxLearningRate { get; set; } = 1.0;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
    public double Tolerance { get; set; } = 1e-6;
    public new int MaxIterations { get; set; } = 1000;
    public new bool UseAdaptiveLearningRate { get; set; } = true;
}