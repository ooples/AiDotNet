namespace AiDotNet.Models.Options;

public class AdagradOptimizerOptions : GradientBasedOptimizerOptions
{
    public new double InitialLearningRate { get; set; } = 0.01;
    public double Epsilon { get; set; } = 1e-8;
    public double Tolerance { get; set; } = 1e-4;

    public new bool UseAdaptiveLearningRate { get; set; } = true;
    public new double MinLearningRate { get; set; } = 1e-6;
    public new double MaxLearningRate { get; set; } = 1.0;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
}