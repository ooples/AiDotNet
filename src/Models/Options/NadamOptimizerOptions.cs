namespace AiDotNet.Models.Options;

public class NadamOptimizerOptions : GradientBasedOptimizerOptions
{
    public double LearningRate { get; set; } = 0.002;
    public double Beta1 { get; set; } = 0.9;
    public double Beta2 { get; set; } = 0.999;
    public double Epsilon { get; set; } = 1e-8;
    public new bool UseAdaptiveLearningRate { get; set; } = true;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
    public new double MinLearningRate { get; set; } = 1e-5;
    public new double MaxLearningRate { get; set; } = 0.1;
    public double Tolerance { get; set; } = 1e-6;
}