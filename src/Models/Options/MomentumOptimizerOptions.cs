namespace AiDotNet.Models.Options;

public class MomentumOptimizerOptions : GradientBasedOptimizerOptions
{
    public new double InitialLearningRate { get; set; } = 0.01;
    public new double InitialMomentum { get; set; } = 0.9;
    public new double MinLearningRate { get; set; } = 1e-5;
    public new double MaxLearningRate { get; set; } = 0.1;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    public new bool UseAdaptiveMomentum { get; set; } = true;
    public new double MinMomentum { get; set; } = 0.5;
    public new double MaxMomentum { get; set; } = 0.99;
    public new double MomentumIncreaseFactor { get; set; } = 1.02;
    public new double MomentumDecreaseFactor { get; set; } = 0.98;
}