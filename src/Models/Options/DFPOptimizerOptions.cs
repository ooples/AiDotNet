namespace AiDotNet.Models.Options;

public class DFPOptimizerOptions : GradientBasedOptimizerOptions
{
    public new double InitialLearningRate { get; set; } = 1.0;
    public new double MinLearningRate { get; set; } = 1e-6;
    public new double MaxLearningRate { get; set; } = 10.0;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
    public double Tolerance { get; set; } = 1e-6;
    public new int MaxIterations { get; set; } = 1000;
    public new bool UseAdaptiveLearningRate { get; set; } = true;
    public int MaxLineSearchIterations { get; set; } = 20;
    public double AdaptationRate { get; set; } = 0.1;
}