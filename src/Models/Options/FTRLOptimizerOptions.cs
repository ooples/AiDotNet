namespace AiDotNet.Models.Options;

public class FTRLOptimizerOptions : GradientBasedOptimizerOptions
{
    public double Alpha { get; set; } = 0.005;
    public double Beta { get; set; } = 1.0;
    public double Lambda1 { get; set; } = 1.0;
    public double Lambda2 { get; set; } = 1.0;
    public new double MinLearningRate { get; set; } = 1e-6;
    public new double MaxLearningRate { get; set; } = 0.1;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
    public double Tolerance { get; set; } = 1e-6;
    public new int MaxIterations { get; set; } = 1000;
    public new bool UseAdaptiveLearningRate { get; set; } = true;
}