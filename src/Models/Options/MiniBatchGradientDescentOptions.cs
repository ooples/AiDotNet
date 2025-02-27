namespace AiDotNet.Models.Options;

public class MiniBatchGradientDescentOptions : GradientBasedOptimizerOptions
{
    public int BatchSize { get; set; } = 32;
    public int MaxEpochs { get; set; } = 100;
    public new double InitialLearningRate { get; set; } = 0.01;
    public new double MinLearningRate { get; set; } = 1e-6;
    public new double MaxLearningRate { get; set; } = 0.1;
    public double LearningRateIncreaseFactor { get; set; } = 1.05;
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
    public double Tolerance { get; set; } = 1e-6;
    public int? Seed { get; set; } = null;
}