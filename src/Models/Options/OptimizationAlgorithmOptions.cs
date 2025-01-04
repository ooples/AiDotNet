namespace AiDotNet.Models.Options;

public class OptimizationAlgorithmOptions
{
    public int MaxIterations { get; set; } = 100;
    public bool UseEarlyStopping { get; set; } = true;
    public int EarlyStoppingPatience { get; set; } = 10;
    public int BadFitPatience { get; set; } = 5;
    public int MinimumFeatures { get; set; }
    public int MaximumFeatures { get; set; }
    public bool UseExpressionTrees { get; set; } = false;

    public double InitialLearningRate { get; set; } = 0.01;
    public bool UseAdaptiveLearningRate { get; set; } = true;
    public double LearningRateDecay { get; set; } = 0.99;
    public double MinLearningRate { get; set; } = 1e-6;
    public double MaxLearningRate { get; set; } = 1.0;
    
    public bool UseAdaptiveMomentum { get; set; } = true;
    public double InitialMomentum { get; set; } = 0.9;
    public double MomentumIncreaseFactor { get; set; } = 1.05;
    public double MomentumDecreaseFactor { get; set; } = 0.95;
    public double MinMomentum { get; set; } = 0.5;
    public double MaxMomentum { get; set; } = 0.99;
}