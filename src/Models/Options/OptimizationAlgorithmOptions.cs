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

    public double MutationRate { get; set; } = 0.01;
    public double ExplorationRate { get; set; } = 0.5;
    public int PopulationSize { get; set; } = 100;
    public double CrossoverRate { get; set; } = 0.7;

    public double MinExplorationRate { get; set; } = 0.1;
    public double MaxExplorationRate { get; set; } = 0.9;
    public double MinMutationRate { get; set; } = 0.001;
    public double MaxMutationRate { get; set; } = 0.1;
    public double MinCrossoverRate { get; set; } = 0.1;
    public double MaxCrossoverRate { get; set; } = 0.9;
    public int MinPopulationSize { get; set; } = 10;
    public int MaxPopulationSize { get; set; } = 1000;
}