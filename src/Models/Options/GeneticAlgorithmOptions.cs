namespace AiDotNet.Models.Options;

public class GeneticAlgorithmOptimizerOptions : OptimizationAlgorithmOptions
{
    public int MaxGenerations { get; set; } = 50;
    public new int PopulationSize { get; set; } = 100;
    public new double MutationRate { get; set; } = 0.01;
    public new double CrossoverRate { get; set; } = 0.8;


    // Crossover rate adaptation parameters
    public double CrossoverRateDecay { get; set; } = 0.95;
    public double CrossoverRateIncrease { get; set; } = 1.05;
    public new double MinCrossoverRate { get; set; } = 0.5;
    public new double MaxCrossoverRate { get; set; } = 0.95;

    
    // Mutation rate adaptation parameters
    public double MutationRateDecay { get; set; } = 0.95;
    public double MutationRateIncrease { get; set; } = 1.05;
    public new double MinMutationRate { get; set; } = 0.001;
    public new double MaxMutationRate { get; set; } = 0.1;
}