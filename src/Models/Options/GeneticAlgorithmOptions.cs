namespace AiDotNet.Models.Options;

public class GeneticAlgorithmOptimizerOptions : OptimizationAlgorithmOptions
{
    public int MaxGenerations { get; set; } = 50;
    public int PopulationSize { get; set; } = 100;
    public double MutationRate { get; set; } = 0.01;
    public double CrossoverRate { get; set; } = 0.8;


    // Crossover rate adaptation parameters
    public double CrossoverRateDecay { get; set; } = 0.95;
    public double CrossoverRateIncrease { get; set; } = 1.05;
    public double MinCrossoverRate { get; set; } = 0.5;
    public double MaxCrossoverRate { get; set; } = 0.95;

    
    // Mutation rate adaptation parameters
    public double MutationRateDecay { get; set; } = 0.95;
    public double MutationRateIncrease { get; set; } = 1.05;
    public double MinMutationRate { get; set; } = 0.001;
    public double MaxMutationRate { get; set; } = 0.1;
}