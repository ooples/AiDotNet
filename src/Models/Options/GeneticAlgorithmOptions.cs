namespace AiDotNet.Models.Options;

public class GeneticAlgorithmOptimizerOptions : OptimizationAlgorithmOptions
{
    public int MaxGenerations { get; set; } = 50;
    public int PopulationSize { get; set; } = 100;
    public double MutationRate { get; set; } = 0.01;
    public double CrossoverRate { get; set; } = 0.8;
}