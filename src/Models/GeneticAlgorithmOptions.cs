namespace AiDotNet.Models;

public class GeneticAlgorithmOptions : OptimizationAlgorithmOptions
{
    public int PopulationSize { get; set; } = 100;
    public double MutationRate { get; set; } = 0.01;
    public double CrossoverRate { get; set; } = 0.8;
}