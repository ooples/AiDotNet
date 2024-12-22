namespace AiDotNet.Models.Options;

public class DifferentialEvolutionOptions : OptimizationAlgorithmOptions
{
    public double CrossoverRate { get; set; } = 0.5;
    public double MutationFactor { get; set; } = 0.8;
    public int PopulationSize { get; set; } = 50;
}