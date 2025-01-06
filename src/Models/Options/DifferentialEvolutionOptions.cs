namespace AiDotNet.Models.Options;

public class DifferentialEvolutionOptions : GeneticAlgorithmOptimizerOptions
{
    public new double CrossoverRate { get; set; } = 0.5;
    public new double MutationRate { get; set; } = 0.8;
    public new int PopulationSize { get; set; } = 50;
}