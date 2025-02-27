namespace AiDotNet.Models.Options;

public class SymbolicRegressionOptions : NonLinearRegressionOptions
{
    public int PopulationSize { get; set; } = 100;
    public int MaxGenerations { get; set; } = 1000;
    public double MutationRate { get; set; } = 0.1;
    public double CrossoverRate { get; set; } = 0.8;
    public double FitnessThreshold { get; set; } = 0.001;
}