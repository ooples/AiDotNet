namespace AiDotNet.Models.Options;

public class CMAESOptimizerOptions<T> : OptimizationAlgorithmOptions
{
    public new int PopulationSize { get; set; } = 4 + (int)(3 * Math.Log(100)); // Default assuming 100 dimensions
    public double InitialStepSize { get; set; } = 0.5;
    public int MaxGenerations { get; set; } = 100;
    public double StopTolerance { get; set; } = 1e-12;
    public int Seed { get; set; } = 42;
}