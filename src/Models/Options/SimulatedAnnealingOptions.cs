namespace AiDotNet.Models.Options;

public class SimulatedAnnealingOptions : OptimizationAlgorithmOptions
{
    public double InitialTemperature { get; set; } = 100.0;
    public double CoolingRate { get; set; } = 0.995;
    public double MinTemperature { get; set; } = 1e-8;
    public double MaxTemperature { get; set; } = 1000.0;
    public new int MaxIterations { get; set; } = 10000;
    public int MaxIterationsWithoutImprovement { get; set; } = 1000;
    public double NeighborGenerationRange { get; set; } = 0.1;
    public double MinNeighborGenerationRange { get; set; } = 0.001;
    public double MaxNeighborGenerationRange { get; set; } = 1.0;
}