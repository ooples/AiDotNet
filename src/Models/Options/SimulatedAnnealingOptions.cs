using AiDotNet.Models.Options;

namespace AiDotNet.Models;

public class SimulatedAnnealingOptions : OptimizationAlgorithmOptions
{
    public double InitialTemperature { get; set; } = 100.0;
    public double CoolingRate { get; set; } = 0.995;
}