namespace AiDotNet.Models;

public class AntColonyOptimizationOptions : OptimizationAlgorithmOptions
{
    public int AntCount { get; set; } = 50;
    public double EvaporationRate { get; set; } = 0.1;
    public double Alpha { get; set; } = 1.0;
    public double Beta { get; set; } = 2.0;
}