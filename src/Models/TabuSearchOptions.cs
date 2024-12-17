namespace AiDotNet.Models;

public class TabuSearchOptions : OptimizationAlgorithmOptions
{
    public int TabuListSize { get; set; } = 10;
    public int NeighborhoodSize { get; set; } = 20;
    public double PerturbationFactor { get; set; } = 0.1;
}