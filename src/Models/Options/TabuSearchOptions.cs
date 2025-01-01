namespace AiDotNet.Models.Options;

public class TabuSearchOptions : OptimizationAlgorithmOptions
{
    public int TabuListSize { get; set; } = 10;
    public int NeighborhoodSize { get; set; } = 20;
    public double PerturbationFactor { get; set; } = 0.1;
    public double MutationRate { get; set; } = 0.1;
    public double MinFeatureRatio { get; set; } = 0.1;
    public double MaxFeatureRatio { get; set; } = 0.9;
}