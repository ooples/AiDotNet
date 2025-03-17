namespace AiDotNet.Models.Options;

public class TabuSearchOptions : GeneticAlgorithmOptimizerOptions
{
    public int TabuListSize { get; set; } = 10;
    public int NeighborhoodSize { get; set; } = 20;
    public double PerturbationFactor { get; set; } = 0.1;
    public new double MutationRate { get; set; } = 0.1;
    public double MinFeatureRatio { get; set; } = 0.1;
    public double MaxFeatureRatio { get; set; } = 0.9;
    public int InitialTabuListSize { get; set; } = 50;
    public int InitialNeighborhoodSize { get; set; } = 20;
    public double InitialMutationRate { get; set; } = 0.1;
    public new double MinMutationRate { get; set; } = 0.01;
    public new double MaxMutationRate { get; set; } = 0.5;
    public double TabuListSizeDecay { get; set; } = 0.95;
    public double TabuListSizeIncrease { get; set; } = 1.05;
    public int MinTabuListSize { get; set; } = 10;
    public int MaxTabuListSize { get; set; } = 100;
    public double NeighborhoodSizeDecay { get; set; } = 0.95;
    public double NeighborhoodSizeIncrease { get; set; } = 1.05;
    public int MinNeighborhoodSize { get; set; } = 5;
    public int MaxNeighborhoodSize { get; set; } = 50;
}