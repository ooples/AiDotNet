namespace AiDotNet.Models.Options;

public class AntColonyOptimizationOptions : OptimizationAlgorithmOptions
{
    public int AntCount { get; set; } = 50;
    public double Beta { get; set; } = 2.0;
    public double InitialPheromoneEvaporationRate { get; set; } = 0.1;
    public double InitialPheromoneIntensity { get; set; } = 1.0;


    // Pheromone evaporation rate adaptation parameters
    public double PheromoneEvaporationRateDecay { get; set; } = 0.95;
    public double PheromoneEvaporationRateIncrease { get; set; } = 1.05;
    public double MinPheromoneEvaporationRate { get; set; } = 0.01;
    public double MaxPheromoneEvaporationRate { get; set; } = 0.5;


    // Pheromone intensity adaptation parameters
    public double PheromoneIntensityDecay { get; set; } = 0.95;
    public double PheromoneIntensityIncrease { get; set; } = 1.05;
    public double MinPheromoneIntensity { get; set; } = 0.1;
    public double MaxPheromoneIntensity { get; set; } = 10.0;
}