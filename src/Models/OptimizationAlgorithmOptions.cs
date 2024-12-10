namespace AiDotNet.Models;

public class OptimizationAlgorithmOptions
{
    public int MaxIterations { get; set; } = 100;
    public bool MaximizeFitness { get; set; } = true;
    public bool CalculatePredictionIntervals { get; set; } = true;
    public double ConfidenceLevel { get; set; } = 0.95;
    public bool UseEarlyStopping { get; set; } = true;
    public int EarlyStoppingPatience { get; set; } = 10;
    public int EarlyStoppingMinDelta { get; set; } = 0;
    public int BadFitPatience { get; set; } = 5;
}