namespace AiDotNet.Models;

public class OptimizationResult
{
    public Vector<double>? BestSolution { get; set; }
    public double FitnessScore { get; set; }
    public int Iterations { get; set; }
    public Vector<double>? FitnessHistory { get; set; }
}