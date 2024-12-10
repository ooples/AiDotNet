namespace AiDotNet.Models;

public class OptimizationIteration
{
    public int Iteration { get; set; }
    public double Fitness { get; set; }
    public FitDetectionResult FitDetectionResult { get; set; } = new();
}