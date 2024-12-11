namespace AiDotNet.Models;

public class OptimizationIteration
{
    public int Iteration { get; set; }
    public double Fitness { get; set; }
    public FitDetectorResult FitDetectionResult { get; set; } = new();
}