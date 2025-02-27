namespace AiDotNet.Models.Options;

public class LearningCurveFitDetectorOptions
{
    public double ConvergenceThreshold { get; set; } = 0.01;
    public int MinDataPoints { get; set; } = 5;
}