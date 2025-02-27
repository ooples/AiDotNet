namespace AiDotNet.Models;

public class PredictionStatsOptions
{
    public double ConfidenceLevel { get; set; } = 0.95;
    public int LearningCurveSteps { get; set; } = 10;
}