namespace AiDotNet.Models;

public class PredictionModelOptions
{
    public double TrainingSplitPercentage { get; set; } = 0.7;
    public double ValidationSplitPercentage { get; set; } = 0.15;
    public double TestingSplitPercentage { get; set; } = 0.15;
    public int RandomSeed { get; set; }
    public double ConfidenceLevel { get; set; } = 0.95;
    public int LearningCurveSteps { get; set; } = 10;
}