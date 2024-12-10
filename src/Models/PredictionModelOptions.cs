namespace AiDotNet.Models;

public class PredictionModelOptions
{
    public double TrainingSplitPercentage { get; set; } = 0.7;
    public double ValidationSplitPercentage { get; set; } = 0.15;
    public double TestingSplitPercentage { get; set; } = 0.15;
    public int RandomSeed { get; set; }
    public bool NormalizeBeforeFeatureSelection { get; set; } = true;
    public int MinimumFeatures { get; set; }
    public int MaximumFeatures { get; set; }
}