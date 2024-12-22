namespace AiDotNet.Models.Options;

public class DataProcessorOptions
{
    public double TrainingSplitPercentage { get; set; } = 0.7;
    public double ValidationSplitPercentage { get; set; } = 0.15;
    public double TestingSplitPercentage { get; set; } = 0.15;
    public int RandomSeed { get; set; } = 42;
    public bool NormalizeBeforeFeatureSelection { get; set; } = true;
}