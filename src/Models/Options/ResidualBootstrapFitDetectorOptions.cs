namespace AiDotNet.Models;

public class ResidualBootstrapFitDetectorOptions
{
    public int NumBootstrapSamples { get; set; } = 1000;
    public int MinSampleSize { get; set; } = 30;
    public double OverfitThreshold { get; set; } = 1.96; // ~95% confidence level
    public double UnderfitThreshold { get; set; } = -1.96; // ~95% confidence level
    public int? Seed { get; set; } = null;
}