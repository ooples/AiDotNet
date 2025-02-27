namespace AiDotNet.Models;

public class ROCCurveFitDetectorOptions
{
    public double GoodFitThreshold { get; set; } = 0.8;
    public double ModerateFitThreshold { get; set; } = 0.7;
    public double PoorFitThreshold { get; set; } = 0.6;
    public double ConfidenceScalingFactor { get; set; } = 1.0;
    public double BalancedDatasetThreshold { get; set; } = 0.5;
}