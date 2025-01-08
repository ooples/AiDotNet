namespace AiDotNet.Models.Options;

public class GradientBoostingFitDetectorOptions
{
    public double OverfitThreshold { get; set; } = 0.05;
    public double SevereOverfitThreshold { get; set; } = 0.1;
    public double GoodFitThreshold { get; set; } = 0.1;
}