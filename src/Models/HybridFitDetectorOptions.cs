namespace AiDotNet.Models;

public class HybridFitDetectorOptions
{
    public double OverfitThreshold { get; set; } = 0.2;
    public double UnderfitThreshold { get; set; } = 0.5;
    public double GoodFitThreshold { get; set; } = 0.8;
}