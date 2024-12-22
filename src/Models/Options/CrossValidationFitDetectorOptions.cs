namespace AiDotNet.Models.Options;

public class CrossValidationFitDetectorOptions
{
    public double OverfitThreshold { get; set; } = 0.1;
    public double UnderfitThreshold { get; set; } = 0.7;
    public double GoodFitThreshold { get; set; } = 0.9;
}