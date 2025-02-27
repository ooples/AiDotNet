namespace AiDotNet.Models.Options;

public class KFoldCrossValidationFitDetectorOptions
{
    public double OverfitThreshold { get; set; } = 0.1;
    public double UnderfitThreshold { get; set; } = 0.5;
    public double HighVarianceThreshold { get; set; } = 0.1;
    public double GoodFitThreshold { get; set; } = 0.7;
    public double StabilityThreshold { get; set; } = 0.05;
}