namespace AiDotNet.Models.Options;

public class BayesianFitDetectorOptions
{
    public double GoodFitThreshold { get; set; } = 5;
    public double OverfitThreshold { get; set; } = 10;
    public double UnderfitThreshold { get; set; } = 2;
}