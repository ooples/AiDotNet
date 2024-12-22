namespace AiDotNet.Models;

public class TimeSeriesCrossValidationFitDetectorOptions
{
    public double OverfitThreshold { get; set; } = 1.2;
    public double UnderfitThreshold { get; set; } = 0.5;
    public double HighVarianceThreshold { get; set; } = 1.1;
    public double GoodFitThreshold { get; set; } = 0.8;
}