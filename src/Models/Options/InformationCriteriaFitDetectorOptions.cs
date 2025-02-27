namespace AiDotNet.Models.Options;

public class InformationCriteriaFitDetectorOptions
{
    public double AicThreshold { get; set; } = 2.0;
    public double BicThreshold { get; set; } = 2.0;
    public double OverfitThreshold { get; set; } = 0.1;
    public double UnderfitThreshold { get; set; } = 0.1;
    public double HighVarianceThreshold { get; set; } = 0.2;
}