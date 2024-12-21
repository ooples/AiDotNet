namespace AiDotNet.Models;

public class GaussianProcessFitDetectorOptions
{
    public double GoodFitThreshold { get; set; } = 0.1;
    public double OverfitThreshold { get; set; } = 0.2;
    public double UnderfitThreshold { get; set; } = 0.3;
    public double LowUncertaintyThreshold { get; set; } = 0.1;
    public double HighUncertaintyThreshold { get; set; } = 0.5;
    public double LengthScale { get; set; } = 1.0;
    public double NoiseVariance { get; set; } = 0.1;
}