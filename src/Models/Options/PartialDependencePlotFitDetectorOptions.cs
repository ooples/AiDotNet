namespace AiDotNet.Models.Options;

public class PartialDependencePlotFitDetectorOptions
{
    public double OverfitThreshold { get; set; } = 0.8;
    public double UnderfitThreshold { get; set; } = 0.2;
    public int NumPoints { get; set; } = 100;
}