namespace AiDotNet.Models.Options;

public class BootstrapFitDetectorOptions
{
    public int NumberOfBootstraps { get; set; } = 1000;
    public double ConfidenceInterval { get; set; } = 0.95;
    public double OverfitThreshold { get; set; } = 0.1;
    public double UnderfitThreshold { get; set; } = 0.7;
    public double GoodFitThreshold { get; set; } = 0.9;
}