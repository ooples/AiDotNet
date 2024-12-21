namespace AiDotNet.Models;

public class JackknifeFitDetectorOptions
{
    public int MinSampleSize { get; set; } = 30;
    public double OverfitThreshold { get; set; } = 0.1;
    public double UnderfitThreshold { get; set; } = 0.1;
    public int MaxIterations { get; set; } = 1000;
}