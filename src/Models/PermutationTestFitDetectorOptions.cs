namespace AiDotNet.Models;

public class PermutationTestFitDetectorOptions
{
    public int NumberOfPermutations { get; set; } = 1000;
    public double SignificanceLevel { get; set; } = 0.05;
    public double OverfitThreshold { get; set; } = 0.1;
    public double UnderfitThreshold { get; set; } = 0.05;
    public double HighVarianceThreshold { get; set; } = 0.1;
}