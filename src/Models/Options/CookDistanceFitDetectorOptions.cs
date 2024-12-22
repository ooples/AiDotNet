namespace AiDotNet.Models.Options;

public class CookDistanceFitDetectorOptions
{
    public double InfluentialThreshold { get; set; } = 4.0 / 100; // 4/n, where n is typically the sample size
    public double OverfitThreshold { get; set; } = 0.1; // 10% of points being influential suggests overfitting
    public double UnderfitThreshold { get; set; } = 0.01; // 1% of points being influential suggests underfitting
}