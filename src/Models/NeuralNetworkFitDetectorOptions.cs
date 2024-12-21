namespace AiDotNet.Models;

public class NeuralNetworkFitDetectorOptions
{
    public double GoodFitThreshold { get; set; } = 0.05;
    public double ModerateFitThreshold { get; set; } = 0.1;
    public double PoorFitThreshold { get; set; } = 0.2;
    public double OverfittingThreshold { get; set; } = 0.2;
}