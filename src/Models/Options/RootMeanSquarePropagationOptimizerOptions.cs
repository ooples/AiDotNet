namespace AiDotNet.Models.Options;

public class RootMeanSquarePropagationOptimizerOptions : GradientBasedOptimizerOptions
{
    public double Decay { get; set; } = 0.9;
    public double Epsilon { get; set; } = 1e-8;
    public double Tolerance { get; set; } = 1e-4;
}