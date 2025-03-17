namespace AiDotNet.Models.Options;

public class RootMeanSquarePropagationOptimizerOptions : GradientBasedOptimizerOptions
{
    public double Decay { get; set; } = 0.9;
    public double Epsilon { get; set; } = 1e-8;
}