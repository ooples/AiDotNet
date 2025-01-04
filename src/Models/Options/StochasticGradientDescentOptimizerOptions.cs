namespace AiDotNet.Models.Options;

public class StochasticGradientDescentOptimizerOptions : GradientBasedOptimizerOptions
{
    public new int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-6;
}