namespace AiDotNet.Models.Options;

public class StochasticGradientDescentOptimizerOptions : GradientBasedOptimizerOptions
{
    public new int MaxIterations { get; set; } = 1000;
}