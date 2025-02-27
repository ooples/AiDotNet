namespace AiDotNet.Models.Options;

public class AdaDeltaOptimizerOptions : GradientBasedOptimizerOptions
{
    public new double InitialLearningRate { get; set; } = 1.0;
    public double Rho { get; set; } = 0.95;
    public double Epsilon { get; set; } = 1e-6;
    public bool UseAdaptiveRho { get; set; } = true;
    public double RhoIncreaseFactor { get; set; } = 1.01;
    public double RhoDecreaseFactor { get; set; } = 0.99;
    public double MinRho { get; set; } = 0.5;
    public double MaxRho { get; set; } = 0.9999;
    public double Tolerance { get; set; } = 1e-6;
}