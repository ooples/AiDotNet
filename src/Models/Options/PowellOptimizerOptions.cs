namespace AiDotNet.Models.Options;

public class PowellOptimizerOptions : OptimizationAlgorithmOptions
{
    public double InitialStepSize { get; set; } = 0.1;
    public double MinStepSize { get; set; } = 1e-6;
    public double MaxStepSize { get; set; } = 1.0;
    public bool UseAdaptiveStepSize { get; set; } = true;
    public double AdaptationRate { get; set; } = 0.1;
}