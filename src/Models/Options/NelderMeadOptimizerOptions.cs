namespace AiDotNet.Models.Options;

public class NelderMeadOptimizerOptions : OptimizationAlgorithmOptions
{
    public double InitialAlpha { get; set; } = 1.0;
    public double InitialBeta { get; set; } = 0.5;
    public double InitialGamma { get; set; } = 2.0;
    public double InitialDelta { get; set; } = 0.5;

    public double MinAlpha { get; set; } = 0.1;
    public double MaxAlpha { get; set; } = 2.0;
    public double MinBeta { get; set; } = 0.1;
    public double MaxBeta { get; set; } = 1.0;
    public double MinGamma { get; set; } = 1.0;
    public double MaxGamma { get; set; } = 3.0;
    public double MinDelta { get; set; } = 0.1;
    public double MaxDelta { get; set; } = 1.0;

    public bool UseAdaptiveParameters { get; set; } = false;
    public double AdaptationRate { get; set; } = 0.1;
}