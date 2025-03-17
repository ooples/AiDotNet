namespace AiDotNet.Models.Options;

public class LevenbergMarquardtOptimizerOptions<T> : GradientBasedOptimizerOptions
{
    public double InitialDampingFactor { get; set; } = 0.1;
    public double DampingFactorIncreaseFactor { get; set; } = 10.0;
    public double DampingFactorDecreaseFactor { get; set; } = 0.1;
    public double MinDampingFactor { get; set; } = 1e-8;
    public double MaxDampingFactor { get; set; } = 1e8;
    public bool UseAdaptiveDampingFactor { get; set; } = true;
    public IMatrixDecomposition<T>? CustomDecomposition { get; set; }
}