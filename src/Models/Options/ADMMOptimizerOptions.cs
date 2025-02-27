namespace AiDotNet.Models.Options;

public class ADMMOptimizerOptions : GradientBasedOptimizerOptions
{
    public double Rho { get; set; } = 1.0;
    public double AbsoluteTolerance { get; set; } = 1e-4;
    public bool UseAdaptiveRho { get; set; } = true;
    public double AdaptiveRhoFactor { get; set; } = 10.0;
    public double AdaptiveRhoIncrease { get; set; } = 2.0;
    public double AdaptiveRhoDecrease { get; set; } = 2.0;
    public RegularizationType RegularizationType { get; set; } = RegularizationType.L1;
    public double RegularizationStrength { get; set; } = 0.1;
    public double ElasticNetMixing { get; set; } = 0.5;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;
}