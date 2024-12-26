namespace AiDotNet.Models.Options;

public class GaussianProcessRegressionOptions : NonLinearRegressionOptions
{
    public double NoiseLevel { get; set; } = 1e-5;
    public bool OptimizeHyperparameters { get; set; } = false;
    public double LengthScale { get; set; } = 1.0;
    public double SignalVariance { get; set; } = 1.0;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}