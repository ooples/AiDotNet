namespace AiDotNet.Models.Options;

public class LocallyWeightedRegressionOptions : NonLinearRegressionOptions
{
    public double Bandwidth { get; set; } = 1.0;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}