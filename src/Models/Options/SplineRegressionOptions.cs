namespace AiDotNet.Models.Options;

public class SplineRegressionOptions : NonLinearRegressionOptions
{
    public int NumberOfKnots { get; set; } = 3;
    public int Degree { get; set; } = 3;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}