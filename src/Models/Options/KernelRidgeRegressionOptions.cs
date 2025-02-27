namespace AiDotNet.Models.Options;

public class KernelRidgeRegressionOptions : NonLinearRegressionOptions
{
    public double LambdaKRR { get; set; } = 1.0;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}