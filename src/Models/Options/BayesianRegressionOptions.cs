namespace AiDotNet.Models.Options;

public class BayesianRegressionOptions<T> : RegressionOptions<T>
{
    public double Alpha { get; set; } = 1.0;
    public double Beta { get; set; } = 1.0;
    public KernelType KernelType { get; set; } = KernelType.Linear;
    public double Gamma { get; set; } = 1.0;
    public double Coef0 { get; set; } = 0.0;
    public int PolynomialDegree { get; set; } = 3;
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;
}