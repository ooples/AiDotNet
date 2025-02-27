namespace AiDotNet.Models.Options;

public class NonLinearRegressionOptions : ModelOptions
{
    public int MaxIterations { get; set; } = 1000;
    public double Tolerance { get; set; } = 1e-3;
    public KernelType KernelType { get; set; } = KernelType.RBF;
    public double Gamma { get; set; } = 1.0;
    public double Coef0 { get; set; } = 0.0;
    public int PolynomialDegree { get; set; } = 3;
}