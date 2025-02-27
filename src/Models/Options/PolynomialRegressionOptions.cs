namespace AiDotNet.Models.Options;

public class PolynomialRegressionOptions<T> : RegressionOptions<T>
{
    public int Degree { get; set; } = 2;
}