namespace AiDotNet.Models;

public class PolynomialRegressionOptions<T> : RegressionOptions<T>
{
    public int Degree { get; set; } = 2;
}