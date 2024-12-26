namespace AiDotNet.Models.Options;

public class SupportVectorRegressionOptions : NonLinearRegressionOptions
{
    public double C { get; set; } = 1.0;
    public double Epsilon { get; set; } = 0.1;
}