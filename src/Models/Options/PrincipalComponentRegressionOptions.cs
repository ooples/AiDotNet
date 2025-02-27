namespace AiDotNet.Models.Options;

public class PrincipalComponentRegressionOptions<T> : RegressionOptions<T>
{
    public int NumComponents { get; set; } = 0;
    public double ExplainedVarianceRatio { get; set; } = 0.95;
}