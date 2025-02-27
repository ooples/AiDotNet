namespace AiDotNet.Models.Options;

public class PartialLeastSquaresRegressionOptions<T> : RegressionOptions<T>
{
    public int NumComponents { get; set; } = 2;
}