namespace AiDotNet.Models;

public class WeightedRegressionOptions<T> : RegressionOptions
{
    public int Order { get; set; } = 1;
    public Vector<T> Weights { get; set; } = Vector<T>.Empty();
}