using AiDotNet.Models.Options;

namespace AiDotNet.Models;

public class WeightedRegressionOptions<T> : RegressionOptions<T>
{
    public int Order { get; set; } = 1;
    public Vector<T> Weights { get; set; } = Vector<T>.Empty();
}