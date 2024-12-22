namespace AiDotNet.Models.Options;

public class RandomForestRegressionOptions<T> : RegressionOptions<T>
{
    public int NumberOfTrees { get; set; } = 100;
    public int MaxDepth { get; set; } = 10;
    public int MinSamplesSplit { get; set; } = 2;
    public double MaxFeatures { get; set; } = 1.0;
    public int? Seed { get; set; }
}