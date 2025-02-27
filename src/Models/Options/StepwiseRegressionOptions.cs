namespace AiDotNet.Models.Options;

public class StepwiseRegressionOptions<T> : RegressionOptions<T>
{
    public StepwiseMethod Method { get; set; } = StepwiseMethod.Forward;
    public int MaxFeatures { get; set; } = int.MaxValue;
    public int MinFeatures { get; set; } = 1;
    public double MinImprovement { get; set; } = 0.001;
}