namespace AiDotNet.Models.Options;

public class GeneralizedAdditiveModelOptions<T> : RegressionOptions<T>
{
    public int NumSplines { get; set; } = 10;
    public int Degree { get; set; } = 3;
}