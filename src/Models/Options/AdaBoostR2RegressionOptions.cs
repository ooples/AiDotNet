namespace AiDotNet.Models.Options;

public class AdaBoostR2RegressionOptions : DecisionTreeOptions
{
    public int NumberOfEstimators { get; set; } = 50;
}