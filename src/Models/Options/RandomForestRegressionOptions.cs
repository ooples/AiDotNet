namespace AiDotNet.Models.Options;

public class RandomForestRegressionOptions : DecisionTreeOptions
{
    public int NumberOfTrees { get; set; } = 100;
}