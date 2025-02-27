namespace AiDotNet.Models.Options;

public class ExtremelyRandomizedTreesRegressionOptions : DecisionTreeOptions
{
    public int NumberOfTrees { get; set; } = 100;
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}