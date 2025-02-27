namespace AiDotNet.Models.Options;

public class DecisionTreeOptions
{
    public int MaxDepth { get; set; } = 10;
    public int MinSamplesSplit { get; set; } = 2;
    public double MaxFeatures { get; set; } = 1.0;
    public int? Seed { get; set; }
    public SplitCriterion SplitCriterion { get; set; } = SplitCriterion.VarianceReduction;
}