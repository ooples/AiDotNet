namespace AiDotNet.Models.Options;

public class OptimizationAlgorithmOptions
{
    public int MaxIterations { get; set; } = 100;
    public bool UseEarlyStopping { get; set; } = true;
    public int EarlyStoppingPatience { get; set; } = 10;
    public int BadFitPatience { get; set; } = 5;
    public int MinimumFeatures { get; set; }
    public int MaximumFeatures { get; set; }
    public bool UseExpressionTrees { get; set; } = false;
}