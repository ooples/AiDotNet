namespace AiDotNet.Models.Options;

public class M5ModelTreeOptions : DecisionTreeOptions
{
    public int MinInstancesPerLeaf { get; set; } = 4;
    public double PruningFactor { get; set; } = 0.05;
    public bool UseLinearRegressionAtLeaves { get; set; } = true;
    public bool UsePruning { get; set; } = true;
    public double SmoothingConstant { get; set; } = 15.0;
}