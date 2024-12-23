namespace AiDotNet.Models.Options;

public class GradientBoostingRegressionOptions : DecisionTreeOptions
{
    public int NumberOfTrees { get; set; } = 100;
    public double LearningRate { get; set; } = 0.1;
    public double SubsampleRatio { get; set; } = 1.0;
    public FitnessCalculatorType FitnessCalculatorType { get; set; } = FitnessCalculatorType.MeanSquaredError;
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}