namespace AiDotNet.Models.Options;

public class ConditionalInferenceTreeOptions : DecisionTreeOptions
{
    public double SignificanceLevel { get; set; } = 0.05;
    public TestStatisticType StatisticalTest { get; set; } = TestStatisticType.TTest;
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}