namespace AiDotNet.Models;

public class ModelStatsOptions
{
    public ConditionNumberMethod ConditionNumberMethod { get; set; } = ConditionNumberMethod.SVD;
    public double MulticollinearityThreshold { get; set; } = 0.8;
    public int MaxVIF { get; set; } = 10;
}