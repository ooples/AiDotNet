namespace AiDotNet.Models.Options;

public class FitnessCalculatorOptions
{
    public FitnessScoreType ScoreType { get; set; } = FitnessScoreType.RSquared;
    public bool UseMaximumValue { get; set; } = true;
}