namespace AiDotNet.Models.Options;

public class FitnessCalculatorOptions
{
    public FitnessCalculatorType ScoreType { get; set; } = FitnessCalculatorType.RSquared;
    public bool UseMaximumValue { get; set; } = true;
}