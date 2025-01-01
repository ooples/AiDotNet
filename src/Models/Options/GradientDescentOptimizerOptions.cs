namespace AiDotNet.Models.Options;

public class GradientDescentOptimizerOptions : OptimizationAlgorithmOptions
{
    public double LearningRate { get; set; } = 0.01;
    public double Tolerance { get; set; } = 1e-6;

    private RegularizationOptions _regularizationOptions;

    public RegularizationOptions RegularizationOptions
    {
        get => _regularizationOptions;
        set => _regularizationOptions = value ?? CreateDefaultRegularizationOptions();
    }

    public GradientDescentOptimizerOptions()
    {
        _regularizationOptions = CreateDefaultRegularizationOptions();
    }

    private static RegularizationOptions CreateDefaultRegularizationOptions()
    {
        // we override the default regularization values to use variables more friendly to gradient descent
        return new RegularizationOptions
        {
            Type = RegularizationType.L2,
            Strength = 0.01,
            L1Ratio = 0.0
        };
    }
}