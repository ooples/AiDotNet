namespace AiDotNet.Models.Options;

public class CoordinateDescentOptimizerOptions : GradientBasedOptimizerOptions
{
    public new double InitialLearningRate { get; set; } = 0.01;
    public new double MinLearningRate { get; set; } = 1e-6;
    public new double MaxLearningRate { get; set; } = 1.0;
    public double LearningRateIncreaseRate { get; set; } = 0.05;
    public double LearningRateDecreaseRate { get; set; } = 0.05;

    public new double InitialMomentum { get; set; } = 0.9;
    public new double MinMomentum { get; set; } = 0.5;
    public new double MaxMomentum { get; set; } = 0.99;
    public double MomentumIncreaseRate { get; set; } = 0.01;
    public double MomentumDecreaseRate { get; set; } = 0.01;

    public double Tolerance { get; set; } = 1e-6;
}