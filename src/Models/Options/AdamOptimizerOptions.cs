namespace AiDotNet.Models.Options;

public class AdamOptimizerOptions : GradientBasedOptimizerOptions
{
    public double LearningRate { get; set; } = 0.001;
    public double Beta1 { get; set; } = 0.9;
    public double Beta2 { get; set; } = 0.999;
    public double Epsilon { get; set; } = 1e-8;
    public double Tolerance { get; set; } = 1e-4;


    public bool UseAdaptiveBetas { get; set; } = true;
    public double MinBeta1 { get; set; } = 0.8;
    public double MaxBeta1 { get; set; } = 0.999;
    public double MinBeta2 { get; set; } = 0.8;
    public double MaxBeta2 { get; set; } = 0.9999;
}