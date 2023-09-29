namespace AiDotNet.Models;

public class RegressionOptions
{
    public double TrainingPctSize { get; set; } = 25;
    public INormalization? Normalization { get; set; }
}