namespace AiDotNet.Models.Options;

public class HeteroscedasticityFitDetectorOptions
{
    public double HeteroscedasticityThreshold { get; set; } = 0.05; // p-value threshold for heteroscedasticity
    public double HomoscedasticityThreshold { get; set; } = 0.1; // p-value threshold for homoscedasticity
}