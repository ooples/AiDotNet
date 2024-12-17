namespace AiDotNet.Models;

public class ParticleSwarmOptimizationOptions : OptimizationAlgorithmOptions
{
    public int SwarmSize { get; set; } = 50;
    public double InertiaWeight { get; set; } = 0.729;
    public double CognitiveParameter { get; set; } = 1.49445;
    public double SocialParameter { get; set; } = 1.49445;
}