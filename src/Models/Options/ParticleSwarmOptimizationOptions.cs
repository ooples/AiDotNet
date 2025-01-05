namespace AiDotNet.Models.Options;

public class ParticleSwarmOptimizationOptions : OptimizationAlgorithmOptions
{
    public int SwarmSize { get; set; } = 50;
    public double InertiaWeight { get; set; } = 0.729;
    public double CognitiveParameter { get; set; } = 1.49445;
    public double SocialParameter { get; set; } = 1.49445;
    public bool UseAdaptiveInertia { get; set; } = false;
    public bool UseAdaptiveWeights { get; set; } = false;
    public double InitialInertia { get; set; } = 0.7;
    public double MinInertia { get; set; } = 0.1;
    public double MaxInertia { get; set; } = 0.9;
    public double InitialCognitiveWeight { get; set; } = 1.5;
    public double InitialSocialWeight { get; set; } = 1.5;
    public double MinCognitiveWeight { get; set; } = 0.5;
    public double MaxCognitiveWeight { get; set; } = 2.5;
    public double MinSocialWeight { get; set; } = 0.5;
    public double MaxSocialWeight { get; set; } = 2.5;
    public double InertiaDecayRate { get; set; } = 0.99;
    public double CognitiveWeightAdaptationRate { get; set; } = 1.0;
    public double SocialWeightAdaptationRate { get; set; } = 1.0;
}