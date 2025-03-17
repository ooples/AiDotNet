namespace AiDotNet.Models.Options;

public class TrustRegionOptimizerOptions : GradientBasedOptimizerOptions
{
    public double InitialTrustRegionRadius { get; set; } = 1.0;
    public double MinTrustRegionRadius { get; set; } = 1e-6;
    public double MaxTrustRegionRadius { get; set; } = 10.0;
    public double AcceptanceThreshold { get; set; } = 0.1;
    public double VerySuccessfulThreshold { get; set; } = 0.75;
    public double UnsuccessfulThreshold { get; set; } = 0.25;
    public double ExpansionFactor { get; set; } = 2.0;
    public double ContractionFactor { get; set; } = 0.5;
    public bool UseAdaptiveTrustRegionRadius { get; set; } = true;
    public double AdaptationRate { get; set; } = 0.1;
    public int MaxCGIterations { get; set; } = 100;
    public double CGTolerance { get; set; } = 1e-6;
}