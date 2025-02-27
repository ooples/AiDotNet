namespace AiDotNet.Models.Options;

public class EnsembleFitDetectorOptions
{
    public List<double> DetectorWeights { get; set; } = [];
    public int MaxRecommendations { get; set; } = 5;
}