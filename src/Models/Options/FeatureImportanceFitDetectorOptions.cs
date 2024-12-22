namespace AiDotNet.Models.Options;

public class FeatureImportanceFitDetectorOptions
{
    /// <summary>
    /// Threshold for considering feature importance as high.
    /// </summary>
    public double HighImportanceThreshold { get; set; } = 0.1;

    /// <summary>
    /// Threshold for considering feature importance as low.
    /// </summary>
    public double LowImportanceThreshold { get; set; } = 0.01;

    /// <summary>
    /// Threshold for considering importance variance as low.
    /// </summary>
    public double LowVarianceThreshold { get; set; } = 0.05;

    /// <summary>
    /// Threshold for considering importance variance as high.
    /// </summary>
    public double HighVarianceThreshold { get; set; } = 0.2;

    /// <summary>
    /// Threshold for considering features as correlated.
    /// </summary>
    public double CorrelationThreshold { get; set; } = 0.7;

    /// <summary>
    /// Threshold for the ratio of uncorrelated feature pairs to consider features as mostly uncorrelated.
    /// </summary>
    public double UncorrelatedRatioThreshold { get; set; } = 0.8;

    /// <summary>
    /// Random seed for feature permutation.
    /// </summary>
    public int RandomSeed { get; set; } = 42;

    /// <summary>
    /// Number of permutations to perform for each feature when calculating importance.
    /// </summary>
    public int NumPermutations { get; set; } = 5;
}