namespace AiDotNet.Audio.Classification;

/// <summary>
/// Features extracted for scene classification.
/// </summary>
public class SceneFeatures
{
    public required double[] MfccMean { get; init; }
    public required double[] MfccStd { get; init; }
    public required double[] MfccDelta { get; init; }
    public double SpectralCentroid { get; init; }
    public double SpectralBandwidth { get; init; }
    public double SpectralFlatness { get; init; }
    public double SpectralContrast { get; init; }
    public double RmsEnergy { get; init; }
    public double ZeroCrossingRate { get; init; }
    public double EnergyVariance { get; init; }
    public required double[] BandEnergies { get; init; }
}
