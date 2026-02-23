namespace AiDotNet.Audio.Classification;

/// <summary>
/// Features extracted for genre classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> GenreFeatures provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class GenreFeatures
{
    /// <summary>Mean of MFCC coefficients across time.</summary>
    public required double[] MfccMean { get; init; }

    /// <summary>Standard deviation of MFCC coefficients across time.</summary>
    public required double[] MfccStd { get; init; }

    /// <summary>Mean spectral centroid.</summary>
    public double SpectralCentroidMean { get; init; }

    /// <summary>Standard deviation of spectral centroid.</summary>
    public double SpectralCentroidStd { get; init; }

    /// <summary>Mean spectral rolloff.</summary>
    public double SpectralRolloffMean { get; init; }

    /// <summary>Zero crossing rate.</summary>
    public double ZeroCrossingRate { get; init; }

    /// <summary>RMS energy.</summary>
    public double RmsEnergy { get; init; }

    /// <summary>Estimated tempo in BPM.</summary>
    public double Tempo { get; init; }
}
