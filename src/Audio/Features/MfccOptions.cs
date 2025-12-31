namespace AiDotNet.Audio.Features;

/// <summary>
/// Options for MFCC extraction.
/// </summary>
public class MfccOptions : AudioFeatureOptions
{
    /// <summary>
    /// Gets or sets the number of MFCC coefficients to compute.
    /// Default is 13 (standard for speech recognition).
    /// </summary>
    public int NumCoefficients { get; set; } = 13;

    /// <summary>
    /// Gets or sets the number of Mel filterbank channels.
    /// Default is 40.
    /// </summary>
    public int NumMels { get; set; } = 40;

    /// <summary>
    /// Gets or sets whether to replace the first coefficient with log energy.
    /// </summary>
    public bool IncludeEnergy { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to append delta (velocity) coefficients.
    /// </summary>
    public bool AppendDelta { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to append delta-delta (acceleration) coefficients.
    /// Requires AppendDelta to be true.
    /// </summary>
    public bool AppendDeltaDelta { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum frequency for the mel filterbank.
    /// </summary>
    public double FMin { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum frequency for the mel filterbank.
    /// Null means use Nyquist frequency (SampleRate / 2).
    /// </summary>
    public double? FMax { get; set; }
}
